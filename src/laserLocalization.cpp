// POINT-LIO LOCALIZATION against a prior map -- the Point-LIO twin of
// FAST_LIO/src/laserLocalization.cpp, which carries the full reasoning.
//
// Same idea, same map format: the prior map is loaded straight into the
// ikd-Tree the filter registers against, so the map constrains the estimate
// from INSIDE the filter at scan rate and there is no map->odom correction
// step to jump. ScanContext + two-stage ICP finds the initial pose, gated on
// agreement, map overlap and (optionally) motion.
//
// DELIBERATELY A COPY, not a shared library with FAST_LIO. The two backends
// differ where it matters -- Point-LIO carries two filter types (state_input
// with the IMU as input, state_output with it as measurement) and runs a
// point-by-point update -- and the pair is small enough that keeping them
// independent beats a common abstraction that has to straddle both.
//
// WHAT DIFFERS FROM THE FAST_LIO FILE:
//   * The handover writes into kf_input or kf_output depending on
//     use_imu_as_input. The world-frame quantities are the same four in both
//     (pos, rot, vel, gravity); omg and acc are BODY frame -- get_f_output
//     rotates acc by s.rot to get the inertial term, and the IMU residual
//     compares omg against the raw gyro -- so they carry over untouched, as
//     bg/ba do.
//   * gravity is a plain vect3 here, not FAST-LIO's S2 manifold, so it is
//     rotated directly.
//   * Point-LIO has no node class or timer: main() owns a 5 kHz spin loop, so
//     the health check is a time check inside that loop rather than a timer
//     callback, and this is a plain rclcpp::Node (see the lifecycle note in
//     the FAST_LIO file for what that costs).
#include <omp.h>
#include <mutex>
#include <cmath>
#include <thread>
#include <atomic>
#include <queue>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <csignal>
#include <Python.h>
#include <so3_math.h>
#include <rclcpp/rclcpp.hpp>
#include <Eigen/Core>
#include "IMU_Processing.hpp"

#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <std_msgs/msg/float32.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/registration/icp.h>
#include "Scancontext/Scancontext.h"
// #include <livox_ros_driver2/msg/custom_msg.hpp>

#include "parameters.h"
#include "Estimator.h"


#define MAXN                (720000)
#define PUBFRAME_PERIOD     (20)

const float MOV_THRESHOLD = 1.5f;

mutex mtx_buffer;
condition_variable sig_buffer;

/*** LOCALIZATION state -- everything below this point is the localization
 *** graft; the rest of the file is Point-LIO's mapping node unchanged. ***/
SCManager scManager;                        // ScanContext DB of the prior map
PointCloudXYZI::Ptr global_map(new PointCloudXYZI());
KD_TREE<PointType> ikdtree_global;          // prior map, swapped in on lock
pcl::KdTreeFLANN<PointType>::Ptr global_map_kdtree;
std::vector<V3D, Eigen::aligned_allocator<V3D>> position_map;
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pose_map;
// Odometry trail during init: a ScanContext match names a SCAN, not "now", so
// the pose it yields has to be carried forward by the odometry accumulated
// since -- hence the whole trail rather than just the latest.
std::vector<V3D, Eigen::aligned_allocator<V3D>> position_init;
std::vector<Eigen::Quaterniond, Eigen::aligned_allocator<Eigen::Quaterniond>> pose_init;
std::queue<std::pair<int, PointCloudXYZI::Ptr>> init_feats_down_bodys;
std::mutex init_feats_mutex, init_state_mutex;
// Sliding window of accepted candidates. Global so rearm_search() can drop it
// on /relocalize -- a candidate from before the rearm pairing with a fresh one
// after it would be a bogus instant "agreement".
std::vector<int> candidate_ids;
std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> candidate_poses;
std::mutex candidate_mutex;
bool global_localization_finish = false;    // a lock has been found
bool global_update = false;                 // the lock has been APPLIED
bool map_swapped = false;                   // ikdtree already holds the prior map
bool keep_searching = true;                 // cleared on shutdown to stop the thread
bool map_loaded = false;
int init_count = 0;
std::pair<int, Eigen::Matrix4d> init_result;
std::mutex seed_mutex;
Eigen::Matrix4d pending_seed = Eigen::Matrix4d::Identity();
std::atomic<bool> has_pending_seed{false};

std::string map_dir_param, map_pose_file_param, map_scan_dir_param;
int    init_agree_count = 2;
double init_agree_dist = 2.0;
double init_icp_coarse = 5.0, init_icp_fine = 1.0;
double sc_lidar_height = 0.5, sc_max_radius = 10.0, sc_dist_thres = 0.15;
int    sc_num_ring = 12, sc_num_sector = 40;
bool   init_require_motion = false;
double init_motion_min = 0.50;
double init_min_overlap = 0.70;
double init_overlap_dist = 0.20;
double prior_map_view_leaf = 0.20;
// Post-lock health check (see the FAST_LIO file): a wrong-but-self-consistent
// lock has no other symptom, because a self-similar place still produces
// plausible matches at the wrong place.
double health_min_overlap = 0.45;
double health_bad_duration = 5.0;
double health_check_period = 1.0;
bool   auto_relocalize = true;
bool   overlap_bad = false;
rclcpp::Time bad_since;
// map -> base_footprint, not map -> the IMU on the mast: REP-105, and the
// bag's /tf_static already owns base_footprint -> the IMU frame, so
// broadcasting that edge here too would give it two parents.
std::string tf_child_frame;
bool  tf_child_resolved = false;
M3D   R_body_to_tfchild(Eye3d);
V3D   t_body_to_tfchild(0, 0, 0);
std::shared_ptr<tf2_ros::Buffer> tf_buffer_g;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_g;
rclcpp::Node *node_g = nullptr;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_g;
rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_overlap_g;
rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_g;

static rclcpp::Logger this_logger() {
    return node_g ? node_g->get_logger() : rclcpp::get_logger("point_lio_localization");
}

string root_dir = ROOT_DIR;

int feats_down_size = 0, time_log_counter = 0, scan_count = 0, publish_count = 0;

int frame_ct = 0;
double time_update_last = 0.0, time_current = 0.0, time_predict_last_const = 0.0, t_last = 0.0;

shared_ptr<ImuProcess> p_imu(new ImuProcess());
bool init_map = false, flg_first_scan = true;
PointCloudXYZI::Ptr ptr_con(new PointCloudXYZI());

// Time Log Variables
double T1[MAXN], s_plot[MAXN], s_plot2[MAXN], s_plot3[MAXN], s_plot11[MAXN];
double match_time = 0, solve_time = 0, propag_time = 0, update_time = 0;

bool lidar_pushed = false, flg_reset = false, flg_exit = false;

vector<BoxPointType> cub_needrm;

deque<PointCloudXYZI::Ptr> lidar_buffer;
deque<double> time_buffer;
deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_deque;

//surf feature in map
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body_space(new PointCloudXYZI());
PointCloudXYZI::Ptr init_feats_world(new PointCloudXYZI());

pcl::VoxelGrid<PointType> downSizeFilterSurf;
pcl::VoxelGrid<PointType> downSizeFilterMap;

V3D euler_cur;

MeasureGroup Measures;

sensor_msgs::msg::Imu imu_last, imu_next;
sensor_msgs::msg::Imu::ConstSharedPtr imu_last_ptr;
nav_msgs::msg::Path path;
nav_msgs::msg::Odometry odomAftMapped;
geometry_msgs::msg::PoseStamped msg_body_pose;

auto logger = rclcpp::get_logger("laserMapping");

void SigHandle(int sig) {
    flg_exit = true;
    RCLCPP_WARN(logger, "catch sig %d", sig);
    sig_buffer.notify_all();
}

inline void dump_lio_state_to_log(FILE *fp) {
    V3D rot_ang;
    if (!use_imu_as_input) {
        rot_ang = SO3ToEuler(kf_output.x_.rot);
    } else {
        rot_ang = SO3ToEuler(kf_input.x_.rot);
    }

    fprintf(fp, "%lf ", Measures.lidar_beg_time - first_lidar_time);
    fprintf(fp, "%lf %lf %lf ", rot_ang(0), rot_ang(1), rot_ang(2));                   // Angle
    if (use_imu_as_input) {
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.pos(0), kf_input.x_.pos(1), kf_input.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.vel(0), kf_input.x_.vel(1), kf_input.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.bg(0), kf_input.x_.bg(1), kf_input.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.ba(0), kf_input.x_.ba(1), kf_input.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_input.x_.gravity(0), kf_input.x_.gravity(1), kf_input.x_.gravity(2)); // Bias_a  
    } else {
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.pos(0), kf_output.x_.pos(1), kf_output.x_.pos(2)); // Pos  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // omega  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.vel(0), kf_output.x_.vel(1), kf_output.x_.vel(2)); // Vel  
        fprintf(fp, "%lf %lf %lf ", 0.0, 0.0, 0.0);                                        // Acc  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.bg(0), kf_output.x_.bg(1), kf_output.x_.bg(2));    // Bias_g  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.ba(0), kf_output.x_.ba(1), kf_output.x_.ba(2));    // Bias_a  
        fprintf(fp, "%lf %lf %lf ", kf_output.x_.gravity(0), kf_output.x_.gravity(1),
                kf_output.x_.gravity(2)); // Bias_a
    }
    fprintf(fp, "\r\n");
    fflush(fp);
}

void pointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
    V3D p_body_lidar(pi->x, pi->y, pi->z);
    V3D p_body_imu;
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            p_body_imu = kf_output.x_.offset_R_L_I.normalized() * p_body_lidar + kf_output.x_.offset_T_L_I;
        } else {
            p_body_imu = kf_input.x_.offset_R_L_I.normalized() * p_body_lidar + kf_input.x_.offset_T_L_I;
        }
    } else {
        p_body_imu = Lidar_R_wrt_IMU * p_body_lidar + Lidar_T_wrt_IMU;
    }
    po->x = p_body_imu(0);
    po->y = p_body_imu(1);
    po->z = p_body_imu(2);
    po->intensity = pi->intensity;
}

int points_cache_size = 0;

void points_cache_collect() // seems for debug
{
    PointVector points_history;
    ikdtree.acquire_removed_points(points_history);
    points_cache_size = points_history.size();
}

BoxPointType LocalMap_Points;
bool Localmap_Initialized = false;

void lasermap_fov_segment() {
    cub_needrm.shrink_to_fit();

    V3D pos_LiD;
    if (use_imu_as_input) {
        pos_LiD = kf_input.x_.pos + kf_input.x_.rot.normalized() * Lidar_T_wrt_IMU;
    } else {
        pos_LiD = kf_output.x_.pos + kf_output.x_.rot.normalized() * Lidar_T_wrt_IMU;
    }
    if (!Localmap_Initialized) {
        for (int i = 0; i < 3; i++) {
            LocalMap_Points.vertex_min[i] = pos_LiD(i) - cube_len / 2.0;
            LocalMap_Points.vertex_max[i] = pos_LiD(i) + cube_len / 2.0;
        }
        Localmap_Initialized = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++) {
        dist_to_map_edge[i][0] = fabs(pos_LiD(i) - LocalMap_Points.vertex_min[i]);
        dist_to_map_edge[i][1] = fabs(pos_LiD(i) - LocalMap_Points.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE ||
            dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move) return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points;
    float mov_dist = max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9,
                         double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++) {
        tmp_boxpoints = LocalMap_Points;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points.vertex_max[i] - mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        } else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE) {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points.vertex_min[i] + mov_dist;
            cub_needrm.emplace_back(tmp_boxpoints);
        }
    }
    LocalMap_Points = New_LocalMap_Points;

    points_cache_collect();
    if (cub_needrm.size() > 0) int kdtree_delete_counter = ikdtree.Delete_Point_Boxes(cub_needrm);
}

void standard_pcl_cbk(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    mtx_buffer.lock();
    scan_count++;
    double preprocess_start_time = omp_get_wtime();
    if (get_time_sec(msg->header.stamp) < last_timestamp_lidar) {
        RCLCPP_ERROR(logger, "lidar loop back, clear buffer");
        // lidar_buffer.shrink_to_fit();

        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    last_timestamp_lidar = msg->header.stamp.sec;

    PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
    PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());
    double time_div = get_time_sec(msg->header.stamp);
    p_pre->process(msg, ptr);
    if (cut_frame) {
        sort(ptr->points.begin(), ptr->points.end(), time_list);

        for (int i = 0; i < ptr->size(); i++) {
            ptr_div->push_back(ptr->points[i]);
            // cout << "check time:" << ptr->points[i].curvature << endl;
            if (ptr->points[i].curvature / double(1000) + get_time_sec(msg->header.stamp) - time_div >
                cut_frame_time_interval) {
                if (ptr_div->size() < 1) continue;
                PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
                *ptr_div_i = *ptr_div;
                lidar_buffer.push_back(ptr_div_i);
                time_buffer.push_back(time_div);
                time_div += ptr->points[i].curvature / double(1000);
                ptr_div->clear();
            }
        }
        if (!ptr_div->empty()) {
            lidar_buffer.push_back(ptr_div);
            // ptr_div->clear();
            time_buffer.push_back(time_div);
        }
    } else if (con_frame) {
        if (frame_ct == 0) {
            time_con = last_timestamp_lidar; //get_time_sec(msg->header.stamp);
        }
        if (frame_ct < con_frame_num) {
            for (int i = 0; i < ptr->size(); i++) {
                ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
                ptr_con->push_back(ptr->points[i]);
            }
            frame_ct++;
        } else {
            PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
            *ptr_con_i = *ptr_con;
            lidar_buffer.push_back(ptr_con_i);
            double time_con_i = time_con;
            time_buffer.push_back(time_con_i);
            ptr_con->clear();
            frame_ct = 0;
        }
    } else {
        lidar_buffer.emplace_back(ptr);
        time_buffer.emplace_back(get_time_sec(msg->header.stamp));
    }
    s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

// void livox_pcl_cbk(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg) {
//     mtx_buffer.lock();
//     double preprocess_start_time = omp_get_wtime();
//     scan_count++;
//     if (get_time_sec(msg->header.stamp) < last_timestamp_lidar) {
//         RCLCPP_ERROR(logger, "lidar loop back, clear buffer");

//         mtx_buffer.unlock();
//         sig_buffer.notify_all();
//         return;
//     }

//     last_timestamp_lidar = get_time_sec(msg->header.stamp);

//     PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
//     PointCloudXYZI::Ptr ptr_div(new PointCloudXYZI());
//     p_pre->process(msg, ptr);
//     double time_div = get_time_sec(msg->header.stamp);
//     if (cut_frame) {
//         sort(ptr->points.begin(), ptr->points.end(), time_list);

//         for (int i = 0; i < ptr->size(); i++) {
//             ptr_div->push_back(ptr->points[i]);
//             if (ptr->points[i].curvature / double(1000) + get_time_sec(msg->header.stamp) - time_div >
//                 cut_frame_time_interval) {
//                 if (ptr_div->size() < 1) continue;
//                 PointCloudXYZI::Ptr ptr_div_i(new PointCloudXYZI());
//                 // cout << "ptr div num:" << ptr_div->size() << endl;
//                 *ptr_div_i = *ptr_div;
//                 // cout << "ptr div i num:" << ptr_div_i->size() << endl;
//                 lidar_buffer.push_back(ptr_div_i);
//                 time_buffer.push_back(time_div);
//                 time_div += ptr->points[i].curvature / double(1000);
//                 ptr_div->clear();
//             }
//         }
//         if (!ptr_div->empty()) {
//             lidar_buffer.push_back(ptr_div);
//             // ptr_div->clear();
//             time_buffer.push_back(time_div);
//         }
//     } else if (con_frame) {
//         if (frame_ct == 0) {
//             time_con = last_timestamp_lidar; //get_time_sec(msg->header.stamp);
//         }
//         if (frame_ct < con_frame_num) {
//             for (int i = 0; i < ptr->size(); i++) {
//                 ptr->points[i].curvature += (last_timestamp_lidar - time_con) * 1000;
//                 ptr_con->push_back(ptr->points[i]);
//             }
//             frame_ct++;
//         } else {
//             PointCloudXYZI::Ptr ptr_con_i(new PointCloudXYZI());
//             *ptr_con_i = *ptr_con;
//             double time_con_i = time_con;
//             lidar_buffer.push_back(ptr_con_i);
//             time_buffer.push_back(time_con_i);
//             ptr_con->clear();
//             frame_ct = 0;
//         }
//     } else {
//         lidar_buffer.emplace_back(ptr);
//         time_buffer.emplace_back(get_time_sec(msg->header.stamp));
//     }
//     s_plot11[scan_count] = omp_get_wtime() - preprocess_start_time;
//     mtx_buffer.unlock();
//     sig_buffer.notify_all();
// }

void imu_cbk(const sensor_msgs::msg::Imu::SharedPtr msg_in) {
    publish_count++;
    sensor_msgs::msg::Imu::SharedPtr msg(new sensor_msgs::msg::Imu(*msg_in));

    msg->header.stamp = get_ros_time(get_time_sec(msg_in->header.stamp) - time_lag_imu_to_lidar);
    double timestamp = get_time_sec(msg->header.stamp);

    mtx_buffer.lock();

    if (timestamp < last_timestamp_imu) {
        RCLCPP_ERROR(logger, "imu loop back, clear deque");
        // imu_deque.shrink_to_fit();
        mtx_buffer.unlock();
        sig_buffer.notify_all();
        return;
    }

    imu_deque.emplace_back(msg);
    last_timestamp_imu = timestamp;
    mtx_buffer.unlock();
    sig_buffer.notify_all();
}

bool sync_packages(MeasureGroup &meas) {
    if (!imu_en) {
        if (!lidar_buffer.empty()) {
            meas.lidar = lidar_buffer.front();
            meas.lidar_beg_time = time_buffer.front();
            time_buffer.pop_front();
            lidar_buffer.pop_front();
            if (meas.lidar->points.size() < 1) {
                cout << "lose lidar" << std::endl;
                return false;
            }
            double end_time = meas.lidar->points.back().curvature;
            for (auto pt: meas.lidar->points) {
                if (pt.curvature > end_time) {
                    end_time = pt.curvature;
                }
            }
            lidar_end_time = meas.lidar_beg_time + end_time / double(1000);
            meas.lidar_last_time = lidar_end_time;
            return true;
        }
        return false;
    }

    if (lidar_buffer.empty() || imu_deque.empty()) {
        return false;
    }

    /*** push a lidar scan ***/
    if (!lidar_pushed) {
        meas.lidar = lidar_buffer.front();
        if (meas.lidar->points.size() < 1) {
            cout << "lose lidar" << endl;
            lidar_buffer.pop_front();
            time_buffer.pop_front();
            return false;
        }
        meas.lidar_beg_time = time_buffer.front();
        double end_time = meas.lidar->points.back().curvature;
        for (auto pt: meas.lidar->points) {
            if (pt.curvature > end_time) {
                end_time = pt.curvature;
            }
        }
        lidar_end_time = meas.lidar_beg_time + end_time / double(1000);

        meas.lidar_last_time = lidar_end_time;
        lidar_pushed = true;
    }

    if (last_timestamp_imu < lidar_end_time) {
        return false;
    }
    /*** push imu data, and pop from imu buffer ***/
    if (p_imu->imu_need_init_) {
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        while ((!imu_deque.empty()) && (imu_time < lidar_end_time)) {
            imu_time = get_time_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    } else if (!init_map) {
        double imu_time = get_time_sec(imu_deque.front()->header.stamp);
        meas.imu.shrink_to_fit();
        meas.imu.emplace_back(imu_last_ptr);

        while ((!imu_deque.empty()) && (imu_time < lidar_end_time)) {
            imu_time = get_time_sec(imu_deque.front()->header.stamp);
            if (imu_time > lidar_end_time) break;
            meas.imu.emplace_back(imu_deque.front());
            imu_last = imu_next;
            imu_last_ptr = imu_deque.front();
            imu_next = *(imu_deque.front());
            imu_deque.pop_front();
        }
    }

    lidar_buffer.pop_front();
    time_buffer.pop_front();
    lidar_pushed = false;
    return true;
}

int process_increments = 0;

void map_incremental() {
    PointVector PointToAdd;
    PointVector PointNoNeedDownsample;
    PointToAdd.reserve(feats_down_size);
    PointNoNeedDownsample.reserve(feats_down_size);

    for (int i = 0; i < feats_down_size; i++) {
        if (!Nearest_Points[i].empty()) {
            const PointVector &points_near = Nearest_Points[i];
            bool need_add = true;
            PointType downsample_result, mid_point;
            mid_point.x = floor(feats_down_world->points[i].x / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.y = floor(feats_down_world->points[i].y / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            mid_point.z = floor(feats_down_world->points[i].z / filter_size_map_min) * filter_size_map_min +
                          0.5 * filter_size_map_min;
            /* If the nearest points is definitely outside the downsample box */
            if (fabs(points_near[0].x - mid_point.x) > 1.732 * filter_size_map_min ||
                fabs(points_near[0].y - mid_point.y) > 1.732 * filter_size_map_min ||
                fabs(points_near[0].z - mid_point.z) > 1.732 * filter_size_map_min) {
                PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
                continue;
            }
            /* Check if there is a point already in the downsample box */
            float dist = calc_dist<float>(feats_down_world->points[i], mid_point);
            for (int readd_i = 0; readd_i < points_near.size(); readd_i++) {
                /* Those points which are outside the downsample box should not be considered. */
                if (fabs(points_near[readd_i].x - mid_point.x) < 0.5 * filter_size_map_min &&
                    fabs(points_near[readd_i].y - mid_point.y) < 0.5 * filter_size_map_min &&
                    fabs(points_near[readd_i].z - mid_point.z) < 0.5 * filter_size_map_min) {
                    need_add = false;
                    break;
                }
            }
            if (need_add) PointToAdd.emplace_back(feats_down_world->points[i]);
        } else {
            // PointToAdd.emplace_back(feats_down_world->points[i]);
            PointNoNeedDownsample.emplace_back(feats_down_world->points[i]);
        }
    }
    int add_point_size = ikdtree.Add_Points(PointToAdd, true);
    ikdtree.Add_Points(PointNoNeedDownsample, false);
}

void publish_init_kdtree(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes) {
    
    if (odom_only) {return;}

    int size_init_ikdtree = ikdtree.size();
    PointCloudXYZI::Ptr laserCloudInit(new PointCloudXYZI(size_init_ikdtree, 1));

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    PointVector().swap(ikdtree.PCL_Storage);
    ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);

    laserCloudInit->points = ikdtree.PCL_Storage;
    pcl::toROSMsg(*laserCloudInit, laserCloudmsg);

    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = odom_header_frame_id;
    if (!odom_only) {
        pubLaserCloudFullRes->publish(laserCloudmsg);
    }
}

PointCloudXYZI::Ptr pcl_wait_pub(new PointCloudXYZI(500000, 1));
PointCloudXYZI::Ptr pcl_wait_save(new PointCloudXYZI());

void publish_frame_world(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFullRes) {

    if (odom_only) {return;}

    if (scan_pub_en) {
        PointCloudXYZI::Ptr laserCloudFullRes(feats_down_body);
        int size = laserCloudFullRes->points.size();

        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            // if (i % 3 == 0)
            // {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity; // feats_down_world->points[i].y; // 
            // }
        }
        sensor_msgs::msg::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorld, laserCloudmsg);

        laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
        laserCloudmsg.header.frame_id = odom_header_frame_id;
        pubLaserCloudFullRes->publish(laserCloudmsg);
        publish_count -= PUBFRAME_PERIOD;
    }

    /**************** save map ****************/
    /* 1. make sure you have enough memories
    /* 2. noted that pcd save will influence the real-time performences **/
    if (pcd_save_en) {
        int size = feats_down_world->points.size();
        PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

        for (int i = 0; i < size; i++) {
            laserCloudWorld->points[i].x = feats_down_world->points[i].x;
            laserCloudWorld->points[i].y = feats_down_world->points[i].y;
            laserCloudWorld->points[i].z = feats_down_world->points[i].z;
            laserCloudWorld->points[i].intensity = feats_down_world->points[i].intensity;
        }

        *pcl_wait_save += *laserCloudWorld;

        static int scan_wait_num = 0;
        scan_wait_num++;
        if (pcl_wait_save->size() > 0 && pcd_save_interval > 0 && scan_wait_num >= pcd_save_interval) {
            pcd_index++;
            string all_points_dir(string(string(ROOT_DIR) + "PCD/scans_") + to_string(pcd_index) + string(".pcd"));
            pcl::PCDWriter pcd_writer;
            cout << "current scan saved to /PCD/" << all_points_dir << endl;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
            pcl_wait_save->clear();
            scan_wait_num = 0;
        }
    }
}

void publish_frame_body(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr &pubLaserCloudFull_body) {

    if (odom_only) {return;}

    int size = feats_undistort->points.size();
    PointCloudXYZI::Ptr laserCloudIMUBody(new PointCloudXYZI(size, 1));

    for (int i = 0; i < size; i++) {
        pointBodyLidarToIMU(&feats_undistort->points[i], \
                            &laserCloudIMUBody->points[i]);
    }

    sensor_msgs::msg::PointCloud2 laserCloudmsg;
    pcl::toROSMsg(*laserCloudIMUBody, laserCloudmsg);
    laserCloudmsg.header.stamp = get_ros_time(lidar_end_time);
    laserCloudmsg.header.frame_id = body_frame;
    pubLaserCloudFull_body->publish(laserCloudmsg);
    publish_count -= PUBFRAME_PERIOD;
}

template<typename T>
void set_posestamp(T &out) {
    if (!use_imu_as_input) {
        out.position.x = kf_output.x_.pos(0);
        out.position.y = kf_output.x_.pos(1);
        out.position.z = kf_output.x_.pos(2);
        out.orientation.x = kf_output.x_.rot.coeffs()[0];
        out.orientation.y = kf_output.x_.rot.coeffs()[1];
        out.orientation.z = kf_output.x_.rot.coeffs()[2];
        out.orientation.w = kf_output.x_.rot.coeffs()[3];
    } else {
        out.position.x = kf_input.x_.pos(0);
        out.position.y = kf_input.x_.pos(1);
        out.position.z = kf_input.x_.pos(2);
        out.orientation.x = kf_input.x_.rot.coeffs()[0];
        out.orientation.y = kf_input.x_.rot.coeffs()[1];
        out.orientation.z = kf_input.x_.rot.coeffs()[2];
        out.orientation.w = kf_input.x_.rot.coeffs()[3];
    }
}

template<typename T>
void set_twist(T &out) {
    if (!use_imu_as_input) {
        out.linear.x = kf_output.x_.vel(0);
        out.linear.y = kf_output.x_.vel(1);
        out.linear.z = kf_output.x_.vel(2);
        out.angular.x = kf_output.x_.omg(0);
        out.angular.y = kf_output.x_.omg(1);
        out.angular.z = kf_output.x_.omg(2);
    } else {
        out.linear.x = kf_input.x_.vel(0);
        out.linear.y = kf_input.x_.vel(1);
        out.linear.z = kf_input.x_.vel(2);
        out.angular.x = imu_last.angular_velocity.x;
        out.angular.y = imu_last.angular_velocity.y;
        out.angular.z = imu_last.angular_velocity.z;
    }
}

void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped,
                      std::shared_ptr<tf2_ros::TransformBroadcaster> &tf_br) {

    odomAftMapped.header.frame_id = odom_header_frame_id;
    odomAftMapped.child_frame_id = odom_child_frame_id;

    if (publish_odometry_without_downsample) {
        odomAftMapped.header.stamp = get_ros_time(time_current);
    } else {
        odomAftMapped.header.stamp = get_ros_time(lidar_end_time);
    }
    set_posestamp(odomAftMapped.pose.pose);
    set_twist(odomAftMapped.twist.twist);

    if (odom_only){
        Matrix3d cov = kf_output.get_P().block<3, 3>(0, 0);

        // Get the position components (first 3x3)
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                odomAftMapped.pose.covariance[6 * i + j] = cov(i, j);
            }
        }

        odomAftMapped.pose.covariance[21] = 0.0;    // Covariance for roll
        odomAftMapped.pose.covariance[28] = 0.0;    // Covariance for pitch
        odomAftMapped.pose.covariance[35] = 0.05;   // Covariance for yaw

        odomAftMapped.twist.covariance[0] = 0.1;    // Covariance for linear velocity on x
        odomAftMapped.twist.covariance[7] = 0.1;    // Covariance for linear velocity on y
        odomAftMapped.twist.covariance[14] = 0.0;   // Covariance for linear velocity on z
        odomAftMapped.twist.covariance[21] = 0.0;  // Covariance for angular velocity (roll)
        odomAftMapped.twist.covariance[28] = 0.0;  // Covariance for angular velocity (pitch)
        odomAftMapped.twist.covariance[35] = 0.05;  // Covariance for angular velocity (yaw)
    }

    pubOdomAftMapped->publish(odomAftMapped);

    geometry_msgs::msg::TransformStamped transform;
    transform.header.frame_id = odom_header_frame_id;
    transform.header.stamp = odomAftMapped.header.stamp;

    // REP-105 wants map -> base_footprint, but the filter estimates map -> the
    // IMU on the mast, and the bag's /tf_static already owns
    // base_footprint -> that IMU frame. Broadcasting the IMU edge here too
    // would give it TWO parents and split the tree, so compose the static
    // body -> child extrinsic out and broadcast map -> child instead.
    //
    // Resolved ONCE and cached: it is static, and until the lookup succeeds
    // nothing is broadcast at all -- a map -> base edge computed from a missing
    // extrinsic would be silently wrong rather than absent.
    if (!tf_child_frame.empty() && tf_child_frame != odom_child_frame_id) {
        if (!tf_child_resolved) {
            if (!tf_buffer_g) return;
            try {
                auto tfs = tf_buffer_g->lookupTransform(
                    odom_child_frame_id, tf_child_frame, tf2::TimePointZero);
                const auto &q = tfs.transform.rotation;
                const auto &v = tfs.transform.translation;
                Eigen::Quaterniond eq(q.w, q.x, q.y, q.z);
                R_body_to_tfchild = eq.toRotationMatrix();
                t_body_to_tfchild = V3D(v.x, v.y, v.z);
                tf_child_resolved = true;
            } catch (const tf2::TransformException &ex) {
                // Throttled, not silent: the only visible effect of returning
                // quietly here is nav2 waiting forever for a map frame that is
                // never going to arrive.
                RCLCPP_WARN_THROTTLE(this_logger(), *node_g->get_clock(), 5000,
                    "cannot resolve %s -> %s yet (%s); NOT broadcasting %s -> %s "
                    "until it does",
                    odom_child_frame_id.c_str(), tf_child_frame.c_str(), ex.what(),
                    odom_header_frame_id.c_str(), tf_child_frame.c_str());
                return;
            }
        }
        const Eigen::Quaterniond q_mb(odomAftMapped.pose.pose.orientation.w,
                                      odomAftMapped.pose.pose.orientation.x,
                                      odomAftMapped.pose.pose.orientation.y,
                                      odomAftMapped.pose.pose.orientation.z);
        const M3D R_mb = q_mb.toRotationMatrix();
        const V3D p_mb(odomAftMapped.pose.pose.position.x,
                       odomAftMapped.pose.pose.position.y,
                       odomAftMapped.pose.pose.position.z);
        const M3D R_mc = R_mb * R_body_to_tfchild;
        const V3D p_mc = R_mb * t_body_to_tfchild + p_mb;
        const Eigen::Quaterniond q_mc(R_mc);
        transform.child_frame_id = tf_child_frame;
        transform.transform.translation.x = p_mc(0);
        transform.transform.translation.y = p_mc(1);
        transform.transform.translation.z = p_mc(2);
        transform.transform.rotation.w = q_mc.w();
        transform.transform.rotation.x = q_mc.x();
        transform.transform.rotation.y = q_mc.y();
        transform.transform.rotation.z = q_mc.z();
    } else {
        transform.child_frame_id = odom_child_frame_id;
        transform.transform.translation.x = odomAftMapped.pose.pose.position.x;
        transform.transform.translation.y = odomAftMapped.pose.pose.position.y;
        transform.transform.translation.z = odomAftMapped.pose.pose.position.z;
        transform.transform.rotation = odomAftMapped.pose.pose.orientation;
    }

    tf_br->sendTransform(transform);

    // /localization/pose -- the SAME pose, but genuinely in tf_child_frame
    // rather than the mast IMU, so a consumer that assumes child_frame_id is
    // the robot base reads the right thing. nav2_params_pointloc.yaml points
    // bt_navigator's odom_topic at this.
    if (pub_localization_g && tf_child_resolved) {
        nav_msgs::msg::Odometry loc;
        loc.header = odomAftMapped.header;
        loc.child_frame_id = tf_child_frame;
        loc.pose.pose.position.x = transform.transform.translation.x;
        loc.pose.pose.position.y = transform.transform.translation.y;
        loc.pose.pose.position.z = transform.transform.translation.z;
        loc.pose.pose.orientation = transform.transform.rotation;
        loc.pose.covariance = odomAftMapped.pose.covariance;

        // Twist is expressed in child_frame_id, so rotating alone is not
        // enough -- the base origin sits off the body origin, so it also picks
        // up the lever-arm term:
        //   w_base = R * w_body
        //   v_base = R * v_body + w_base x (R * t)
        const M3D R_bb = R_body_to_tfchild.transpose();
        const V3D r_b  = R_bb * t_body_to_tfchild;
        const auto &tw = odomAftMapped.twist.twist;
        const V3D v_b_in(tw.linear.x, tw.linear.y, tw.linear.z);
        const V3D w_b_in(tw.angular.x, tw.angular.y, tw.angular.z);
        const V3D w_o = R_bb * w_b_in;
        const V3D v_o = R_bb * v_b_in + w_o.cross(r_b);
        loc.twist.twist.linear.x = v_o(0);
        loc.twist.twist.linear.y = v_o(1);
        loc.twist.twist.linear.z = v_o(2);
        loc.twist.twist.angular.x = w_o(0);
        loc.twist.twist.angular.y = w_o(1);
        loc.twist.twist.angular.z = w_o(2);
        loc.twist.covariance = odomAftMapped.twist.covariance;
        pub_localization_g->publish(loc);
    }
}

void publish_path(const rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr &pubPath) {

    if (odom_only) {return;}

    set_posestamp(msg_body_pose.pose);
    // msg_body_pose.header.stamp = ros::Time::now();
    msg_body_pose.header.stamp = get_ros_time(lidar_end_time);
    msg_body_pose.header.frame_id = odom_header_frame_id;
    static int jjj = 0;
    jjj++;
    // if (jjj % 2 == 0) // if path is too large, the rvis will crash
    {
        path.poses.emplace_back(msg_body_pose);
        pubPath->publish(path);
    }
}

bool load_prior_map(const rclcpp::Logger &log)
{
    // Named per run, not a bare pose.json: this directory also holds
    // pepper_map_lc_poses.txt for the other stack, and a second map would drop
    // a second pose file beside it. An undated name makes those silently
    // interchangeable -- the same failure as a .pcd paired with the wrong poses.
    const std::string pose_path =
        map_pose_file_param.find('/') != std::string::npos
            ? map_pose_file_param
            : map_dir_param + "/" + map_pose_file_param;
    std::ifstream pose_file(pose_path);
    if (!pose_file.is_open()) {
        RCLCPP_ERROR(log, "cannot open %s", pose_path.c_str());
        return false;
    }
    double tx, ty, tz, w, x, y, z;
    int count = 0;
    while (pose_file >> tx >> ty >> tz >> w >> x >> y >> z)
    {
        Eigen::Quaterniond q(w, x, y, z);
        q.normalize();
        V3D pos(tx, ty, tz);
        const std::string pcd = map_scan_dir_param + "/" + std::to_string(count) + ".pcd";
        PointCloudXYZI::Ptr temp(new PointCloudXYZI());
        if (pcl::io::loadPCDFile(pcd, *temp) < 0) {
            RCLCPP_ERROR(log, "cannot read %s (pose.json has %d entries so far)",
                         pcd.c_str(), count);
            return false;
        }
        position_map.push_back(pos);
        pose_map.push_back(q);
        scManager.makeAndSaveScancontextAndKeys(*temp);   // in the keyframe's own frame
        PointCloudXYZI::Ptr in_map(new PointCloudXYZI());
        pcl::transformPointCloud(*temp, *in_map, pos, q);
        *global_map += *in_map;
        count++;
    }
    if (count == 0) { RCLCPP_ERROR(log, "%s is empty", pose_path.c_str()); return false; }
    RCLCPP_INFO(log, "Prior map: %d keyframes, %zu pts", count, global_map->size());
    return true;
}

/*** Odometry pose at an init-phase scan, as map-agnostic odom <- IMU.
 *** Read under init_feats_mutex: the main thread appends to these vectors while
 *** the search thread reads them, and /relocalize clears them outright. ***/
bool odom_at(int id, Eigen::Matrix4d &T)
{
    std::lock_guard<std::mutex> lk(init_feats_mutex);
    if (id < 0 || id >= (int)position_init.size() || id >= (int)pose_init.size())
        return false;
    T = Eigen::Matrix4d::Identity();
    T.block<3,3>(0,0) = pose_init[id].toRotationMatrix();
    T.block<3,1>(0,3) = position_init[id];
    return true;
}

/*** Fraction of a scan that lands on the prior map at a proposed pose.
 ***
 *** The agreement check alone is NOT sufficient. MEASURED starting mid-bag in
 *** the corridor: ScanContext matched keyframes 203 and 191, which are adjacent
 *** to each other near the origin, so the two estimates agreed to 1.86 m and
 *** passed a 2 m limit -- while the robot was 41 m away. Two wrong matches to
 *** the same wrong place agree with each other perfectly. Agreement measures
 *** self-consistency, not correctness.
 ***
 *** This asks the map instead: put the scan where the candidate says, and see
 *** how much of it lands on something. A pose 41 m out overlaps almost nothing,
 *** and no amount of internal consistency can fake that. ***/
double map_overlap(const PointCloudXYZI::Ptr &scan_body, const Eigen::Matrix4d &T_map_body)
{
    if (!global_map_kdtree || scan_body->empty()) return 0.0;
    PointCloudXYZI::Ptr in_map(new PointCloudXYZI());
    pcl::transformPointCloud(*scan_body, *in_map, T_map_body.cast<float>());
    const double r2 = init_overlap_dist * init_overlap_dist;
    std::vector<int> idx(1); std::vector<float> d2(1);
    size_t hit = 0;
    for (const auto &pt : in_map->points) {
        if (global_map_kdtree->nearestKSearch(pt, 1, idx, d2) > 0 && d2[0] <= r2) ++hit;
    }
    return double(hit) / double(in_map->size());
}

/*** Background thread: find where we are, using ScanContext + ICP.
 ***
 *** Runs only until a lock is accepted. It consumes the undistorted scans the
 *** main loop queues during the init phase, and requires init_agree_count
 *** independent locks agreeing within init_agree_dist -- one descriptor hit in a
 *** corridor is not evidence, several that agree are. ***/
void global_localization_thread(rclcpp::Logger log)
{
    rclcpp::Rate rate(20);
    while (rclcpp::ok())
    {
        bool already_locked, keep_going;
        {
            std::lock_guard<std::mutex> lk(init_state_mutex);
            already_locked = global_localization_finish;
            keep_going = keep_searching;
        }
        // on_deactivate() sets this false and joins us -- exit promptly rather
        // than idle-sleeping through a lifecycle transition that's waiting on us.
        if (!keep_going) return;
        // Idle, not finished: /relocalize clears this flag to re-arm the search,
        // so returning here would make relocalization impossible for the life
        // of the process.
        if (already_locked) { rate.sleep(); continue; }
        if (!map_loaded) { rate.sleep(); continue; }

        auto candidate_count = []() {
            std::lock_guard<std::mutex> lk(candidate_mutex);
            return (int)candidate_ids.size();
        };
        while (candidate_count() < init_agree_count && rclcpp::ok())
        {
            std::pair<int, PointCloudXYZI::Ptr> item;
            {
                std::lock_guard<std::mutex> lk(init_feats_mutex);
                if (init_feats_down_bodys.empty()) { item.second = nullptr; }
                else { item = init_feats_down_bodys.front(); init_feats_down_bodys.pop(); }
            }
            if (!item.second) { rate.sleep(); continue; }

            PointCloudXYZI::Ptr scan(new PointCloudXYZI());
            pcl::copyPointCloud(*item.second, *scan);

            scManager.makeAndSaveScancontextAndKeys(*scan);
            auto hit = scManager.detectLoopClosureID();
            const int   match_id = hit.first;
            const float yaw_init = hit.second;
            scManager.dropBackScancontextAndKeys();       // do not grow the DB with live scans
            if (match_id == -1) { continue; }

            // ScanContext resolves yaw only; undo it, then ICP for the rest.
            Eigen::Matrix4d T_sc = Eigen::Matrix4d::Identity();
            T_sc.block<3,3>(0,0) = Eigen::Matrix3d(
                Eigen::AngleAxisd(-yaw_init, V3D(0,0,1)));
            pcl::transformPointCloud(*scan, *scan, T_sc);

            PointCloudXYZI::Ptr kf_cloud(new PointCloudXYZI());
            const std::string kf_pcd = map_scan_dir_param + "/" + std::to_string(match_id) + ".pcd";
            if (pcl::io::loadPCDFile(kf_pcd, *kf_cloud) < 0) { continue; }

            // Coarse then fine: the coarse pass has to survive a ScanContext hit
            // that is the right PLACE but metres off; the fine pass is the answer.
            Eigen::Matrix4d T_corr = T_sc;
            pcl::PointCloud<PointType>::Ptr unused(new pcl::PointCloud<PointType>());
            for (double maxd : {init_icp_coarse, init_icp_fine}) {
                pcl::IterativeClosestPoint<PointType, PointType> icp;
                icp.setMaxCorrespondenceDistance(maxd);
                icp.setInputSource(scan);
                icp.setInputTarget(kf_cloud);
                icp.align(*unused);
                if (!icp.hasConverged()) { T_corr.setZero(); break; }
                Eigen::Matrix4d Ti = icp.getFinalTransformation().cast<double>();
                pcl::transformPointCloud(*scan, *scan, Ti);
                T_corr = (Ti * T_corr).eval();
            }
            if (T_corr.isZero()) { continue; }

            Eigen::Matrix4d T_kf = Eigen::Matrix4d::Identity();
            T_kf.block<3,3>(0,0) = pose_map[match_id].toRotationMatrix();
            T_kf.block<3,1>(0,3) = position_map[match_id];

            Eigen::Matrix4d T_i_l = Eigen::Matrix4d::Identity();
            T_i_l.block<3,3>(0,0) = Lidar_R_wrt_IMU;
            T_i_l.block<3,1>(0,3) = Lidar_T_wrt_IMU;

            // map <- IMU at the scan this estimate came from
            const Eigen::Matrix4d T_cand = T_kf * T_corr * T_i_l.inverse();

            // Does the scan actually fit the map there? Checked against the
            // ORIGINAL scan, not the ICP-transformed copy, and in the lidar
            // frame the pose describes.
            const Eigen::Matrix4d T_cand_lidar = T_cand * T_i_l;
            const double ov = map_overlap(item.second, T_cand_lidar);
            if (ov < init_min_overlap) {
                RCLCPP_WARN(log, "[init] discarded keyframe %d: only %.0f%% of the "
                                 "scan lands on the map there (need %.0f%%)",
                            match_id, 100.0 * ov, 100.0 * init_min_overlap);
                continue;
            }
            // Motion gate: the new candidate must come from a scan the robot
            // has actually travelled from, or it is not independent evidence.
            int oldest_id = -1;
            {
                std::lock_guard<std::mutex> lk(candidate_mutex);
                if (!candidate_ids.empty()) oldest_id = candidate_ids.front();
            }
            if (init_require_motion && oldest_id != -1) {
                Eigen::Matrix4d T0, Tn;
                if (!odom_at(oldest_id, T0) || !odom_at(item.first, Tn)) continue;
                const double moved =
                    (Tn.block<3,1>(0,3) - T0.block<3,1>(0,3)).norm();
                if (moved < init_motion_min) {
                    RCLCPP_INFO(log, "[init] holding: only %.2f m travelled since "
                                     "the oldest kept estimate (need %.2f) -- move the robot",
                                moved, init_motion_min);
                    continue;
                }
            }
            int kept_count;
            {
                std::lock_guard<std::mutex> lk(candidate_mutex);
                candidate_poses.push_back(T_cand);
                candidate_ids.push_back(item.first);
                kept_count = (int)candidate_ids.size();
            }
            RCLCPP_INFO(log, "[init] candidate %d/%d: matched map keyframe %d "
                             "(%.0f%% overlap)",
                        kept_count, init_agree_count, match_id, 100.0 * ov);
        }

        std::vector<int> ids;
        std::vector<Eigen::Matrix4d, Eigen::aligned_allocator<Eigen::Matrix4d>> poses;
        {
            std::lock_guard<std::mutex> lk(candidate_mutex);
            ids = candidate_ids;
            poses = candidate_poses;
        }
        // A concurrent /relocalize (or auto-relocalize) can have cleared the
        // window while the loop above was mid-flight; re-check rather than
        // trust the count that satisfied the while condition.
        if ((int)ids.size() < init_agree_count) continue;

        // Each pose is the map pose at ITS OWN scan, so comparing them raw
        // penalises a correct pair for the distance the robot covered between
        // them -- the old 2 m tolerance was quietly absorbing that. Transport
        // each estimate back to the first scan's instant using the odometry
        // between, and the comparison becomes a true consistency test:
        //     predicted_0 = pose_i * T_odom(id_i)^-1 * T_odom(id_0)
        double spread = 0.0;
        for (size_t i = 1; i < poses.size(); ++i) {
            Eigen::Matrix4d Pi = poses[i];
            if (init_require_motion) {
                Eigen::Matrix4d T0, Ti;
                if (odom_at(ids[0], T0) && odom_at(ids[i], Ti)) {
                    Pi = poses[i] * Ti.inverse() * T0;
                } else {
                    RCLCPP_WARN(log, "[init] odometry trail unavailable; comparing "
                                     "estimates uncompensated");
                }
            }
            spread = std::max(spread,
                (Pi.block<3,1>(0,3) - poses[0].block<3,1>(0,3)).norm());
        }

        if (spread < init_agree_dist) {
            init_result.first  = ids[0];
            init_result.second = poses[0];
            {
                std::lock_guard<std::mutex> lk(init_state_mutex);
                global_localization_finish = true;
            }
            {
                std::lock_guard<std::mutex> lk(init_feats_mutex);
                std::queue<std::pair<int, PointCloudXYZI::Ptr>> empty;
                std::swap(init_feats_down_bodys, empty);
            }
            {
                std::lock_guard<std::mutex> lk(candidate_mutex);
                candidate_ids.clear();
                candidate_poses.clear();
            }
            RCLCPP_INFO(log, "[init] LOCKED: %d estimates agree to %.2f m (limit %.2f)",
                        init_agree_count, spread, init_agree_dist);
            continue;   // idle until /relocalize re-arms us
        }
        // Drop only the OLDEST candidate rather than the whole window: a
        // persistently good match should not be discarded just because it was
        // paired with one stale/bad one. With init_agree_count == 2 this means
        // the next new candidate is compared against the one just kept, not a
        // fresh pair from zero -- so a real, repeatable place is found in one
        // extra scan instead of requiring two brand-new ones every time.
        RCLCPP_WARN(log, "[init] rejected: estimates disagree by %.2f m (limit %.2f) "
                         "-- ambiguous place, dropping the oldest and retrying",
                    spread, init_agree_dist);
        {
            std::lock_guard<std::mutex> lk(candidate_mutex);
            if (!candidate_ids.empty()) {
                candidate_ids.erase(candidate_ids.begin());
                candidate_poses.erase(candidate_poses.begin());
            }
        }
    }
}

/*** Shared by /relocalize and the auto-relocalize path: drop the odometry
 *** trail and the candidate window, and stand the current lock down so
 *** global_localization_thread starts over. ***/
void rearm_search()
{
    {
        std::lock_guard<std::mutex> lk(init_feats_mutex);
        std::queue<std::pair<int, PointCloudXYZI::Ptr>> empty;
        std::swap(init_feats_down_bodys, empty);
        position_init.clear();
        pose_init.clear();
        init_count = 0;
    }
    {
        std::lock_guard<std::mutex> lk(candidate_mutex);
        candidate_ids.clear();
        candidate_poses.clear();
    }
    {
        std::lock_guard<std::mutex> lk(init_state_mutex);
        global_localization_finish = false;
    }
    global_update = false;
}

void publish_diagnostic(uint8_t level, const std::string &message,
                        const std::string &overlap_value = "")
{
    if (!pub_diag_g) return;
    diagnostic_msgs::msg::DiagnosticStatus status;
    status.level = level;
    status.name = "point_lio_localization: pose lock";
    status.hardware_id = "point_lio_localization";
    status.message = message;
    if (!overlap_value.empty()) {
        diagnostic_msgs::msg::KeyValue kv;
        kv.key = "map_overlap";
        kv.value = overlap_value;
        status.values.push_back(kv);
    }
    diagnostic_msgs::msg::DiagnosticArray arr;
    arr.header.stamp = node_g->get_clock()->now();
    arr.status.push_back(status);
    pub_diag_g->publish(arr);
}


int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto nh = std::make_shared<rclcpp::Node>("point_lio_localization");
    readParameters(nh);
    node_g = nh.get();

    /*** LOCALIZATION parameters. Declared here rather than in parameters.cpp
     *** so the graft stays contained to this file. ***/
    nh->declare_parameter<std::string>("localization.map_dir", "");
    nh->declare_parameter<std::string>("localization.map_scan_dir", "");
    nh->declare_parameter<std::string>("localization.map_pose_file", "pose.json");
    nh->declare_parameter<int>("localization.init_agree_count", 2);
    nh->declare_parameter<double>("localization.init_agree_dist", 2.0);
    nh->declare_parameter<double>("localization.init_icp_coarse", 5.0);
    nh->declare_parameter<double>("localization.init_icp_fine", 1.0);
    nh->declare_parameter<double>("localization.sc_lidar_height", 0.5);
    nh->declare_parameter<double>("localization.sc_max_radius", 10.0);
    nh->declare_parameter<double>("localization.sc_dist_thres", 0.15);
    nh->declare_parameter<int>("localization.sc_num_ring", 12);
    nh->declare_parameter<int>("localization.sc_num_sector", 40);
    nh->declare_parameter<bool>("localization.init_require_motion", false);
    nh->declare_parameter<double>("localization.init_motion_min", 0.50);
    nh->declare_parameter<double>("localization.init_min_overlap", 0.70);
    nh->declare_parameter<double>("localization.init_overlap_dist", 0.20);
    nh->declare_parameter<double>("localization.prior_map_view_leaf", 0.20);
    nh->declare_parameter<double>("localization.health_min_overlap", 0.45);
    nh->declare_parameter<double>("localization.health_bad_duration", 5.0);
    nh->declare_parameter<double>("localization.health_check_period", 1.0);
    nh->declare_parameter<bool>("localization.auto_relocalize", true);
    nh->declare_parameter<std::string>("publish.tf_child_frame", "base_footprint");
    nh->get_parameter("localization.map_dir", map_dir_param);
    nh->get_parameter("localization.map_scan_dir", map_scan_dir_param);
    nh->get_parameter("localization.map_pose_file", map_pose_file_param);
    nh->get_parameter("localization.init_agree_count", init_agree_count);
    nh->get_parameter("localization.init_agree_dist", init_agree_dist);
    nh->get_parameter("localization.init_icp_coarse", init_icp_coarse);
    nh->get_parameter("localization.init_icp_fine", init_icp_fine);
    nh->get_parameter("localization.sc_lidar_height", sc_lidar_height);
    nh->get_parameter("localization.sc_max_radius", sc_max_radius);
    nh->get_parameter("localization.sc_dist_thres", sc_dist_thres);
    nh->get_parameter("localization.sc_num_ring", sc_num_ring);
    nh->get_parameter("localization.sc_num_sector", sc_num_sector);
    nh->get_parameter("localization.init_require_motion", init_require_motion);
    nh->get_parameter("localization.init_motion_min", init_motion_min);
    nh->get_parameter("localization.init_min_overlap", init_min_overlap);
    nh->get_parameter("localization.init_overlap_dist", init_overlap_dist);
    nh->get_parameter("localization.prior_map_view_leaf", prior_map_view_leaf);
    nh->get_parameter("localization.health_min_overlap", health_min_overlap);
    nh->get_parameter("localization.health_bad_duration", health_bad_duration);
    nh->get_parameter("localization.health_check_period", health_check_period);
    nh->get_parameter("localization.auto_relocalize", auto_relocalize);
    nh->get_parameter("publish.tf_child_frame", tf_child_frame);
    if (map_pose_file_param.empty()) map_pose_file_param = "pose.json";
    if (map_scan_dir_param.empty()) map_scan_dir_param = map_dir_param + "/pcd";
    // Frames. After the handover the filter state IS the prior map's pose, so
    // publishing into Point-LIO's mapping default ('camera_init', its own start
    // frame) would name it wrongly -- the same misnomer FRAMES.md warns about.
    // Override rather than silently mislabel, so `ros2 run` behaves too.
    if (odom_header_frame_id == "camera_init") {
        odom_header_frame_id = "map";
        RCLCPP_INFO(nh->get_logger(),
            "odom_header_frame_id was the mapping default 'camera_init'; using "
            "'map' -- after the lock this estimate IS the prior map's pose.");
    }
    if (odom_child_frame_id == "aft_mapped") {
        RCLCPP_WARN(nh->get_logger(),
            "odom_child_frame_id is still 'aft_mapped'. It must name the frame "
            "the filter actually estimates (publish.body_frame of the LIO "
            "config, e.g. camera_imu_optical_frame) or the %s -> %s extrinsic "
            "cannot be resolved and NO map TF will be broadcast.",
            odom_child_frame_id.c_str(), tf_child_frame.c_str());
    }

    if (map_dir_param.empty()) {
        RCLCPP_FATAL(nh->get_logger(),
            "localization.map_dir is required -- point it at a directory "
            "holding the pose file and per-keyframe .pcd clouds.");
        return 1;
    }
    // spin_thread=true: the listener needs its OWN thread. The main loop below
    // blocks on the filter update, so a shared executor starves /tf_static
    // callbacks and the buffer never fills -- a lookup that silently never
    // resolves while the frames are being published the whole time.
    tf_buffer_g = std::make_shared<tf2_ros::Buffer>(nh->get_clock());
    tf_listener_g = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_g, nh, true);
    cout << "lidar_type: " << lidar_type << endl;

    path.header.stamp = get_ros_time(lidar_end_time);
    path.header.frame_id = odom_header_frame_id;

    /*** variables definition for counting ***/
    int frame_num = 0;
    double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_incre = 0, aver_time_solve = 0, aver_time_propag = 0;
    std::time_t startTime, endTime;

    /*** initialize variables ***/
    double FOV_DEG = (fov_deg + 10.0) > 179.9 ? 179.9 : (fov_deg + 10.0);
    double HALF_FOV_COS = cos((FOV_DEG) * 0.5 * PI_M / 180.0);

    memset(point_selected_surf, true, sizeof(point_selected_surf));
    downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
    downSizeFilterMap.setLeafSize(filter_size_map_min, filter_size_map_min, filter_size_map_min);
    Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
    Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
    if (extrinsic_est_en) {
        if (!use_imu_as_input) {
            kf_output.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_output.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        } else {
            kf_input.x_.offset_R_L_I = Lidar_R_wrt_IMU;
            kf_input.x_.offset_T_L_I = Lidar_T_wrt_IMU;
        }
    }
    p_imu->lidar_type = p_pre->lidar_type = lidar_type;
    p_imu->imu_en = imu_en;

    kf_input.init_dyn_share_modified(get_f_input, df_dx_input, h_model_input);
    kf_output.init_dyn_share_modified_2h(get_f_output, df_dx_output, h_model_output, h_model_IMU_output);
    Eigen::Matrix<double, 24, 24> P_init = MD(24, 24)::Identity() * 0.01;
    P_init.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
    P_init.block<6, 6>(15, 15) = MD(6, 6)::Identity() * 0.001;
    P_init.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
    kf_input.change_P(P_init);
    Eigen::Matrix<double, 30, 30> P_init_output = MD(30, 30)::Identity() * 0.01;
    P_init_output.block<3, 3>(21, 21) = MD(3, 3)::Identity() * 0.0001;
    P_init_output.block<6, 6>(6, 6) = MD(6, 6)::Identity() * 0.0001;
    P_init_output.block<6, 6>(24, 24) = MD(6, 6)::Identity() * 0.001;
    kf_input.change_P(P_init);
    kf_output.change_P(P_init_output);
    Eigen::Matrix<double, 24, 24> Q_input = process_noise_cov_input();
    Eigen::Matrix<double, 30, 30> Q_output = process_noise_cov_output();
    /*** debug record ***/
    FILE *fp;
    string pos_log_dir = root_dir + "/Log/pos_log.txt";
    fp = fopen(pos_log_dir.c_str(), "w");

    ofstream fout_out, fout_imu_pbp;
    fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), ios::out);
    fout_imu_pbp.open(DEBUG_FILE_DIR("imu_pbp.txt"), ios::out);
    if (fout_out && fout_imu_pbp)
        cout << "~~~~" << ROOT_DIR << " file opened" << endl;
    else
        cout << "~~~~" << ROOT_DIR << " doesn't exist" << endl;

    /*** ROS subscribe initialization ***/
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_pcl;
    // rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_pcl_livox_;
    // if (p_pre->lidar_type == AVIA) {
    //     sub_pcl_livox_ = nh->create_subscription<livox_ros_driver2::msg::CustomMsg>(lid_topic, 20, livox_pcl_cbk);
    // } else {
    sub_pcl = nh->create_subscription<sensor_msgs::msg::PointCloud2>(lid_topic, rclcpp::SensorDataQoS(), standard_pcl_cbk);
    // }
    // SensorDataQoS (BEST_EFFORT), not a plain depth: a plain depth is
    // RELIABLE, which matches NOTHING against the BEST_EFFORT publisher every
    // real IMU driver offers, so rmw silently delivers no IMU and Point-LIO
    // never initialises. Same bug fixed in FAST_LIO; see the long note there.
    auto sub_imu = nh->create_subscription<sensor_msgs::msg::Imu>(
        imu_topic, rclcpp::SensorDataQoS().keep_last(200000), imu_cbk);

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudFullRes_body;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudEffect;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubLaserCloudMap;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr pubPath;

    if (!odom_only){
        pubLaserCloudFullRes = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/cloud_registered", 100000);
        pubLaserCloudFullRes_body = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/cloud_registered_body", 100000);
        pubLaserCloudEffect = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/cloud_effected", 100000);
        pubLaserCloudMap = nh->create_publisher<sensor_msgs::msg::PointCloud2>
                ("/Laser_map", 100000);
        pubPath = nh->create_publisher<nav_msgs::msg::Path>
                ("/path", 100000);
    }

    // Choose topic name depending on odom_only value
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pubOdomAftMapped;
    if (odom_only){
        pubOdomAftMapped = nh->create_publisher<nav_msgs::msg::Odometry>
                ("/odom_corrected", 100000);
    } else {
        pubOdomAftMapped = nh->create_publisher<nav_msgs::msg::Odometry>
                ("/aft_mapped_to_init", 100000);
    }

    //auto plane_pub = nh->create_publisher<visualization_msgs::msg::Marker>
    //        ("/planner_normal", 1000);
    auto tf_broadcaster = std::make_shared<tf2_ros::TransformBroadcaster>(nh);

    /*** LOCALIZATION bring-up. The prior map is loaded and the search armed
     *** BEFORE the scan loop starts, so no scan is processed against an empty
     *** map. A missing map is fatal: this node localizes, it has nothing to do
     *** without one. ***/
    pub_localization_g = nh->create_publisher<nav_msgs::msg::Odometry>("/localization/pose", 10);
    pub_overlap_g = nh->create_publisher<std_msgs::msg::Float32>("/localization/overlap", 10);
    pub_diag_g = nh->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);

    // The map DB and the live scans must be described with identical geometry
    // or the descriptors are not comparable at all.
    scManager.set_geometry(sc_lidar_height, sc_max_radius,
                           sc_num_ring, sc_num_sector, sc_dist_thres);
    RCLCPP_INFO(nh->get_logger(), "Map: poses %s/%s  scans %s/",
                map_dir_param.c_str(), map_pose_file_param.c_str(),
                map_scan_dir_param.c_str());
    if (!load_prior_map(nh->get_logger())) {
        RCLCPP_FATAL(nh->get_logger(), "failed to load prior map from %s",
                     map_dir_param.c_str());
        return 1;
    }
    global_map_kdtree.reset(new pcl::KdTreeFLANN<PointType>());
    global_map_kdtree->setInputCloud(global_map);
    ikdtree_global.set_downsample_param(filter_size_map_min);
    ikdtree_global.Build(global_map->points);
    map_loaded = true;
    RCLCPP_INFO(nh->get_logger(),
        "Prior map ready (%zu pts). Searching for initial pose: ScanContext + "
        "ICP, %d estimates must agree within %.2f m.",
        global_map->size(), init_agree_count, init_agree_dist);

    // Latched (transient local) so RViz shows it on connect rather than only
    // if it happens to be listening at startup. Downsampled for display only.
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap;
    {
        rclcpp::QoS qos(1);
        qos.transient_local().reliable();
        pubPriorMap = nh->create_publisher<sensor_msgs::msg::PointCloud2>("/prior_map", qos);
        PointCloudXYZI::Ptr shown(new PointCloudXYZI());
        pcl::VoxelGrid<PointType> vg;
        vg.setLeafSize(prior_map_view_leaf, prior_map_view_leaf, prior_map_view_leaf);
        vg.setInputCloud(global_map);
        vg.filter(*shown);
        sensor_msgs::msg::PointCloud2 msg;
        pcl::toROSMsg(*shown, msg);
        msg.header.frame_id = "map";
        msg.header.stamp = nh->get_clock()->now();
        pubPriorMap->publish(msg);
    }

    // /relocalize -- "I do not trust where I think I am". The prior map STAYS
    // in the ikd-Tree while searching: the search thread does not use it, and
    // rebuilding a multi-million-point tree on a service call would be absurd.
    // While lost the filter simply finds no correspondences and coasts, which
    // is what being lost IS -- and the next lock is applied relative, so
    // however wrong the coasted pose is, it cancels.
    auto srv_relocalize = nh->create_service<std_srvs::srv::Trigger>(
        "/relocalize",
        [](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
           std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
            if (!map_loaded) {
                res->success = false;
                res->message = "prior map not loaded";
                return;
            }
            rearm_search();
            RCLCPP_WARN(this_logger(),
                "/relocalize: searching again. The pose is NOT trustworthy "
                "until the next 'Localized' line.");
            res->success = true;
            res->message = "Global search re-armed; watch the log for [init].";
        });

    // /initialpose -- a seed, as an ALTERNATIVE to the ScanContext search
    // rather than a replacement: the search keeps running, whichever locks
    // first wins. RViz publishes map <- base_footprint while the filter state
    // is map <- body, so the static body->base extrinsic is composed out.
    auto sub_initialpose = nh->create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
        "/initialpose", 1,
        [](const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg) {
            if (!map_loaded || !tf_child_resolved) {
                RCLCPP_WARN(this_logger(),
                    "/initialpose ignored: map or body->%s extrinsic not ready yet",
                    tf_child_frame.c_str());
                return;
            }
            const auto &q = msg->pose.pose.orientation;
            const auto &t = msg->pose.pose.position;
            Eigen::Quaterniond eq(q.w, q.x, q.y, q.z);
            eq.normalize();
            Eigen::Matrix4d T_map_base = Eigen::Matrix4d::Identity();
            T_map_base.block<3,3>(0,0) = eq.toRotationMatrix();
            T_map_base.block<3,1>(0,3) = V3D(t.x, t.y, t.z);
            Eigen::Matrix4d T_body_base = Eigen::Matrix4d::Identity();
            T_body_base.block<3,3>(0,0) = R_body_to_tfchild;
            T_body_base.block<3,1>(0,3) = t_body_to_tfchild;
            {
                std::lock_guard<std::mutex> lk(seed_mutex);
                pending_seed = T_map_base * T_body_base.inverse();
            }
            has_pending_seed = true;
            {
                std::lock_guard<std::mutex> lk(init_state_mutex);
                global_localization_finish = true;   // stand the search down
            }
            RCLCPP_INFO(this_logger(), "/initialpose accepted: seeding at x=%.2f y=%.2f",
                        t.x, t.y);
        });

    std::thread init_thread(global_localization_thread, nh->get_logger());
    rclcpp::Time last_health_check = nh->get_clock()->now();

//------------------------------------------------------------------------------------------------------
    signal(SIGINT, SigHandle);
    rclcpp::Rate rate(5000);
    while (rclcpp::ok()) {
        if (flg_exit) break;
        //ros::spinOnce();
        rclcpp::executors::SingleThreadedExecutor executor;
        executor.add_node(nh);
        executor.spin_some(); // 处理当前可用的回调

        if (sync_packages(Measures)) {
            if (flg_first_scan) {
                first_lidar_time = Measures.lidar_beg_time;
                flg_first_scan = false;
                cout << "first lidar time" << first_lidar_time << endl;
            }

            if (flg_reset) {
                RCLCPP_WARN(logger, "reset when rosbag play back");
                p_imu->Reset();
                flg_reset = false;
                continue;
            }
            double t0, t1, t2, t3, t4, t5, match_start, solve_start;
            match_time = 0;
            solve_time = 0;
            propag_time = 0;
            update_time = 0;
            t0 = omp_get_wtime();

            p_imu->Process(Measures, feats_undistort);

            if (feats_undistort->empty() || feats_undistort == nullptr) {
                continue;
            }
            if (imu_en) {
                if (!p_imu->gravity_align_) {
                    while (Measures.lidar_beg_time > get_time_sec(imu_next.header.stamp)) {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        // imu_deque.pop();
                    }
                    if (non_station_start) {
                        state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc << VEC_FROM_ARRAY(gravity_init);
                        state_out.acc *= -1;
                    } else {
                        state_in.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                        state_out.gravity = -1 * p_imu->mean_acc * G_m_s2 / acc_norm;
                        state_out.acc = p_imu->mean_acc * G_m_s2 / acc_norm;
                    }
                    if (gravity_align) {
                        Eigen::Matrix3d rot_init;
                        p_imu->gravity_ << VEC_FROM_ARRAY(gravity);
                        p_imu->Set_init(state_in.gravity, rot_init);
                        state_in.gravity = state_out.gravity = p_imu->gravity_;
                        state_in.rot = state_out.rot = rot_init;
                        state_in.rot.normalize();
                        state_out.rot.normalize();
                        state_out.acc = -rot_init.transpose() * state_out.gravity;
                    }
                    kf_input.change_x(state_in);
                    kf_output.change_x(state_out);
                }
            } else {
                if (!p_imu->gravity_align_) {
                    state_in.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.gravity << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc << VEC_FROM_ARRAY(gravity_init);
                    state_out.acc *= -1;
                }
            }
            /*** THE HANDOVER. A lock has been found but not yet applied: move
             *** the filter state into map coordinates and give it the prior map
             *** to register against. From here the estimate IS the map pose --
             *** there is no map -> odom correction, which is the whole point.
             ***
             *** The lock names the scan it was computed from and odometry has
             *** run on since, so carry it forward:
             ***   T_map_now = T_map_at_lock * T_odom_at_lock^-1 * T_odom_now
             *** A seed takes precedence and is applied directly -- no carry
             *** forward -- so it is correct whether or not a lock exists.
             ***
             *** This is a change of WORLD FRAME, not a pose edit, so every
             *** world-frame quantity rotates with it. In Point-LIO those are
             *** pos, rot, vel and gravity; omg and acc are BODY frame (see the
             *** file header) and carry over untouched, as bg/ba do. Leaving
             *** gravity behind would leave it pointing sideways in the new
             *** frame and the filter diverges within a few scans. ***/
            {
                Eigen::Matrix4d T_map_now = Eigen::Matrix4d::Identity();
                bool teleport = false;

                if (has_pending_seed.exchange(false)) {
                    std::lock_guard<std::mutex> lk(seed_mutex);
                    T_map_now = pending_seed;
                    teleport = true;
                } else {
                    std::unique_lock<std::mutex> lk(init_state_mutex);
                    const bool locked = global_localization_finish;
                    lk.unlock();
                    if (locked && !global_update) {
                        const int id = init_result.first;
                        Eigen::Matrix4d T_odom_lock = Eigen::Matrix4d::Identity();
                        T_odom_lock.block<3,3>(0,0) = pose_init[id].toRotationMatrix();
                        T_odom_lock.block<3,1>(0,3) = position_init[id];
                        Eigen::Matrix4d T_odom_now = Eigen::Matrix4d::Identity();
                        if (use_imu_as_input) {
                            T_odom_now.block<3,3>(0,0) = kf_input.x_.rot.normalized().toRotationMatrix();
                            T_odom_now.block<3,1>(0,3) = kf_input.x_.pos;
                        } else {
                            T_odom_now.block<3,3>(0,0) = kf_output.x_.rot.normalized().toRotationMatrix();
                            T_odom_now.block<3,1>(0,3) = kf_output.x_.pos;
                        }
                        T_map_now = init_result.second * T_odom_lock.inverse() * T_odom_now;
                        teleport = true;
                    }
                }

                if (teleport) {
                    const M3D R_new = T_map_now.block<3,3>(0,0);
                    if (use_imu_as_input) {
                        state_input gs = kf_input.x_;
                        const M3D R_delta = R_new * gs.rot.normalized().toRotationMatrix().transpose();
                        gs.pos = T_map_now.block<3,1>(0,3);
                        gs.rot = R_new;
                        gs.vel = R_delta * gs.vel;
                        gs.gravity = R_delta * gs.gravity;
                        kf_input.change_x(gs);
                        state_in = kf_input.x_;
                    } else {
                        state_output gs = kf_output.x_;
                        const M3D R_delta = R_new * gs.rot.normalized().toRotationMatrix().transpose();
                        gs.pos = T_map_now.block<3,1>(0,3);
                        gs.rot = R_new;
                        gs.vel = R_delta * gs.vel;
                        gs.gravity = R_delta * gs.gravity;
                        // omg/acc are body frame -- untouched on purpose.
                        kf_output.change_x(gs);
                        state_out = kf_output.x_;
                    }
                    // Hand the filter the prior map, ONCE: ikdtree_global is
                    // moved-from afterwards, and on a /relocalize the tree
                    // already holds it, so a second swap installs an empty one.
                    if (!map_swapped) {
                        ikdtree = std::move(ikdtree_global);
                        map_swapped = true;
                        init_map = true;      // stop Point-LIO building its own
                    }
                    global_update = true;
                    RCLCPP_INFO(nh->get_logger(),
                        "Localized: filter is now in the map frame at "
                        "x=%.2f y=%.2f z=%.2f; prior map is read-only from here.",
                        T_map_now(0,3), T_map_now(1,3), T_map_now(2,3));
                }
            }

            /*** Segment the map in lidar FOV ***/
            lasermap_fov_segment();
            /*** downsample the feature points in a scan ***/
            t1 = omp_get_wtime();
            if (space_down_sample) {
                downSizeFilterSurf.setInputCloud(feats_undistort);
                downSizeFilterSurf.filter(*feats_down_body);
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            } else {
                feats_down_body = Measures.lidar;
                sort(feats_down_body->points.begin(), feats_down_body->points.end(), time_list);
            }
            time_seq = time_compressing<int>(feats_down_body);
            feats_down_size = feats_down_body->points.size();

            /*** initialize the map kdtree ***/
            if (!init_map) {
                if (ikdtree.Root_Node == nullptr) //
                    // if(feats_down_size > 5)
                {
                    ikdtree.set_downsample_param(filter_size_map_min);
                }

                feats_down_world->resize(feats_down_size);
                for (int i = 0; i < feats_down_size; i++) {
                    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
                }
                for (size_t i = 0; i < feats_down_world->size(); i++) {
                    init_feats_world->points.emplace_back(feats_down_world->points[i]);
                }
                if (init_feats_world->size() < init_map_size) continue;
                ikdtree.Build(init_feats_world->points);
                init_map = true;
                publish_init_kdtree(pubLaserCloudMap); //(pubLaserCloudFullRes);
                continue;
            }
            /*** ICP and Kalman filter update ***/
            normvec->resize(feats_down_size);
            feats_down_world->resize(feats_down_size);

            Nearest_Points.resize(feats_down_size);

            t2 = omp_get_wtime();

            /*** iterated state estimation ***/
            crossmat_list.reserve(feats_down_size);
            pbody_list.reserve(feats_down_size);
            // pbody_ext_list.reserve(feats_down_size);

            for (size_t i = 0; i < feats_down_body->size(); i++) {
                V3D point_this(feats_down_body->points[i].x,
                               feats_down_body->points[i].y,
                               feats_down_body->points[i].z);
                pbody_list[i] = point_this;
                if (extrinsic_est_en) {
                    if (!use_imu_as_input) {
                        point_this = kf_output.x_.offset_R_L_I.normalized() * point_this + kf_output.x_.offset_T_L_I;
                    } else {
                        point_this = kf_input.x_.offset_R_L_I.normalized() * point_this + kf_input.x_.offset_T_L_I;
                    }
                } else {
                    point_this = Lidar_R_wrt_IMU * point_this + Lidar_T_wrt_IMU;
                }
                M3D point_crossmat;
                point_crossmat << SKEW_SYM_MATRX(point_this);
                crossmat_list[i] = point_crossmat;
            }

            if (!use_imu_as_input) {
                bool imu_upda_cov = false;
                effct_feat_num = 0;
                /**** point by point update ****/

                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++) {
                    PointType &point_body = feats_down_body->points[idx + time_seq[k]];

                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;

                    if (is_first_frame) {
                        if (imu_en) {
                            while (time_current > get_time_sec(imu_next.header.stamp)) {
                                imu_last = imu_next;
                                imu_next = *(imu_deque.front());
                                imu_deque.pop_front();
                                // imu_deque.pop();
                            }

                            angvel_avr
                                    << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                            acc_avr
                                    << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;
                        }
                        is_first_frame = false;
                        imu_upda_cov = true;
                        time_update_last = time_current;
                        time_predict_last_const = time_current;
                    }
                    if (imu_en) {
                        bool imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                        while (imu_comes) {
                            imu_upda_cov = true;
                            angvel_avr
                                    << imu_next.angular_velocity.x, imu_next.angular_velocity.y, imu_next.angular_velocity.z;
                            acc_avr
                                    << imu_next.linear_acceleration.x, imu_next.linear_acceleration.y, imu_next.linear_acceleration.z;

                            /*** covariance update ***/
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                            double dt = get_time_sec(imu_last.header.stamp) - time_predict_last_const;
                            kf_output.predict(dt, Q_output, input_in, true, false);
                            time_predict_last_const = get_time_sec(imu_last.header.stamp); // big problem
                            imu_comes = time_current > get_time_sec(imu_next.header.stamp);
                            // if (!imu_comes)
                            {
                                double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;

                                if (dt_cov > 0.0) {
                                    time_update_last = get_time_sec(imu_last.header.stamp);
                                    double propag_imu_start = omp_get_wtime();

                                    kf_output.predict(dt_cov, Q_output, input_in, false, true);

                                    propag_time += omp_get_wtime() - propag_imu_start;
                                    double solve_imu_start = omp_get_wtime();
                                    kf_output.update_iterated_dyn_share_IMU();
                                    solve_time += omp_get_wtime() - solve_imu_start;
                                }
                            }
                        }
                    }

                    double dt = time_current - time_predict_last_const;
                    double propag_state_start = omp_get_wtime();
                    if (!prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_output.predict(dt, Q_output, input_in, true, false);
                    propag_time += omp_get_wtime() - propag_state_start;
                    time_predict_last_const = time_current;
                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

                    double t_update_start = omp_get_wtime();

                    if (feats_down_size < 1) {
                        RCLCPP_WARN(logger, "No point, skip this scan!\n");
                        idx += time_seq[k];
                        continue;
                    }
                    if (!kf_output.update_iterated_dyn_share_modified()) {
                        idx = idx + time_seq[k];
                        continue;
                    }

                    if (prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (!imu_en && (dt_cov >= imu_time_inte)) // (point_cov_not_prop && imu_prop_cov)
                        {
                            double propag_cov_start = omp_get_wtime();
                            kf_output.predict(dt_cov, Q_output, input_in, false, true);
                            imu_upda_cov = false;
                            time_update_last = time_current;
                            propag_time += omp_get_wtime() - propag_cov_start;
                        }
                    }

                    solve_start = omp_get_wtime();

                    if (publish_odometry_without_downsample) {
                        /******* Publish odometry *******/

                        publish_odometry(pubOdomAftMapped, tf_broadcaster);
                        if (runtime_pos_log) {
                            state_out = kf_output.x_;
                            euler_cur = SO3ToEuler(state_out.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                     << euler_cur.transpose() << " " << state_out.pos.transpose() << " "
                                     << state_out.vel.transpose() << " " << state_out.omg.transpose() << " "
                                     << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " "
                                     << state_out.bg.transpose() << " " << state_out.ba.transpose() << " "
                                     << feats_undistort->points.size() << endl;
                        }
                    }

                    for (int j = 0; j < time_seq[k]; j++) {
                        PointType &point_body_j = feats_down_body->points[idx + j + 1];
                        PointType &point_world_j = feats_down_world->points[idx + j + 1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }

                    solve_time += omp_get_wtime() - solve_start;

                    update_time += omp_get_wtime() - t_update_start;
                    idx += time_seq[k];
                    // cout << "pbp output effect feat num:" << effct_feat_num << endl;
                }
            } else {
                bool imu_prop_cov = false;
                effct_feat_num = 0;

                double pcl_beg_time = Measures.lidar_beg_time;
                idx = -1;
                for (k = 0; k < time_seq.size(); k++) {
                    PointType &point_body = feats_down_body->points[idx + time_seq[k]];
                    time_current = point_body.curvature / 1000.0 + pcl_beg_time;
                    if (is_first_frame) {
                        while (time_current > get_time_sec(imu_next.header.stamp)) {
                            imu_last = imu_next;
                            imu_next = *(imu_deque.front());
                            imu_deque.pop_front();
                            // imu_deque.pop();
                        }
                        imu_prop_cov = true;
                        // imu_upda_cov = true;

                        is_first_frame = false;
                        t_last = time_current;
                        time_update_last = time_current;
                        // if(prop_at_freq_of_imu)
                        {
                            input_in.gyro << imu_last.angular_velocity.x,
                                    imu_last.angular_velocity.y,
                                    imu_last.angular_velocity.z;

                            input_in.acc << imu_last.linear_acceleration.x,
                                    imu_last.linear_acceleration.y,
                                    imu_last.linear_acceleration.z;
                            // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                            //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                            //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);

                            // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                            //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                            // 0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);

                            // angvel_avr -= state.bias_g;
                            input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        }
                    }

                    while (time_current > get_time_sec(imu_next.header.stamp)) // && !imu_deque.empty())
                    {
                        imu_last = imu_next;
                        imu_next = *(imu_deque.front());
                        imu_deque.pop_front();
                        input_in.gyro
                                << imu_last.angular_velocity.x, imu_last.angular_velocity.y, imu_last.angular_velocity.z;
                        input_in.acc
                                << imu_last.linear_acceleration.x, imu_last.linear_acceleration.y, imu_last.linear_acceleration.z;

                        // angvel_avr<<0.5 * (imu_last.angular_velocity.x + imu_next.angular_velocity.x),
                        //             0.5 * (imu_last.angular_velocity.y + imu_next.angular_velocity.y),
                        //             0.5 * (imu_last.angular_velocity.z + imu_next.angular_velocity.z);

                        // acc_avr   <<0.5 * (imu_last.linear_acceleration.x + imu_next.linear_acceleration.x),
                        //             0.5 * (imu_last.linear_acceleration.y + imu_next.linear_acceleration.y),
                        //             0.5 * (imu_last.linear_acceleration.z + imu_next.linear_acceleration.z);
                        input_in.acc = input_in.acc * G_m_s2 / acc_norm;
                        double dt = get_time_sec(imu_last.header.stamp) - t_last;

                        // if(!prop_at_freq_of_imu)
                        // {       
                        double dt_cov = get_time_sec(imu_last.header.stamp) - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = get_time_sec(imu_last.header.stamp); //time_current;
                        }
                        kf_input.predict(dt, Q_input, input_in, true, false);
                        t_last = get_time_sec(imu_last.header.stamp);
                        imu_prop_cov = true;
                        // imu_upda_cov = true;
                    }

                    double dt = time_current - t_last;
                    t_last = time_current;
                    double propag_start = omp_get_wtime();

                    if (!prop_at_freq_of_imu) {
                        double dt_cov = time_current - time_update_last;
                        if (dt_cov > 0.0) {
                            kf_input.predict(dt_cov, Q_input, input_in, false, true);
                            time_update_last = time_current;
                        }
                    }
                    kf_input.predict(dt, Q_input, input_in, true, false);

                    propag_time += omp_get_wtime() - propag_start;

                    // if(k == 0)
                    // {
                    //     fout_imu_pbp << Measures.lidar_last_time - first_lidar_time << " " << imu_last.angular_velocity.x << " " << imu_last.angular_velocity.y << " " << imu_last.angular_velocity.z \
                    //             << " " << imu_last.linear_acceleration.x << " " << imu_last.linear_acceleration.y << " " << imu_last.linear_acceleration.z << endl;
                    // }

                    double t_update_start = omp_get_wtime();

                    if (feats_down_size < 1) {
                        RCLCPP_WARN(logger, "No point, skip this scan!\n");

                        idx += time_seq[k];
                        continue;
                    }
                    if (!kf_input.update_iterated_dyn_share_modified()) {
                        idx = idx + time_seq[k];
                        continue;
                    }

                    solve_start = omp_get_wtime();

                    // if(prop_at_freq_of_imu)
                    // {
                    //     double dt_cov = time_current - time_update_last;
                    //     if ((imu_prop_cov && dt_cov > 0.0) || (dt_cov >= imu_time_inte * 1.2)) 
                    //     {
                    //         double propag_cov_start = omp_get_wtime();
                    //         kf_input.predict(dt_cov, Q_input, input_in, false, true); 
                    //         propag_time += omp_get_wtime() - propag_cov_start;
                    //         time_update_last = time_current;
                    //         imu_prop_cov = false;
                    //     }
                    // }
                    if (publish_odometry_without_downsample) {
                        /******* Publish odometry *******/

                        publish_odometry(pubOdomAftMapped, tf_broadcaster);
                        if (runtime_pos_log) {
                            state_in = kf_input.x_;
                            euler_cur = SO3ToEuler(state_in.rot);
                            fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                     << euler_cur.transpose() << " " << state_in.pos.transpose() << " "
                                     << state_in.vel.transpose() << " " << state_in.bg.transpose() << " "
                                     << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " "
                                     << feats_undistort->points.size() << endl;
                        }
                    }

                    for (int j = 0; j < time_seq[k]; j++) {
                        PointType &point_body_j = feats_down_body->points[idx + j + 1];
                        PointType &point_world_j = feats_down_world->points[idx + j + 1];
                        pointBodyToWorld(&point_body_j, &point_world_j);
                    }
                    solve_time += omp_get_wtime() - solve_start;

                    update_time += omp_get_wtime() - t_update_start;
                    idx = idx + time_seq[k];
                }
            }

            /******* Publish odometry downsample *******/
            if (!publish_odometry_without_downsample) {
                publish_odometry(pubOdomAftMapped, tf_broadcaster);
            }

            /*** add the feature points to map kdtree ***/
            t3 = omp_get_wtime();

            // Only while still searching. After the lock the prior map is
            // READ-ONLY: adding live scans would let drift contaminate the very
            // thing being localized against.
            if (feats_down_size > 4 && !global_update) {
                map_incremental();
            }

            /*** Feed the init thread while still searching: the downsampled
             *** scan for ScanContext, and the odometry pose it was taken at, so
             *** a lock found several scans later can be carried forward. The id
             *** queued alongside the cloud INDEXES these vectors, so the trail
             *** and the queue are appended under one lock -- appending outside
             *** it let the search thread race /relocalize's clear. ***/
            {
                std::unique_lock<std::mutex> lk(init_state_mutex);
                const bool searching = !global_localization_finish;
                lk.unlock();
                if (searching) {
                    PointCloudXYZI::Ptr snapshot(new PointCloudXYZI());
                    pcl::copyPointCloud(*feats_down_body, *snapshot);
                    V3D p_now;
                    Eigen::Quaterniond q_now;
                    if (use_imu_as_input) {
                        p_now = kf_input.x_.pos;
                        q_now = Eigen::Quaterniond(kf_input.x_.rot.normalized().toRotationMatrix());
                    } else {
                        p_now = kf_output.x_.pos;
                        q_now = Eigen::Quaterniond(kf_output.x_.rot.normalized().toRotationMatrix());
                    }
                    std::lock_guard<std::mutex> flk(init_feats_mutex);
                    position_init.push_back(p_now);
                    pose_init.push_back(q_now);
                    // Bounded: ScanContext plus two ICP passes is slower than
                    // the scan rate, so an unbounded queue grows without limit.
                    if (init_feats_down_bodys.size() < 10) {
                        init_feats_down_bodys.push({init_count, snapshot});
                    }
                    init_count++;
                }
            }

            /*** Post-lock health check. Re-scores the CURRENT pose against the
             *** prior map with the same map_overlap() used during init, because
             *** a wrong-but-self-consistent lock has no other symptom -- a
             *** self-similar place still produces plausible correspondences at
             *** the wrong place. Only SUSTAINED low overlap is acted on: a
             *** single dip is normal when turning into unmapped space or when
             *** someone crosses the scan. No timer here -- Point-LIO's main()
             *** owns the loop, so this is a time check inside it. ***/
            if (global_update && map_loaded) {
                const rclcpp::Time now_t = nh->get_clock()->now();
                if ((now_t - last_health_check).seconds() >= health_check_period) {
                    last_health_check = now_t;
                    Eigen::Matrix4d T_body_now = Eigen::Matrix4d::Identity();
                    Eigen::Matrix4d T_i_l = Eigen::Matrix4d::Identity();
                    if (use_imu_as_input) {
                        T_body_now.block<3,3>(0,0) = kf_input.x_.rot.normalized().toRotationMatrix();
                        T_body_now.block<3,1>(0,3) = kf_input.x_.pos;
                        T_i_l.block<3,3>(0,0) = kf_input.x_.offset_R_L_I.normalized().toRotationMatrix();
                        T_i_l.block<3,1>(0,3) = kf_input.x_.offset_T_L_I;
                    } else {
                        T_body_now.block<3,3>(0,0) = kf_output.x_.rot.normalized().toRotationMatrix();
                        T_body_now.block<3,1>(0,3) = kf_output.x_.pos;
                        T_i_l.block<3,3>(0,0) = kf_output.x_.offset_R_L_I.normalized().toRotationMatrix();
                        T_i_l.block<3,1>(0,3) = kf_output.x_.offset_T_L_I;
                    }
                    const double ov = map_overlap(feats_down_body, T_body_now * T_i_l);
                    std_msgs::msg::Float32 ov_msg;
                    ov_msg.data = static_cast<float>(ov);
                    pub_overlap_g->publish(ov_msg);
                    char ov_str[16];
                    std::snprintf(ov_str, sizeof(ov_str), "%.2f", ov);

                    if (ov < health_min_overlap) {
                        if (!overlap_bad) { overlap_bad = true; bad_since = now_t; }
                        const double bad_for = (now_t - bad_since).seconds();
                        RCLCPP_WARN(nh->get_logger(),
                            "[health] overlap %.0f%% (need %.0f%%), bad for %.1f s (limit %.1f)",
                            100.0 * ov, 100.0 * health_min_overlap, bad_for, health_bad_duration);
                        if (auto_relocalize && bad_for >= health_bad_duration) {
                            RCLCPP_ERROR(nh->get_logger(),
                                "[health] overlap stayed below %.0f%% for %.1f s -- this "
                                "lock looks wrong. Re-arming the global search.",
                                100.0 * health_min_overlap, bad_for);
                            publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                                "Lock looks wrong; auto-relocalize re-armed the search", ov_str);
                            rearm_search();
                            overlap_bad = false;
                        } else {
                            publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                                "Overlap below threshold; watching before acting", ov_str);
                        }
                    } else {
                        overlap_bad = false;
                        publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::OK,
                            "Localized; tracking the prior map", ov_str);
                    }
                }
            }

            t5 = omp_get_wtime();
            /******* Publish points *******/
            if (path_en) publish_path(pubPath);
            if (scan_pub_en || pcd_save_en) publish_frame_world(pubLaserCloudFullRes);
            if (scan_pub_en && scan_body_pub_en) publish_frame_body(pubLaserCloudFullRes_body);

            /*** Debug variables Logging ***/
            if (runtime_pos_log) {
                frame_num++;
                aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
                { aver_time_icp = aver_time_icp * (frame_num - 1) / frame_num + update_time / frame_num; }
                aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
                aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + solve_time / frame_num;
                aver_time_propag = aver_time_propag * (frame_num - 1) / frame_num + propag_time / frame_num;
                T1[time_log_counter] = Measures.lidar_beg_time;
                s_plot[time_log_counter] = t5 - t0;
                s_plot2[time_log_counter] = feats_undistort->points.size();
                s_plot3[time_log_counter] = aver_time_consu;
                time_log_counter++;
                printf("[ mapping ]: time: IMU + Map + Input Downsample: %0.6f ave match: %0.6f ave solve: %0.6f  ave ICP: %0.6f  map incre: %0.6f ave total: %0.6f icp: %0.6f propogate: %0.6f \n",
                       t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp,
                       aver_time_propag);
                if (!publish_odometry_without_downsample) {
                    if (!use_imu_as_input) {
                        state_out = kf_output.x_;
                        euler_cur = SO3ToEuler(state_out.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                 << euler_cur.transpose() << " " << state_out.pos.transpose() << " "
                                 << state_out.vel.transpose() << " " << state_out.omg.transpose() << " "
                                 << state_out.acc.transpose() << " " << state_out.gravity.transpose() << " "
                                 << state_out.bg.transpose() << " " << state_out.ba.transpose() << " "
                                 << feats_undistort->points.size() << endl;
                    } else {
                        state_in = kf_input.x_;
                        euler_cur = SO3ToEuler(state_in.rot);
                        fout_out << setw(20) << Measures.lidar_beg_time - first_lidar_time << " "
                                 << euler_cur.transpose() << " " << state_in.pos.transpose() << " "
                                 << state_in.vel.transpose() << " " << state_in.bg.transpose() << " "
                                 << state_in.ba.transpose() << " " << state_in.gravity.transpose() << " "
                                 << feats_undistort->points.size() << endl;
                    }
                }
                dump_lio_state_to_log(fp);
            }
        }
        rate.sleep();
    }
    //--------------------------save map-----------------------------------
    /* 1. make sure you have enough memories
       2. noted that pcd save will influence the real-time performences **/
    if (pcl_wait_save->size() > 0 && pcd_save_en) {
        // map_file_path empty => upstream default (ROOT_DIR/PCD/scans.pcd,
        // i.e. inside the source tree). Set it to write somewhere real --
        // same knob as FAST-LIO's map_file_path.
        string all_points_dir = map_file_path.empty()
            ? string(string(ROOT_DIR) + "PCD/scans.pcd")
            : map_file_path;
        std::filesystem::path out_path(all_points_dir);
        if (out_path.has_parent_path()) {
            std::error_code ec;
            std::filesystem::create_directories(out_path.parent_path(), ec);
        }
        pcl::PCDWriter pcd_writer;
        pcd_writer.writeBinary(all_points_dir, *pcl_wait_save);
        std::cout << "Point-LIO saved accumulated map (" << pcl_wait_save->size()
                  << " pts) to " << all_points_dir << std::endl;
    }
    fout_out.close();
    fout_imu_pbp.close();

    // Stop and JOIN the search thread before main returns. Without this, a
    // still-joinable std::thread is destroyed at scope exit and the process
    // dies with "terminate called without an active exception" on every
    // Ctrl-C -- which is exactly what it did before this was added. The
    // thread's outer loop checks keep_searching before it checks whether a
    // lock exists, so clearing it is enough to make it return within one tick.
    {
        std::lock_guard<std::mutex> lk(init_state_mutex);
        keep_searching = false;
    }
    if (init_thread.joinable()) init_thread.join();

    return 0;
}
