// =============================================================================
//  point_lio_localization -- Point-LIO running against a prior map.
//
//  The prior map is loaded straight into the ikd-Tree the filter registers
//  against, so it constrains the estimate from INSIDE the filter at scan rate.
//  There is no map->odom correction step, and so nothing to jump. ScanContext
//  plus two-stage ICP finds the initial pose, gated on agreement between
//  independent estimates, on map overlap, and optionally on motion.
//
//  Structure:
//    pointlio_core.hpp         estimator core, shared with point_lio_mapping
//    localization_search.hpp   prior map, the search, health reporting
//    this file                 localization globals, publish_odometry, main()
//
//  Differences from the FAST_LIO twin (FAST_LIO/src/laserLocalization.cpp,
//  which carries the longer reasoning):
//    * The handover writes into kf_input or kf_output depending on
//      use_imu_as_input. omg and acc are BODY frame and carry over untouched,
//      as bg/ba do; only pos, rot, vel and gravity are world frame.
//    * gravity is a plain vect3 here, not FAST-LIO's S2 manifold, so it is
//      rotated directly.
//    * No node class or timer: main() owns a 5 kHz spin loop, so the health
//      check is a time check inside that loop.
//
//  Deliberately not sharing an estimator with FAST_LIO: Point-LIO carries two
//  filter types and runs a point-by-point update, so a common abstraction
//  would have to straddle both and serve neither.
// =============================================================================
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
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/vector3.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
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
// SHARED into ikdtree on lock, never moved. It used to be std::move'd, which
// emptied it: after a re-arm the prior map was gone and could not be handed
// back, so the filter was left holding a map it could not use and unable to
// rebuild one. Sharing keeps it available for every subsequent lock.
KD_TREE<PointType>::Ptr ikdtree_global(new KD_TREE<PointType>());
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
bool keep_searching = true;                 // cleared on shutdown to stop the thread
// Set once max_relock_attempts locks have been claimed and then lost. The
// search thread idles rather than re-finding the same aliased place forever;
// only /initialpose or /relocalize clears it.
bool search_halted = false;
// Latched at the first handover: separates "still looking, never had a pose"
// (WARN) from "had a pose and lost it" (ERROR), which the Nav2 watchdog acts
// on differently.
bool has_ever_locked = false;
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
double init_min_overlap = 0.70;
double init_overlap_dist = 0.12;
// A candidate is scored against this many CONSECUTIVE live scans before the
// filter is teleported and the prior map swapped in. Until then nothing is
// committed, so rejecting one costs nothing.
int    lock_verify_scans = 3;
int    lock_verify_passes = 0;
// Seconds of consecutive effct_feat_num == 0 before a lock is dropped. With no
// correspondences the filter applies no update at all and runs open-loop on
// IMU, where position error grows quadratically -- standing still does not
// help. Far shorter than the health check, which is tuned for a lock that is
// wrong but still trackable.
double no_match_duration = 1.0;
bool   no_match_bad = false;
rclcpp::Time no_match_since;
// Locks claimed and then lost in a row before the automatic search gives up
// and waits for a manual /initialpose. 0 retries forever.
int    relock_attempts = 0;
int    max_relock_attempts = 2;
// Metres/second the platform cannot exceed (a Pepper does ~0.55). Above this
// the estimate is broken whatever produced it.
double max_speed = 1.0;
// Covariance pos/rot are reset to at a handover, m^2 and rad^2 (~0.7 m /
// ~11 deg one-sigma). change_x() alone leaves P at the pre-jump track's
// confidence, so the filter resists the map correcting whatever error
// remains in the jump. The old handover never reopened P at all.
double seed_pos_cov = 0.5;
double seed_rot_cov = 0.04;
double prior_map_view_leaf = 0.20;
// Post-lock health check (see the FAST_LIO file): a wrong-but-self-consistent
// lock has no other symptom, because a self-similar place still produces
// plausible matches at the wrong place.
double health_min_overlap = 0.45;
double health_bad_duration = 5.0;
double health_check_period = 1.0;
// auto_relocalize REMOVED. It re-armed the search while the robot kept
// driving, and the next handover inherited the velocity from that
// unconstrained window at full confidence, so attempts compounded instead of
// converging. The health check now reports only; recovery belongs to the Nav2
// watchdog, which stops the robot first.
bool   overlap_bad = false;
rclcpp::Time bad_since;
// map -> base_footprint, not map -> the IMU on the mast: REP-105, and the
// bag's /tf_static already owns base_footprint -> the IMU frame, so
// broadcasting that edge here too would give it two parents.
std::string tf_child_frame;
// /Odometry's frame_id BEFORE a lock (and again after a re-arm). The filter is
// in its own arbitrary-origin, mount-tilted frame then, NOT the map: stamping
// it as the map put an unverified, drifting pose on the wire labelled as if it
// were map-registered. Matches the mapping stack's name for the same concept.
std::string odom_init_frame = "lio_init";
std::shared_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_broadcaster_g;
bool  tf_child_resolved = false;
M3D   R_body_to_tfchild(Eye3d);
V3D   t_body_to_tfchild(0, 0, 0);
std::shared_ptr<tf2_ros::Buffer> tf_buffer_g;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_g;
rclcpp::Node *node_g = nullptr;
rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_localization_g;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pubPriorMap;  // lifted out of main()
rclcpp::Publisher<std_msgs::msg::Float32>::SharedPtr pub_overlap_g;
rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_diag_g;
// What the search is CURRENTLY testing, so the hunt is visible instead of
// being a silent gap between "searching" and "Localized". candidate_scan is
// the live scan placed AT the candidate pose: laid over /prior_map it shows at
// a glance whether the alignment is real, which a percentage does not convey.
rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pub_cand_pose_g;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_cand_scan_g;

/* Logger for the free functions here and in localization_search.hpp, which
   have no node handle. Falls back to a named logger before on_configure. */
static rclcpp::Logger this_logger() {
    return node_g ? node_g->get_logger() : rclcpp::get_logger("point_lio_localization");
}

#include "pointlio_core.hpp"

// The estimator core is shared with point_lio_mapping and lives in
// pointlio_core.hpp; it used to be duplicated here verbatim. Below: the prior map
// and its ScanContext DB, publish_odometry (which unlike mapping's also emits
// /localization/pose and diagnostics), the initial-pose search thread, and
// main().

/* IMU <- lidar. Read from the filter state ONLY when the filter is actually
   estimating the extrinsic: pointlio_core.hpp seeds offset_R_L_I/offset_T_L_I
   under extrinsic_est_en and leaves them identity/zero otherwise, so reading
   them unconditionally silently substitutes identity for a real mount. Same
   rule pointBodyLidarToIMU() and the h_model loop use. */
static Eigen::Matrix4d imu_T_lidar()
{
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    if (extrinsic_est_en) {
        if (use_imu_as_input) {
            T.block<3,3>(0,0) = kf_input.x_.offset_R_L_I.normalized().toRotationMatrix();
            T.block<3,1>(0,3) = kf_input.x_.offset_T_L_I;
        } else {
            T.block<3,3>(0,0) = kf_output.x_.offset_R_L_I.normalized().toRotationMatrix();
            T.block<3,1>(0,3) = kf_output.x_.offset_T_L_I;
        }
    } else {
        T.block<3,3>(0,0) = Lidar_R_wrt_IMU;
        T.block<3,1>(0,3) = Lidar_T_wrt_IMU;
    }
    return T;
}

/* Publish the tracked pose. Unlike the mapping node's, this also broadcasts
   map -> tf_child_frame and emits /localization/pose in that frame, with the
   twist corrected for the lever arm between the IMU and the robot base. */
void publish_odometry(const rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr &pubOdomAftMapped,
                      std::shared_ptr<tf2_ros::TransformBroadcaster> &tf_br) {

    // Not the map frame until locked: see odom_init_frame's comment.
    odomAftMapped.header.frame_id = global_update ? odom_header_frame_id : odom_init_frame;
    odomAftMapped.child_frame_id = odom_child_frame_id;
    // The shared cloud/path publishers (pointlio_core.hpp) stamp with
    // world_pub_frame; mirror it so /cloud_registered and /path carry the same
    // honest frame as /Odometry this scan. Lock state rides along as intensity:
    // the live scan reads RED while searching and GREEN once locked.
    world_pub_frame = odomAftMapped.header.frame_id;
    world_pub_intensity = global_update ? 100.0f : 0.0f;

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



#include "localization_search.hpp"

/*** Apply a map<-IMU pose to whichever filter is running.
 ***
 *** The handover is a change of WORLD FRAME, not a pose edit, so pos, rot,
 *** vel and gravity all rotate with it -- leaving gravity behind points it
 *** sideways and the filter diverges within a few scans. omg/acc/bg/ba are
 *** BODY frame and carry over untouched.
 ***
 *** The covariance is reopened as well: change_x() alone leaves P at the
 *** pre-jump track's confidence, so the filter RESISTS the map correcting
 *** whatever error remains in the jump. ***/
void apply_map_pose(const Eigen::Matrix4d &T_map_now)
{
    const M3D R_new = T_map_now.block<3,3>(0,0);
    if (use_imu_as_input) {
        state_input gs = kf_input.x_;
        const M3D R_delta = R_new * gs.rot.normalized().toRotationMatrix().transpose();
        gs.pos = T_map_now.block<3,1>(0,3);
        gs.rot = R_new;
        gs.vel = R_delta * gs.vel;
        gs.gravity = R_delta * gs.gravity;
        kf_input.change_x(gs);
        auto P = kf_input.get_P();
        P.block<3,3>(0,0) = M3D::Identity() * seed_pos_cov;   // pos
        P.block<3,3>(3,3) = M3D::Identity() * seed_rot_cov;   // rot
        kf_input.change_P(P);
        state_in = kf_input.x_;
    } else {
        state_output gs = kf_output.x_;
        const M3D R_delta = R_new * gs.rot.normalized().toRotationMatrix().transpose();
        gs.pos = T_map_now.block<3,1>(0,3);
        gs.rot = R_new;
        gs.vel = R_delta * gs.vel;
        gs.gravity = R_delta * gs.gravity;
        kf_output.change_x(gs);
        auto P = kf_output.get_P();
        P.block<3,3>(0,0) = M3D::Identity() * seed_pos_cov;
        P.block<3,3>(3,3) = M3D::Identity() * seed_rot_cov;
        kf_output.change_P(P);
        state_out = kf_output.x_;
    }
}

/*** The filter's current pose in its OWN frame, and the live IMU<-lidar
 *** extrinsic, whichever filter is running. ***/
void current_body_and_extrinsic(Eigen::Matrix4d &T_body, Eigen::Matrix4d &T_i_l)
{
    T_body = Eigen::Matrix4d::Identity();
    T_i_l  = Eigen::Matrix4d::Identity();
    if (use_imu_as_input) {
        T_body.block<3,3>(0,0) = kf_input.x_.rot.normalized().toRotationMatrix();
        T_body.block<3,1>(0,3) = kf_input.x_.pos;
        T_i_l.block<3,3>(0,0) = kf_input.x_.offset_R_L_I.normalized().toRotationMatrix();
        T_i_l.block<3,1>(0,3) = kf_input.x_.offset_T_L_I;
    } else {
        T_body.block<3,3>(0,0) = kf_output.x_.rot.normalized().toRotationMatrix();
        T_body.block<3,1>(0,3) = kf_output.x_.pos;
        T_i_l.block<3,3>(0,0) = kf_output.x_.offset_R_L_I.normalized().toRotationMatrix();
        T_i_l.block<3,1>(0,3) = kf_output.x_.offset_T_L_I;
    }
}

double current_speed()
{
    return use_imu_as_input ? kf_input.x_.vel.norm() : kf_output.x_.vel.norm();
}

/*** Counts a lock CLAIMED and then LOST, and past max_relock_attempts halts
 *** the automatic search. Only real losses count -- a candidate rejected in
 *** verification never became a lock and cost nothing, and rejections are
 *** routine, so counting those would halt almost immediately. ***/
void record_relock_attempt()
{
    relock_attempts++;
    if (max_relock_attempts > 0 && relock_attempts >= max_relock_attempts) {
        {
            std::lock_guard<std::mutex> lk(init_state_mutex);
            search_halted = true;
        }
        RCLCPP_ERROR(this_logger(),
            "[health] %d locks claimed and lost in a row -- giving up on the "
            "automatic search. This place is aliasing badly enough that "
            "retrying just finds it again. SET THE POSE MANUALLY: use RViz's "
            "2D Pose Estimate (/initialpose), or call /relocalize.",
            relock_attempts);
        publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
            "Search halted; waiting for a manual /initialpose");
    }
}

/*** Score a pending ScanContext lock against the LIVE scan and apply it only
 *** once it has passed lock_verify_scans times in a row.
 ***
 *** Nothing touches the filter until the candidate has earned it. Applying on
 *** announcement is a one-way bet placed before the evidence is in: if the
 *** candidate is wrong the filter loses the live map it was tracking against
 *** and gains one it cannot register from that pose, every correspondence
 *** fails, the iEKF applies no update, and the state runs open-loop on IMU
 *** and accelerates away. Rejecting is cheap precisely because nothing was
 *** given up -- the filter keeps tracking throughout. ***/
void verify_and_apply_lock()
{
    int id = -1;
    Eigen::Matrix4d T_cand = Eigen::Matrix4d::Identity();
    bool locked = false;
    {
        std::lock_guard<std::mutex> lk(init_state_mutex);
        locked = global_localization_finish;
        if (locked) { id = init_result.first; T_cand = init_result.second; }
    }
    if (!locked || global_update) return;
    if (feats_down_body->empty()) return;

    // Bounds-checked: a re-arm clears the trail and restarts ids at 0, so an
    // id from before it must not index the new one.
    Eigen::Matrix4d T_odom_lock;
    if (!odom_at(id, T_odom_lock)) {
        RCLCPP_WARN(this_logger(),
            "[init] lock references trail entry %d but the trail was re-armed "
            "underneath it; discarding.", id);
        rearm_search();
        return;
    }

    Eigen::Matrix4d T_odom_now, T_i_l;
    current_body_and_extrinsic(T_odom_now, T_i_l);

    // Where the candidate says we are NOW, carried forward through the
    // odometry since its scan. Re-derived every scan, so each check is against
    // a fresh observation rather than the one that produced the candidate.
    const Eigen::Matrix4d T_map_now = T_cand * T_odom_lock.inverse() * T_odom_now;
    const Eigen::Matrix4d T_map_lidar = T_map_now * T_i_l;

    // Publish what is being tested BEFORE deciding, so a rejected candidate is
    // still seen: in RViz the candidate visibly hops as places are tried.
    if (pub_cand_pose_g && pub_cand_scan_g) {
        geometry_msgs::msg::PoseStamped cp;
        cp.header.stamp = get_ros_time(lidar_end_time);
        cp.header.frame_id = odom_header_frame_id;
        const Eigen::Quaterniond q(M3D(T_map_now.block<3,3>(0,0)));
        cp.pose.position.x = T_map_now(0,3);
        cp.pose.position.y = T_map_now(1,3);
        cp.pose.position.z = T_map_now(2,3);
        cp.pose.orientation.w = q.w(); cp.pose.orientation.x = q.x();
        cp.pose.orientation.y = q.y(); cp.pose.orientation.z = q.z();
        pub_cand_pose_g->publish(cp);

        PointCloudXYZI::Ptr at_cand(new PointCloudXYZI());
        pcl::transformPointCloud(*feats_down_body, *at_cand, T_map_lidar.cast<float>());
        sensor_msgs::msg::PointCloud2 cs;
        pcl::toROSMsg(*at_cand, cs);
        cs.header.stamp = cp.header.stamp;
        cs.header.frame_id = odom_header_frame_id;
        pub_cand_scan_g->publish(cs);
    }

    const double ov = map_overlap(feats_down_body, T_map_lidar);
    if (ov < init_min_overlap) {
        RCLCPP_WARN(this_logger(),
            "[init] candidate REJECTED after %d/%d checks: only %.0f%% of the "
            "live scan lands on the map at the pose it predicts (need %.0f%%). "
            "The map was never swapped and the filter is still tracking "
            "normally; searching again.",
            lock_verify_passes, lock_verify_scans, 100.0 * ov,
            100.0 * init_min_overlap);
        reject_candidate();
        return;
    }

    lock_verify_passes++;
    if (lock_verify_passes < lock_verify_scans) {
        RCLCPP_INFO(this_logger(),
            "[init] verifying candidate: %d/%d at %.0f%% overlap",
            lock_verify_passes, lock_verify_scans, 100.0 * ov);
        return;
    }

    apply_map_pose(T_map_now);
    // Shared, not moved: ikdtree_global keeps the prior map for the next lock
    // after a re-arm.
    ikdtree = ikdtree_global;
    map_swapped = true;
    init_map = true;
    global_update = true;
    has_ever_locked = true;
    lock_verify_passes = 0;
    RCLCPP_INFO(this_logger(),
        "Localized: filter is now in the map frame at x=%.2f y=%.2f z=%.2f "
        "(verified over %d scans, %.0f%% overlap); prior map is read-only "
        "from here.",
        T_map_now(0,3), T_map_now(1,3), T_map_now(2,3),
        lock_verify_scans, 100.0 * ov);
}


// The prior-map loading and the initial-pose search (ScanContext + ICP +
// the agreement/overlap/motion gates) live there. Below: main().



/* Node entry point, in four phases: localization parameters and frame setup,
   the shared setup_common(), the prior-map bring-up that must complete before
   any scan is processed, then the 5 kHz spin loop -- handover, process_scan(),
   init-thread feed and health check -- and shutdown_common(). */
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

    nh->declare_parameter<double>("localization.init_min_overlap", 0.70);
    nh->declare_parameter<double>("localization.init_overlap_dist", 0.20);
    nh->declare_parameter<double>("localization.prior_map_view_leaf", 0.20);
    nh->declare_parameter<int>("localization.lock_verify_scans", 3);
    nh->get_parameter("localization.lock_verify_scans", lock_verify_scans);
    nh->declare_parameter<double>("localization.no_match_duration", 1.0);
    nh->get_parameter("localization.no_match_duration", no_match_duration);
    nh->declare_parameter<int>("localization.max_relock_attempts", 2);
    nh->get_parameter("localization.max_relock_attempts", max_relock_attempts);
    nh->declare_parameter<double>("localization.max_speed", 1.0);
    nh->get_parameter("localization.max_speed", max_speed);
    nh->declare_parameter<double>("localization.seed_pos_cov", 0.5);
    nh->get_parameter("localization.seed_pos_cov", seed_pos_cov);
    nh->declare_parameter<double>("localization.seed_rot_cov", 0.04);
    nh->get_parameter("localization.seed_rot_cov", seed_rot_cov);
    nh->declare_parameter<double>("localization.health_min_overlap", 0.45);
    nh->declare_parameter<double>("localization.health_bad_duration", 5.0);
    nh->declare_parameter<double>("localization.health_check_period", 1.0);
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

    nh->get_parameter("localization.init_min_overlap", init_min_overlap);
    nh->get_parameter("localization.init_overlap_dist", init_overlap_dist);
    nh->get_parameter("localization.prior_map_view_leaf", prior_map_view_leaf);
    nh->get_parameter("localization.health_min_overlap", health_min_overlap);
    nh->get_parameter("localization.health_bad_duration", health_bad_duration);
    nh->get_parameter("localization.health_check_period", health_check_period);
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
    nh->declare_parameter<std::string>("publish.odom_init_frame", "lio_init");
    nh->get_parameter("publish.odom_init_frame", odom_init_frame);
    world_pub_frame = odom_init_frame;   // sane value before the first scan
    // ANCHOR THE MAP FRAME so RViz has a Fixed Frame from startup. Without it
    // the map frame does not exist in TF until the first lock, and RViz draws
    // NOTHING -- not even /prior_map -- during the one phase worth watching.
    // This edge is map -> odom_init_frame ONLY, and identity: nothing
    // publishes odom_init_frame -> body until a lock, so no consumer can
    // resolve a map-relative robot pose from it and wait_for_map_then_start
    // still blocks on the full chain.
    static_tf_broadcaster_g = std::make_shared<tf2_ros::StaticTransformBroadcaster>(nh);
    {
        geometry_msgs::msg::TransformStamped anchor;
        anchor.header.stamp = nh->get_clock()->now();
        anchor.header.frame_id = odom_header_frame_id;
        anchor.child_frame_id = odom_init_frame;
        anchor.transform.rotation.w = 1.0;
        static_tf_broadcaster_g->sendTransform(anchor);
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
    // Filter init, extrinsics, process noise, debug logs, subs and pubs --
    // shared with the other node; see pointlio_core.hpp.
    setup_common(nh);

    /*** LOCALIZATION bring-up. The prior map is loaded and the search armed
     *** BEFORE the scan loop starts, so no scan is processed against an empty
     *** map. A missing map is fatal: this node localizes, it has nothing to do
     *** without one. ***/
    pub_localization_g = nh->create_publisher<nav_msgs::msg::Odometry>("/localization/pose", 10);
    pub_overlap_g = nh->create_publisher<std_msgs::msg::Float32>("/localization/overlap", 10);
    pub_diag_g = nh->create_publisher<diagnostic_msgs::msg::DiagnosticArray>("/diagnostics", 10);
    pub_cand_pose_g = nh->create_publisher<geometry_msgs::msg::PoseStamped>(
        "/localization/candidate_pose", 10);
    pub_cand_scan_g = nh->create_publisher<sensor_msgs::msg::PointCloud2>(
        "/localization/candidate_scan", 10);

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
    ikdtree_global->set_downsample_param(filter_size_map_min);
    ikdtree_global->Build(global_map->points);
    map_loaded = true;
    RCLCPP_INFO(nh->get_logger(),
        "Prior map ready (%zu pts). Searching for initial pose: ScanContext + "
        "ICP, %d estimates must agree within %.2f m.",
        global_map->size(), init_agree_count, init_agree_dist);

    // Latched (transient local) so RViz shows it on connect rather than only
    // if it happens to be listening at startup. Downsampled for display only.
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
            // An explicit operator request for another try gets a fresh
            // budget -- otherwise a single /relocalize after the search gave
            // up walks straight back into search_halted on the next failure.
            relock_attempts = 0;
            {
                std::lock_guard<std::mutex> lk(init_state_mutex);
                search_halted = false;
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
                search_halted = false;               // operator pose: clean slate
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
            /*** THE HANDOVER: move the filter into map coordinates and give it
             *** the prior map. From here the estimate IS the map pose.
             ***
             *** The lock names a past scan, so carry it forward by the odometry
             *** since:  T_map_now = T_map_at_lock * T_odom_at_lock^-1 * T_odom_now
             *** A /initialpose seed is applied directly instead, no carry forward.
             ***
             *** This is a change of WORLD FRAME, not a pose edit, so pos, rot, vel
             *** and gravity all rotate with it -- leaving gravity behind points it
             *** sideways and the filter diverges within a few scans. omg/acc are
             *** body frame and carry over untouched, as bg/ba do. ***/
            // Only the OPERATOR seed teleports here, before the scan is
            // processed: an /initialpose is an assertion about where the robot
            // is, so it is applied immediately and not second-guessed. A
            // ScanContext lock is NOT applied here -- it is verified against
            // live scans first, after this scan's update (see
            // verify_and_apply_lock, called below).
            if (has_pending_seed.exchange(false)) {
                Eigen::Matrix4d T_map_now;
                {
                    std::lock_guard<std::mutex> lk(seed_mutex);
                    T_map_now = pending_seed;
                }
                apply_map_pose(T_map_now);
                // UNCONDITIONAL tree reset: whatever tree was in use describes
                // the OLD belief -- pre-lock a live map around a pose now
                // declared wrong, post-lock the prior map positioned by a lock
                // being overridden. Gating this on !map_swapped left the
                // previous tree in place for a seed issued while locked.
                ikdtree = ikdtree_global;
                map_swapped = true;
                init_map = true;
                global_update = true;
                has_ever_locked = true;
                lock_verify_passes = 0;
                relock_attempts = 0;
                {
                    std::lock_guard<std::mutex> lk(init_state_mutex);
                    search_halted = false;
                }
                RCLCPP_INFO(nh->get_logger(),
                    "Seeded from /initialpose: filter is now at x=%.2f y=%.2f "
                    "z=%.2f; previous map discarded, prior map in use.",
                    T_map_now(0,3), T_map_now(1,3), T_map_now(2,3));
            }

            // The whole scan pipeline; see pointlio_core.hpp. Returns false
            // where this loop body used to `continue`.
            if (!process_scan()) continue;

            // PHYSICAL PLAUSIBILITY GUARD -- velocity AND the IMU states are
            // ZEROED, not scaled.
            //
            // UNCONDITIONAL. Scoping this to the open-loop case looks right --
            // a corrected velocity is observed and should not be touched --
            // but wrong correspondences produce a wrong velocity just as
            // readily as no correspondences do, and then the clamp never runs.
            // A Pepper tops out near 0.55 m/s, so above max_speed the estimate
            // is broken whatever produced it and zero is a far better guess
            // than the diverging value. This can never clip real motion, and
            // it kills the quadratic position runaway at its source, since
            // position only integrates velocity.
            if (current_speed() > max_speed) {
                RCLCPP_WARN_THROTTLE(nh->get_logger(), *nh->get_clock(), 1000,
                    "[health] velocity %.1f m/s exceeds this robot's %.2f m/s "
                    "limit (effective points: %d) -- the estimate is diverging, "
                    "not moving. Velocity and IMU states reset to zero.",
                    current_speed(), max_speed, effct_feat_num);
                if (use_imu_as_input) {
                    state_input sv = kf_input.x_;
                    sv.vel = V3D(0, 0, 0);
                    // ba is what becomes phantom acceleration: propagation
                    // integrates (a_meas - ba + g), so a ba that has absorbed
                    // garbage rebuilds the runaway even after vel is zeroed.
                    sv.ba = V3D(0, 0, 0);
                    sv.bg = V3D(0, 0, 0);
                    kf_input.change_x(sv);
                    auto P = kf_input.get_P();
                    P.block<3,3>(12,12) = M3D::Identity() * 1.0;    // vel
                    P.block<3,3>(15,15) = M3D::Identity() * 1e-2;   // bg
                    P.block<3,3>(18,18) = M3D::Identity() * 1e-2;   // ba
                    kf_input.change_P(P);
                    state_in = kf_input.x_;
                } else {
                    state_output sv = kf_output.x_;
                    sv.vel = V3D(0, 0, 0);
                    // Point-LIO's output filter carries acc and omg as REAL
                    // states, not inputs, so unlike FAST-LIO there is an
                    // acceleration to reset here -- and it must be, or it
                    // drives the next propagation straight back out.
                    sv.acc = V3D(0, 0, 0);
                    sv.omg = V3D(0, 0, 0);
                    sv.ba = V3D(0, 0, 0);
                    sv.bg = V3D(0, 0, 0);
                    kf_output.change_x(sv);
                    auto P = kf_output.get_P();
                    P.block<3,3>(12,12) = M3D::Identity() * 1.0;    // vel
                    P.block<3,3>(15,15) = M3D::Identity() * 1e-2;   // omg
                    P.block<3,3>(18,18) = M3D::Identity() * 1e-2;   // acc
                    P.block<3,3>(24,24) = M3D::Identity() * 1e-2;   // bg
                    P.block<3,3>(27,27) = M3D::Identity() * 1e-2;   // ba
                    kf_output.change_P(P);
                    state_out = kf_output.x_;
                }
            }

            // ZERO-MATCH GUARD. effct_feat_num == 0 means no usable
            // correspondence for ANY point, so the filter applied no
            // correction: consecutive such scans are pure IMU propagation and
            // the pose accelerates away. Re-arming is what breaks the
            // trapdoor -- rearm_search hands back a fresh LIVE tree, so the
            // filter has something self-consistent to track again.
            if (effct_feat_num < 1) {
                const rclcpp::Time now_nm = nh->get_clock()->now();
                if (!no_match_bad) { no_match_bad = true; no_match_since = now_nm; }
                const double bad_for = (now_nm - no_match_since).seconds();
                if (bad_for >= no_match_duration) {
                    if (global_update) {
                        RCLCPP_ERROR(nh->get_logger(),
                            "[health] zero effective points for %.1f s -- this lock "
                            "cannot be tracked and the pose is diverging on IMU "
                            "alone. Dropping it and searching again.", bad_for);
                        publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                            "Zero-match lock dropped; searching again");
                        record_relock_attempt();
                    } else {
                        RCLCPP_WARN(nh->get_logger(),
                            "[init] zero effective points for %.1f s before ever "
                            "locking -- resetting local tracking so the odometry "
                            "trail a lock is carried forward through cannot be "
                            "corrupted by unbounded drift.", bad_for);
                    }
                    rearm_search();
                    no_match_bad = false;
                    continue;   // nothing valid left to do with this scan
                }
            } else {
                no_match_bad = false;
            }

            // Score a pending lock against THIS scan and apply it only once it
            // has held up. Runs here, after the update, because both inputs
            // are then current.
            verify_and_apply_lock();

            // Only while still searching. After the lock the prior map is
            // READ-ONLY: adding live scans would let drift contaminate the very
            // thing being localized against.
            // Gated on map_swapped, NOT global_update. rearm_search clears
            // global_update, so gating on it reopened the PRIOR map for
            // writing for the whole duration of a re-arm -- merging in exactly
            // the scans whose pose was just declared untrustworthy. While the
            // tree IS the prior map it stays read-only; a re-arm swaps in a
            // fresh live tree and only that is written.
            if (feats_down_size > 4 && !map_swapped) {
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
                    /*** Bounded, and biased to the NEWEST scans. ScanContext
                     *** plus two ICP passes are slower than the scan rate, so
                     *** dropping new arrivals left the search working through
                     *** the stalest scans in the buffer -- every lock then
                     *** landed far behind the robot and had to be extrapolated
                     *** through odometry to be usable. Evict the oldest
                     *** instead. Depth matches FAST-LIO's 5. ***/
                    while (init_feats_down_bodys.size() >= 5) {
                        init_feats_down_bodys.pop();
                    }
                    init_feats_down_bodys.push({init_count, snapshot});
                    init_count++;
                }
            }

            /*** Post-lock health check: re-score the current pose with the same
             *** map_overlap() used during init. A wrong-but-self-consistent lock
             *** has no other symptom. Only SUSTAINED low overlap is acted on --
             *** single dips are normal when turning into unmapped space. ***/
            // Unlocked is TWO situations and consumers act on them
            // differently. Never yet localized is normal startup -- nothing
            // lost, nothing wrong: WARN. Unlocked AFTER holding a lock means
            // the pose was thrown away and there is no map -> base_footprint
            // edge at all, so anything navigating on it must STOP: ERROR.
            // localization_watchdog treats only ERROR as lost.
            if (!global_update && map_loaded) {
                const rclcpp::Time now_u = nh->get_clock()->now();
                if ((now_u - last_health_check).seconds() >= health_check_period) {
                    last_health_check = now_u;
                    if (has_ever_locked) {
                        publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                            "Lock lost; searching again -- no map pose is being published");
                    } else {
                        publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                            "Searching for initial pose (not yet localized)");
                    }
                }
            }
            if (global_update && map_loaded) {
                const rclcpp::Time now_t = nh->get_clock()->now();
                if ((now_t - last_health_check).seconds() >= health_check_period) {
                    last_health_check = now_t;
                    Eigen::Matrix4d T_body_now = Eigen::Matrix4d::Identity();
                    if (use_imu_as_input) {
                        T_body_now.block<3,3>(0,0) = kf_input.x_.rot.normalized().toRotationMatrix();
                        T_body_now.block<3,1>(0,3) = kf_input.x_.pos;
                    } else {
                        T_body_now.block<3,3>(0,0) = kf_output.x_.rot.normalized().toRotationMatrix();
                        T_body_now.block<3,1>(0,3) = kf_output.x_.pos;
                    }
                    const Eigen::Matrix4d T_i_l = imu_T_lidar();
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
                        // Sustained, not momentary: escalate to ERROR so an
                        // operator sees it, but do NOT act. This used to
                        // re-arm the search itself, which freed the filter
                        // from the map while the hunt restarted and let the
                        // next handover inherit that velocity at full
                        // confidence, so attempts compounded. Recovery is now
                        // deliberate: /initialpose or /relocalize.
                        if (bad_for >= health_bad_duration) {
                            RCLCPP_ERROR(nh->get_logger(),
                                "[health] overlap stayed below %.0f%% for %.1f s -- this "
                                "lock looks wrong. Correct it with /initialpose, or call "
                                "/relocalize to search again.",
                                100.0 * health_min_overlap, bad_for);
                            publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                                "Lock looks wrong; waiting for /initialpose or /relocalize",
                                ov_str);
                        } else {
                            publish_diagnostic(diagnostic_msgs::msg::DiagnosticStatus::WARN,
                                "Overlap below threshold; watching", ov_str);
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
    // Save the map if enabled and close the debug logs; see pointlio_core.hpp.
    shutdown_common();

    // Join the search thread: a still-joinable std::thread destroyed at scope
    // exit terminates the process. Clearing keep_searching returns it in one tick.
    {
        std::lock_guard<std::mutex> lk(init_state_mutex);
        keep_searching = false;
    }
    if (init_thread.joinable()) init_thread.join();

    return 0;
}
