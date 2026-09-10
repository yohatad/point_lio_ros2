#pragma once
// =============================================================================
//  localization_search.hpp -- finding where we are on a prior map.
//
//  Split out of laserLocalization.cpp, which was 1645 lines with main() alone
//  taking 1013 of them. Nothing here runs during normal tracking: this is the
//  bootstrap that tells the filter where on the map it started, plus the
//  health reporting that watches whether that answer still holds.
//
//    load_prior_map()             keyframe clouds + poses -> ScanContext DB
//    odom_at()                    odometry pose at an init-phase scan
//    map_overlap()                fraction of a scan landing on the prior map
//    global_localization_thread() the search: ScanContext -> ICP -> gates
//    rearm_search()               drop the current lock, search again
//    publish_diagnostic()         /diagnostics status
//
//  A header, not a .cpp, so this stays pure code motion: these functions read
//  the localization globals declared at the top of laserLocalization.cpp, and
//  a separate translation unit would mean externing all of them. Include it
//  after those globals and after pointlio_core.hpp.
// =============================================================================

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
