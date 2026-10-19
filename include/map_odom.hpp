// map_odom.hpp -- REP-105 map -> odom publishing for the localizers.
//
// Shared verbatim by fast_lio and point_lio's laserLocalization.cpp; keep the
// two copies identical.
//
// By default the localizers broadcast map -> base_footprint straight out of
// the filter (no odom frame). With publish.odom_frame set they broadcast
// map -> odom instead, composed so that
//
//     map -> odom -> base_footprint  ==  the filter's map -> base_footprint
//
// at the scan stamp, with odom -> base_footprint coming from a continuous
// source (Pepper's wheel odometry via pepper_slam's wheel_odom_tf.py). The
// point is containment: a handover, an /initialpose seed or a re-arm moves
// map -> odom, while the local costmap and controller, which read odom, never
// see the step. And if the localizer stalls, odom -> base_footprint keeps
// moving, so the robot still has a live pose to stop on.
//
// The Gate is the localization-side equivalent of lio_odom_guard: map -> odom
// should change by centimetres per scan (wheel drift over ~100 ms), so a large
// step outside a deliberate handover is held rather than passed on.

#pragma once

#include <cmath>
#include <cstdio>
#include <string>

#include <Eigen/Geometry>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/exceptions.h>
#include <tf2/time.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>

namespace map_odom
{

struct Params
{
    std::string odom_frame;        // "" => legacy map -> base_footprint
    bool   planar = true;          // compose in x, y, yaw only (wheel odom is planar)
    // MEASURED: 0.05 left "held" warnings on slam_20260823_aligned; 0.1 passed
    // (TEST_PLAN_map_odom.md tests 3-4). Longer than the ~90 ms scan period, so
    // a full-length wait costs a scan -- see odom_misses in the gate status.
    double lookup_timeout = 0.1;   // s to wait for odom -> base at the scan stamp
    double future_dating = 0.2;    // s added to the stamp, as AMCL's transform_tolerance
    double max_step_lin = 0.30;    // m per scan before a step is held
    double max_step_ang = 0.175;   // rad (~10 deg) per scan before a step is held
    double max_hold = 2.0;         // s of held steps before the new value is adopted
    // s the gate status stays ERROR after a forced adopt. Must outlast the
    // watchdog's lost_duration or a one-shot step never trips it; 0 = no ERROR.
    double error_latch = 6.0;
};

inline Eigen::Isometry3d to_iso(const geometry_msgs::msg::Transform &t)
{
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = Eigen::Quaterniond(t.rotation.w, t.rotation.x,
                                    t.rotation.y, t.rotation.z).normalized().toRotationMatrix();
    T.translation() = Eigen::Vector3d(t.translation.x, t.translation.y, t.translation.z);
    return T;
}

inline geometry_msgs::msg::Transform to_msg(const Eigen::Isometry3d &T)
{
    geometry_msgs::msg::Transform t;
    const Eigen::Quaterniond q(T.linear());
    t.translation.x = T.translation().x();
    t.translation.y = T.translation().y();
    t.translation.z = T.translation().z();
    t.rotation.w = q.w();
    t.rotation.x = q.x();
    t.rotation.y = q.y();
    t.rotation.z = q.z();
    return t;
}

// Keep x, y, z and yaw; drop roll/pitch. The filter's base_footprint carries
// ~3 deg of LIO attitude that planar wheel odometry cannot represent; left in,
// it would rotate map -> odom's tilt axis every time the robot turned.
inline Eigen::Isometry3d flatten(const Eigen::Isometry3d &T)
{
    const Eigen::Matrix3d R = T.linear();
    const double yaw = std::atan2(R(1, 0), R(0, 0));
    Eigen::Isometry3d F = Eigen::Isometry3d::Identity();
    F.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
    F.translation() = T.translation();
    return F;
}

class Gate
{
public:
    enum Verdict { ACCEPT, HOLD, ADOPT };

    // Call wherever the filter is DELIBERATELY moved (lock handover,
    // /initialpose seed): the next map -> odom is accepted whatever its step.
    void reset() { have_last_ = false; holding_ = false; }

    Verdict check(const Eigen::Isometry3d &T, double now_s, const Params &p)
    {
        if (!have_last_) {
            last_ = T;
            have_last_ = true;
            holding_ = false;
            step_lin_ = step_ang_ = 0.0;
            return ACCEPT;
        }
        const Eigen::Isometry3d d = last_.inverse() * T;
        step_lin_ = d.translation().norm();
        step_ang_ = Eigen::AngleAxisd(d.linear()).angle();
        if (step_lin_ <= p.max_step_lin && step_ang_ <= p.max_step_ang) {
            last_ = T;
            holding_ = false;
            return ACCEPT;
        }
        if (!holding_) {
            holding_ = true;
            hold_since_ = now_s;
        }
        if (now_s - hold_since_ < p.max_hold) return HOLD;
        // The filter has disagreed with the held edge for max_hold straight:
        // that is a persistent correction, not a glitch. Adopt it rather than
        // leave the robot dead-reckoning on wheels indefinitely; whether the
        // lock itself is right is the overlap health check's job.
        last_ = T;
        holding_ = false;
        last_adopt_s_ = now_s;
        return ADOPT;
    }

    const Eigen::Isometry3d &last() const { return last_; }
    double step_lin() const { return step_lin_; }
    double step_ang() const { return step_ang_; }
    // For the diagnostic status. last_adopt_s_ survives reset(): a deliberate
    // move does not un-happen an earlier forced adopt.
    bool holding() const { return holding_; }
    double last_adopt_s() const { return last_adopt_s_; }
    long misses() const { return misses_; }
    void count_miss() { ++misses_; }

private:
    Eigen::Isometry3d last_ = Eigen::Isometry3d::Identity();
    bool   have_last_ = false;
    bool   holding_ = false;
    double hold_since_ = 0.0;
    double step_lin_ = 0.0;
    double step_ang_ = 0.0;
    double last_adopt_s_ = -1e30;
    long   misses_ = 0;
};

// Broadcast map -> odom for a filter pose given as map -> child (the same
// edge the legacy path would have sent). Returns false, and sends nothing,
// when odom -> child is unavailable at the scan stamp: a stale TF makes Nav2
// stop, whereas a guessed edge would let it drive on a wrong pose.
inline bool publish(const geometry_msgs::msg::TransformStamped &map_child,
                    const Params &p, Gate &gate, tf2_ros::Buffer &buffer,
                    tf2_ros::TransformBroadcaster &br,
                    const rclcpp::Logger &logger, rclcpp::Clock &clock)
{
    geometry_msgs::msg::TransformStamped odom_child;
    try {
        // AT the scan stamp, never "latest": a newer odom sample would turn
        // the robot's motion since the scan into a false map -> odom step.
        odom_child = buffer.lookupTransform(
            p.odom_frame, map_child.child_frame_id,
            tf2_ros::fromMsg(map_child.header.stamp),
            tf2::durationFromSec(p.lookup_timeout));
    } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(logger, clock, 5000,
            "no %s -> %s at the scan stamp (%s); NOT broadcasting %s -> %s. "
            "Is wheel_odom_tf running?",
            p.odom_frame.c_str(), map_child.child_frame_id.c_str(), ex.what(),
            map_child.header.frame_id.c_str(), p.odom_frame.c_str());
        gate.count_miss();
        return false;
    }

    Eigen::Isometry3d T_mb = to_iso(map_child.transform);
    Eigen::Isometry3d T_ob = to_iso(odom_child.transform);
    if (p.planar) {
        T_mb = flatten(T_mb);
        T_ob = flatten(T_ob);
    }
    const Eigen::Isometry3d T_mo = T_mb * T_ob.inverse();

    const double now_s = rclcpp::Time(map_child.header.stamp).seconds();
    const Gate::Verdict v = gate.check(T_mo, now_s, p);
    if (v == Gate::HOLD) {
        RCLCPP_WARN_THROTTLE(logger, clock, 1000,
            "%s -> %s step of %.2f m / %.1f deg held (limit %.2f m / %.1f deg); "
            "keeping the previous correction",
            map_child.header.frame_id.c_str(), p.odom_frame.c_str(),
            gate.step_lin(), gate.step_ang() * 180.0 / M_PI,
            p.max_step_lin, p.max_step_ang * 180.0 / M_PI);
    } else if (v == Gate::ADOPT) {
        RCLCPP_ERROR(logger,
            "%s -> %s disagreed for %.1f s; adopting a %.2f m / %.1f deg step",
            map_child.header.frame_id.c_str(), p.odom_frame.c_str(), p.max_hold,
            gate.step_lin(), gate.step_ang() * 180.0 / M_PI);
    }

    // On HOLD the previous edge is re-sent with the new stamp, so TF stays
    // fresh and the robot rides on odom alone for the hold window.
    geometry_msgs::msg::TransformStamped out;
    out.header.frame_id = map_child.header.frame_id;
    out.child_frame_id = p.odom_frame;
    out.header.stamp = (rclcpp::Time(map_child.header.stamp) +
                        rclcpp::Duration::from_seconds(p.future_dating));
    out.transform = to_msg(gate.last());
    br.sendTransform(out);
    return true;
}

// The gate as a /diagnostics status, so a watchdog can act on it rather than
// the steps only reaching the log. ERROR is LATCHED for p.error_latch after a
// forced adopt: an adopt is a one-shot event, and a status that returns to OK
// on the next scan never outlasts a watchdog's sustained-badness window.
inline diagnostic_msgs::msg::DiagnosticStatus gate_status(
    const Gate &gate, double now_s, const Params &p,
    const std::string &name, const std::string &hardware_id)
{
    diagnostic_msgs::msg::DiagnosticStatus st;
    st.name = name;
    st.hardware_id = hardware_id;

    const double since_adopt = now_s - gate.last_adopt_s();
    char buf[256];
    if (p.error_latch > 0.0 && since_adopt <= p.error_latch) {
        st.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
        snprintf(buf, sizeof(buf),
                 "adopted a %.2f m / %.1f deg step %.1f s ago; the pose moved "
                 "under the planner", gate.step_lin(),
                 gate.step_ang() * 180.0 / M_PI, since_adopt);
    } else if (gate.holding()) {
        st.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
        snprintf(buf, sizeof(buf),
                 "holding a %.2f m / %.1f deg step; riding on wheel odometry",
                 gate.step_lin(), gate.step_ang() * 180.0 / M_PI);
    } else {
        st.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
        snprintf(buf, sizeof(buf), "ok; last step %.3f m / %.2f deg",
                 gate.step_lin(), gate.step_ang() * 180.0 / M_PI);
    }
    st.message = buf;

    const auto kv = [&st](const char *k, const std::string &v) {
        diagnostic_msgs::msg::KeyValue e;
        e.key = k;
        e.value = v;
        st.values.push_back(e);
    };
    kv("step_lin_m", std::to_string(gate.step_lin()));
    kv("step_ang_deg", std::to_string(gate.step_ang() * 180.0 / M_PI));
    // Lookup failures: a rising count means odom -> base is late or absent,
    // which is also what a too-short lookup_timeout looks like.
    kv("odom_misses", std::to_string(gate.misses()));
    return st;
}

}  // namespace map_odom
