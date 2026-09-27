#include "gn10_pointcloud_localization/odom_projection.hpp"
#include "gn10_pointcloud_localization/pose_fusion_filter.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Geometry>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <diagnostic_msgs/msg/key_value.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

namespace {

double seconds(const builtin_interfaces::msg::Time& stamp)
{
    return static_cast<double>(stamp.sec) + static_cast<double>(stamp.nanosec) * 1e-9;
}

Eigen::Isometry3d poseToEigen(const geometry_msgs::msg::Pose& pose)
{
    Eigen::Quaterniond q(
        pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z
    );
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() = q.normalized().toRotationMatrix();
    result.translation() = Eigen::Vector3d(pose.position.x, pose.position.y, pose.position.z);
    return result;
}

class PoseFusionNode : public rclcpp::Node
{
public:
    PoseFusionNode() : Node("gn10_pose_fusion_node"), filter_(loadConfig())
    {
        history_s_ = get_parameter("fusion.history_s").as_double();
        max_odom_gap_s_ = get_parameter("fusion.max_odom_gap_s").as_double();
        max_odom_step_m_ = get_parameter("fusion.max_odom_step_m").as_double();
        max_odom_step_yaw_rad_ =
            get_parameter("fusion.max_odom_step_yaw_rad").as_double();
        map_frame_ = declare_parameter<std::string>("frames.map_frame", "map");
        base_frame_ = declare_parameter<std::string>("frames.base_frame", "base_link");
        lidar_frame_ = declare_parameter<std::string>("frames.lidar_frame", "livox_frame");
        expected_odom_frame_ =
            declare_parameter<std::string>("frames.odom_frame", "camera_init");
        expected_body_frame_ = declare_parameter<std::string>("frames.body_frame", "body");
        constrain_to_floor_ = declare_parameter<bool>("fusion.constrain_to_floor", true);
        map_only_enabled_ = declare_parameter<bool>("fusion.map_only_updates", false);
        max_odom_speed_ = declare_parameter("fusion.max_odom_speed_m_s", 5.0);
        max_odom_rotation_rate_ = declare_parameter("fusion.max_odom_rotation_rad_s", 3.0);
        max_floor_tilt_ = declare_parameter("fusion.max_floor_tilt_rad", 0.35);
        max_prediction_age_s_ = declare_parameter("fusion.max_prediction_age_s", 1.0);
        if (!std::isfinite(max_odom_speed_) || max_odom_speed_ <= 0 ||
            !std::isfinite(max_odom_rotation_rate_) || max_odom_rotation_rate_ <= 0 ||
            !std::isfinite(max_floor_tilt_) || max_floor_tilt_ <= 0 ||
            !std::isfinite(max_prediction_age_s_) || max_prediction_age_s_ <= 0)
            throw std::runtime_error("Odometry health limits must be finite and positive");
        const auto extrinsic = declare_parameter<std::vector<double>>(
            "imu_to_lidar.xyz", {-0.011, -0.02329, 0.04412}
        );
        const auto angles = declare_parameter<std::vector<double>>(
            "imu_to_lidar.rpy", {0.0, 0.0, 0.0}
        );
        if (extrinsic.size() != 3 || angles.size() != 3) {
            throw std::runtime_error("imu_to_lidar.xyz and .rpy must each contain 3 values");
        }
        lidar_to_imu_ = Eigen::Isometry3d::Identity();
        lidar_to_imu_.linear() = (Eigen::AngleAxisd(angles[2], Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(angles[1], Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(angles[0], Eigen::Vector3d::UnitX()))
                                     .toRotationMatrix();
        lidar_to_imu_.translation() = Eigen::Vector3d(extrinsic[0], extrinsic[1], extrinsic[2]);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
        broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
        const auto odom_topic = declare_parameter<std::string>("topics.odom", "/Odometry");
        const auto match_topic = declare_parameter<std::string>(
            "topics.match", "/platform_constraint_raw"
        );
        const auto output_topic = declare_parameter<std::string>(
            "topics.output", "/platform_constraint"
        );
        pub_motion_ = create_publisher<nav_msgs::msg::Odometry>(
            declare_parameter<std::string>("topics.motion_output", "/gn10/odom_base"), 50);
        publisher_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            output_topic, 20
        );
        prior_publisher_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            declare_parameter<std::string>("topics.matching_prior", "/gn10/matching_prior"), 50);
        odom_subscription_ = create_subscription<nav_msgs::msg::Odometry>(
            odom_topic, 20,
            [this](nav_msgs::msg::Odometry::ConstSharedPtr msg) { onOdometry(*msg); }
        );
        match_subscription_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            match_topic, 20,
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg) {
                onMatch(*msg);
            }
        );
        initial_pose_subscription_ = create_subscription<geometry_msgs::msg::PoseWithCovarianceStamped>(
            "/initialpose", 10,
            [this](geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr msg) {
                if (msg->header.frame_id != map_frame_ || !previous_odom_base_ ||
                    !odom_healthy_) return;
                const auto& p = msg->pose.pose;
                const auto& q = p.orientation;
                const double norm = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
                if (!std::isfinite(norm) || std::abs(norm-1.0) > 0.01) return;
                const gn10::Pose2d pose{p.position.x, p.position.y,
                    std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))};
                if (!filter_.setMapPose(pose)) return;
                pending_matches_.clear(); recovery_evidence_.clear();
                anchor_odom_base_ = *previous_odom_base_;
                anchor_map_yaw_ = pose.yaw;
                last_accepted_stamp_ = filter_.latestStamp();
                recovery_stamp_ = last_accepted_stamp_;
                publish(latest_motion_stamp_, *previous_odom_base_);
                RCLCPP_INFO(get_logger(), "Map pose explicitly initialized by /initialpose");
            });
        diagnostics_publisher_ = create_publisher<diagnostic_msgs::msg::DiagnosticArray>(
            "/gn10_pose_fusion/diagnostics", 10
        );
        using namespace std::chrono_literals;
        diagnostics_timer_ = create_wall_timer(1s, [this] { publishDiagnostics(); });
    }

private:
    gn10::FusionConfig loadConfig()
    {
        gn10::FusionConfig config;
        config.history_s = declare_parameter("fusion.history_s", config.history_s);
        config.max_odom_gap_s =
            declare_parameter("fusion.max_odom_gap_s", config.max_odom_gap_s);
        config.max_odom_step_m =
            declare_parameter("fusion.max_odom_step_m", config.max_odom_step_m);
        config.max_odom_step_yaw_rad = declare_parameter(
            "fusion.max_odom_step_yaw_rad", config.max_odom_step_yaw_rad
        );
        config.max_match_skew_s =
            declare_parameter("fusion.max_match_skew_s", config.max_match_skew_s);
        config.process_translation_per_m = declare_parameter(
            "fusion.process_translation_per_m", config.process_translation_per_m
        );
        config.process_translation_per_s = declare_parameter(
            "fusion.process_translation_per_s", config.process_translation_per_s
        );
        config.process_yaw_per_rad = declare_parameter(
            "fusion.process_yaw_per_rad", config.process_yaw_per_rad
        );
        config.process_yaw_per_s =
            declare_parameter("fusion.process_yaw_per_s", config.process_yaw_per_s);
        config.match_xy_stddev =
            declare_parameter("fusion.match_xy_stddev", config.match_xy_stddev);
        config.match_yaw_stddev =
            declare_parameter("fusion.match_yaw_stddev", config.match_yaw_stddev);
        config.max_match_translation_m = declare_parameter(
            "fusion.max_match_translation_m", config.max_match_translation_m);
        config.max_match_yaw_rad = declare_parameter(
            "fusion.max_match_yaw_rad", config.max_match_yaw_rad);
        config.max_recovery_translation_m = declare_parameter("fusion.max_recovery_translation_m", config.max_recovery_translation_m);
        config.max_recovery_yaw_rad = declare_parameter("fusion.max_recovery_yaw_rad", config.max_recovery_yaw_rad);
        config.recovery_step_m = declare_parameter("fusion.recovery_step_m", config.recovery_step_m);
        config.recovery_step_yaw_rad = declare_parameter("fusion.recovery_step_yaw_rad", config.recovery_step_yaw_rad);
        config.innovation_gate =
            declare_parameter("fusion.innovation_gate", config.innovation_gate);
        if (config.history_s <= 0 || config.max_odom_gap_s <= 0 ||
            config.max_odom_step_m <= 0 || config.max_odom_step_yaw_rad <= 0 ||
            config.max_match_skew_s <= 0 || config.match_xy_stddev <= 0 ||
            config.match_yaw_stddev <= 0 || config.max_match_translation_m <= 0 ||
            config.max_match_yaw_rad <= 0 || config.max_recovery_translation_m <= 0 ||
            config.max_recovery_yaw_rad <= 0 || config.recovery_step_m <= 0 ||
            config.recovery_step_yaw_rad <= 0 || config.innovation_gate <= 0) {
            throw std::runtime_error("Fusion time, measurement noise and gate must be positive");
        }
        return config;
    }

    bool initializeExtrinsic()
    {
        if (body_to_base_) return true;
        if (expected_body_frame_ == base_frame_) {
            body_to_base_ = Eigen::Isometry3d::Identity();
            return true;
        }
        try {
            const auto base_to_lidar = tf_buffer_->lookupTransform(
                base_frame_, lidar_frame_, tf2::TimePointZero
            );
            body_to_base_ = lidar_to_imu_ *
                tf2::transformToEigen(base_to_lidar).inverse();
            return true;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Waiting for base_link to Livox static TF: %s", ex.what()
            );
            return false;
        }
    }

    void resetMotion()
    {
        filter_.reset();map_only_active_=false;
        previous_odom_base_.reset();
        full_odom_history_.clear();
        anchor_odom_base_.reset();
        pending_matches_.clear(); recovery_evidence_.clear();
        virtual_odom_ = {};
        initial_floor_rotation_.reset();
        odom_healthy_ = true; healthy_streak_ = 0;
        recovery_stamp_ = -std::numeric_limits<double>::infinity();
        last_accepted_stamp_ = std::numeric_limits<double>::quiet_NaN();
    }

    void recordAcceptedMatch(double stamp, const gn10::Pose2d& measurement)
    {
        ++matches_accepted_;
        last_accepted_stamp_ = stamp;
        if (anchor_odom_base_) return;
        const auto nearest = std::min_element(
            full_odom_history_.begin(), full_odom_history_.end(),
            [stamp](const auto& a, const auto& b) {
                return std::abs(a.first - stamp) < std::abs(b.first - stamp);
            }
        );
        if (nearest != full_odom_history_.end()) {
            anchor_odom_base_ = nearest->second;
            anchor_map_yaw_ = measurement.yaw;
        }
    }

    void onOdometry(const nav_msgs::msg::Odometry& msg)
    {
        ++odom_received_;
        if (msg.header.frame_id != expected_odom_frame_ ||
            msg.child_frame_id != expected_body_frame_) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "FAST-LIO odometry frames do not match configured odom/body frames"
            );
            return;
        }
        if (!initializeExtrinsic()) return;
        const auto& q = msg.pose.pose.orientation;
        if (!std::isfinite(q.w) || !std::isfinite(q.x) ||
            !std::isfinite(q.y) || !std::isfinite(q.z) ||
            q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z < 0.5) return;

        const double stamp = seconds(msg.header.stamp);
        const Eigen::Isometry3d odom_base = poseToEigen(msg.pose.pose) * *body_to_base_;
        if (!std::isfinite(stamp) || !odom_base.translation().allFinite()) return;
        if (!initial_floor_rotation_) initial_floor_rotation_ = odom_base.linear();
        bool usable = true;
        if (previous_odom_base_) {
            const double dt = stamp - previous_odom_stamp_;
            if (dt <= 0.0 && dt >= -0.5) return;
            const Eigen::Isometry3d delta = previous_odom_base_->inverse() * odom_base;
            const double rotation = Eigen::AngleAxisd(delta.linear()).angle();
            if (dt < -0.5) {
                resetMotion(); // A new bag time epoch, not a loss of sensor packets.
                initial_floor_rotation_ = odom_base.linear();
            } else {
                const auto relative = initial_floor_rotation_->transpose() * odom_base.linear();
                const double tilt = std::acos(std::clamp(relative(2, 2), -1.0, 1.0));
                usable = dt > 0 && dt <= max_odom_gap_s_ &&
                    delta.translation().norm() <= max_odom_step_m_ &&
                    rotation <= max_odom_step_yaw_rad_ &&
                    delta.translation().norm()/dt <= max_odom_speed_ &&
                    rotation/dt <= max_odom_rotation_rate_ &&
                    (!constrain_to_floor_ || tilt <= max_floor_tilt_);
                if (usable && odom_healthy_) {
                    virtual_odom_ = gn10::integrateBodyMotion(
                        virtual_odom_, *previous_odom_base_, odom_base);
                }
            }
        }
        previous_odom_base_ = odom_base;
        previous_odom_stamp_ = stamp;
        if (!usable) {
            ++odom_rejected_; odom_healthy_ = false; healthy_streak_ = 0;
            recovery_stamp_ = std::numeric_limits<double>::infinity();
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "Odometry failed motion/floor limits; withholding map TF, retaining map anchor");
            return;
        }
        if (!odom_healthy_) {
            if (++healthy_streak_ < 3) return;
            // Start a new continuous segment without applying the rejected displacement.
            // Its gap remains visible to the deskewer; no fabricated scan motion.
            odom_healthy_ = true;
            recovery_stamp_ = stamp;
        }
        nav_msgs::msg::Odometry base_motion;
        base_motion.header = msg.header;
        base_motion.header.frame_id = "gn10_motion_odom"; // origin at the first valid base pose
        base_motion.child_frame_id = base_frame_;
        base_motion.pose.pose.position.x = virtual_odom_.x;
        base_motion.pose.pose.position.y = virtual_odom_.y;
        base_motion.pose.pose.orientation.z = std::sin(virtual_odom_.yaw * 0.5);
        base_motion.pose.pose.orientation.w = std::cos(virtual_odom_.yaw * 0.5);
        pub_motion_->publish(base_motion);
        if (map_only_active_) {
            if (stamp < filter_.latestStamp()) return;
            filter_.resynchronizeOdometry(stamp,virtual_odom_);
            map_only_active_=false; recovery_evidence_.clear();
        } else filter_.addOdometry(stamp, virtual_odom_);
        latest_motion_stamp_ = msg.header.stamp;
        previous_odom_base_ = odom_base;
        previous_odom_stamp_ = stamp;
        full_odom_history_.emplace_back(stamp, odom_base);
        while (full_odom_history_.size() > 1 &&
               stamp - full_odom_history_.front().first > history_s_ + 0.5) {
            full_odom_history_.pop_front();
        }

        for (auto it = pending_matches_.begin(); it != pending_matches_.end();) {
            if (it->first > stamp + 0.15) {
                ++it;
                continue;
            }
            if (!tryMapMatch(it->first, it->second)) {
                recordRejection();
            } else {
                recordAcceptedMatch(it->first, it->second);
            }
            it = pending_matches_.erase(it);
        }
        if (filter_.hasPose() && anchor_odom_base_) {
            const bool allow_map_output = last_accepted_stamp_ >= recovery_stamp_ &&
                stamp-last_accepted_stamp_ <= max_prediction_age_s_;
            publish(msg.header.stamp, odom_base, allow_map_output);
        }
    }

    void onMatch(const geometry_msgs::msg::PoseWithCovarianceStamped& msg)
    {
        if (msg.header.frame_id != map_frame_) return;
        ++raw_matches_received_;
        const auto& p = msg.pose.pose;
        const gn10::Pose2d measurement{
            p.position.x, p.position.y,
            gn10::wrapYaw(2.0 * std::atan2(p.orientation.z, p.orientation.w))
        };
        const double stamp = seconds(msg.header.stamp);
        if (map_only_enabled_ && constrain_to_floor_ && filter_.hasPose() && anchor_odom_base_ &&
            stamp > filter_.latestStamp() && (!odom_healthy_ || map_only_active_ ||
            stamp-filter_.latestStamp() > 1e-4)) {
            filter_.addMapPrediction(stamp);map_only_active_=true;
            if (tryMapMatch(stamp,measurement)) {
                recordAcceptedMatch(stamp,measurement);++map_only_matches_;
                publish(msg.header.stamp,*previous_odom_base_,true);
            } else {
                recordRejection();
                publish(msg.header.stamp,*previous_odom_base_,false);
            }
            return;
        }
        if (stamp > filter_.latestStamp() + 0.15) {
            pending_matches_.emplace_back(stamp, measurement);
            if (pending_matches_.size() > 100) {
                pending_matches_.pop_front();
                ++matches_rejected_;
            }
            return;
        }
        if (!tryMapMatch(stamp, measurement)) {
            recordRejection();
        } else {
            recordAcceptedMatch(stamp, measurement);
            if (previous_odom_base_ && filter_.hasPose() && anchor_odom_base_) {
                const bool allow_map_output = odom_healthy_ && last_accepted_stamp_ >= recovery_stamp_ &&
                    filter_.latestStamp()-last_accepted_stamp_ <= max_prediction_age_s_;
                publish(latest_motion_stamp_, full_odom_history_.back().second, allow_map_output);
            }
        }
    }

    bool tryMapMatch(double stamp, const gn10::Pose2d& measurement)
    {
        if (filter_.addMatch(stamp, measurement)) { recovery_evidence_.clear(); return true; }
        if ((!odom_healthy_ && !map_only_active_) || filter_.lastMatchRejection() != gn10::MatchRejection::Innovation) {
            recovery_evidence_.clear(); return false;
        }
        const auto predicted = filter_.predictionAt(stamp);
        if (!predicted) return false;
        const gn10::Pose2d error{measurement.x-predicted->x, measurement.y-predicted->y,
            gn10::wrapYaw(measurement.yaw-predicted->yaw)};
        if (std::hypot(error.x,error.y) > get_parameter("fusion.max_recovery_translation_m").as_double() ||
            std::abs(error.yaw) > get_parameter("fusion.max_recovery_yaw_rad").as_double()) {
            recovery_evidence_.clear(); return false;
        }
        if (!recovery_evidence_.empty()) {
            const auto& previous = recovery_evidence_.back();
            if (stamp <= previous.first) return false;
            if (stamp-previous.first > 1.0 ||
                std::hypot(error.x-previous.second.x,error.y-previous.second.y) >
                    2.0*std::sqrt(2.0)*get_parameter("fusion.match_xy_stddev").as_double() ||
                std::abs(gn10::wrapYaw(error.yaw-previous.second.yaw)) >
                    2.0*std::sqrt(2.0)*get_parameter("fusion.match_yaw_stddev").as_double())
                recovery_evidence_.clear();
        }
        recovery_evidence_.emplace_back(stamp,error);
        if (recovery_evidence_.size() > 3) recovery_evidence_.pop_front();
        if (recovery_evidence_.size() < 3 || stamp-recovery_evidence_.front().first < 0.19) return false;
        if (!filter_.addMatch(stamp, measurement, true)) return false;
        ++recovery_matches_; recovery_evidence_.clear();
        return true;
    }

    void recordRejection()
    {
        ++matches_rejected_;
        switch (filter_.lastMatchRejection()) {
            case gn10::MatchRejection::Innovation: ++rejected_gate_; break;
            case gn10::MatchRejection::Timestamp: ++rejected_timestamp_; break;
            case gn10::MatchRejection::Duplicate: ++rejected_duplicate_; break;
            case gn10::MatchRejection::NoOdometry: ++rejected_no_odometry_; break;
            default: break;
        }
        RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Map match rejected (gate=%zu, timestamp=%zu, duplicate=%zu, no_odom=%zu)",
            rejected_gate_, rejected_timestamp_, rejected_duplicate_, rejected_no_odometry_
        );
    }

    void publishDiagnostics()
    {
        diagnostic_msgs::msg::DiagnosticArray array;
        array.header.stamp = now();
        diagnostic_msgs::msg::DiagnosticStatus status;
        status.name = "GN10 pose fusion";
        status.hardware_id = "gn10_pose_fusion_node";
        const double current_stamp = seconds(array.header.stamp);
        const double age = std::isfinite(last_accepted_stamp_) ?
            std::max(0.0,filter_.latestStamp()-last_accepted_stamp_) : std::numeric_limits<double>::infinity();
        const double sensor_latency = current_stamp - filter_.latestStamp();
        if (map_only_active_ && filter_.hasPose()) {
            status.level=diagnostic_msgs::msg::DiagnosticStatus::WARN;
            status.message=age<=max_prediction_age_s_ ?
                "Map observations only; odometry prediction unavailable" :
                "Map-only mode; awaiting a fresh supported observation";
        } else if (!odom_healthy_) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::ERROR;
            status.message = "Odometry motion/floor limits exceeded; map TF withheld";
        } else if (last_accepted_stamp_ < recovery_stamp_) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            status.message = "Motion recovered; waiting for a new local map match";
        } else if (!filter_.hasPose()) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            status.message = "Waiting for FAST-LIO odometry and first accepted map match";
        } else if (!std::isfinite(age) || age < 0.0 || age > max_prediction_age_s_) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            status.message = "Map match is stale; map TF withheld beyond prediction limit";
        } else {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::OK;
            status.message = "Recent map match accepted";
        }
        const auto add = [&status](const std::string& key, const std::string& value) {
            diagnostic_msgs::msg::KeyValue item;
            item.key = key;
            item.value = value;
            status.values.push_back(std::move(item));
        };
        add("recovery_matches", std::to_string(recovery_matches_));
        add("recovery_evidence", std::to_string(recovery_evidence_.size()));
        add("odom_rejected", std::to_string(odom_rejected_));
        add("map_only_active",map_only_active_ ? "true" : "false");
        add("map_only_matches",std::to_string(map_only_matches_));
        add("odom_healthy", odom_healthy_ ? "true" : "false");
        add("sensor_latency_s", std::isfinite(sensor_latency) ? std::to_string(sensor_latency) : "never");
        add("odom_received", std::to_string(odom_received_));
        add("raw_matches_received", std::to_string(raw_matches_received_));
        add("matches_accepted", std::to_string(matches_accepted_));
        add("matches_rejected", std::to_string(matches_rejected_));
        add("rejected_gate", std::to_string(rejected_gate_));
        add("rejected_timestamp", std::to_string(rejected_timestamp_));
        add("rejected_duplicate", std::to_string(rejected_duplicate_));
        add("rejected_no_odometry", std::to_string(rejected_no_odometry_));
        add("matches_pending", std::to_string(pending_matches_.size()));
        add("last_accepted_age_s", std::isfinite(age) ? std::to_string(age) : "never");
        array.status.push_back(std::move(status));
        diagnostics_publisher_->publish(array);
    }

    void publish(
        const builtin_interfaces::msg::Time& stamp,
        const Eigen::Isometry3d& odom_base, bool allow_map_output = true
    )
    {
        const auto state = filter_.pose();
        const auto covariance = filter_.covariance();
        const Eigen::Isometry3d map_base = gn10::reconstructMapBase(
            state, *anchor_odom_base_, anchor_map_yaw_, odom_base, constrain_to_floor_
        );
        const Eigen::Quaterniond q(map_base.linear());
        geometry_msgs::msg::PoseWithCovarianceStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = map_frame_;
        pose.pose.pose.position.x = state.x;
        pose.pose.pose.position.y = state.y;
        pose.pose.pose.position.z = map_base.translation().z();
        pose.pose.pose.orientation.x = q.x();
        pose.pose.pose.orientation.y = q.y();
        pose.pose.pose.orientation.z = q.z();
        pose.pose.pose.orientation.w = q.w();
        constexpr int axes[3] = {0, 1, 5};
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                pose.pose.covariance[axes[row] * 6 + axes[col]] = covariance(row, col);
        pose.pose.covariance[14] = 0.1;  // z, roll and pitch have no map measurement.
        pose.pose.covariance[21] = 0.1;
        pose.pose.covariance[28] = 0.1;
        // Prediction is an internal search aid, not an accepted map measurement/TF.
        prior_publisher_->publish(pose);
        if (!allow_map_output) return;
        publisher_->publish(pose);

        geometry_msgs::msg::TransformStamped transform;
        transform.header = pose.header;
        transform.child_frame_id = base_frame_;
        transform.transform.translation.x = state.x;
        transform.transform.translation.y = state.y;
        transform.transform.translation.z = map_base.translation().z();
        transform.transform.rotation = pose.pose.pose.orientation;
        broadcaster_->sendTransform(transform);
    }

    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_motion_;
    gn10::PoseFusionFilter filter_;
    bool constrain_to_floor_{true};
    bool map_only_enabled_{false},map_only_active_{false};
    size_t map_only_matches_{0};
    std::string map_frame_, base_frame_, lidar_frame_, expected_odom_frame_, expected_body_frame_;
    Eigen::Isometry3d lidar_to_imu_;
    std::optional<Eigen::Isometry3d> body_to_base_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
        match_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr initial_pose_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr publisher_, prior_publisher_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
    rclcpp::TimerBase::SharedPtr diagnostics_timer_;
    std::deque<std::pair<double, gn10::Pose2d>> pending_matches_;
    std::deque<std::pair<double, Eigen::Isometry3d>> full_odom_history_;
    std::optional<Eigen::Isometry3d> previous_odom_base_;
    std::optional<Eigen::Isometry3d> anchor_odom_base_;
    gn10::Pose2d virtual_odom_;
    double previous_odom_stamp_{0.0};
    builtin_interfaces::msg::Time latest_motion_stamp_;
    std::deque<std::pair<double, gn10::Pose2d>> recovery_evidence_;
    size_t recovery_matches_{0};
    double anchor_map_yaw_{0.0};
    double recovery_stamp_{-std::numeric_limits<double>::infinity()};
    double history_s_{5.0};
    double max_odom_gap_s_{3.0};
    double max_prediction_age_s_{1.0};
    double max_odom_speed_{5.0}, max_odom_rotation_rate_{3.0}, max_floor_tilt_{0.35};
    std::optional<Eigen::Matrix3d> initial_floor_rotation_;
    bool odom_healthy_{true};
    size_t healthy_streak_{0}, odom_rejected_{0};
    double max_odom_step_m_{2.0};
    double max_odom_step_yaw_rad_{1.5};
    size_t odom_received_{0};
    size_t raw_matches_received_{0};
    size_t matches_accepted_{0};
    size_t matches_rejected_{0};
    size_t rejected_gate_{0};
    size_t rejected_timestamp_{0};
    size_t rejected_duplicate_{0};
    size_t rejected_no_odometry_{0};
    double last_accepted_stamp_{std::numeric_limits<double>::quiet_NaN()};
};
}  // namespace

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseFusionNode>());
    rclcpp::shutdown();
    return 0;
}
