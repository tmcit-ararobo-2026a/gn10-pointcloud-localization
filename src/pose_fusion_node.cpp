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
        publisher_ = create_publisher<geometry_msgs::msg::PoseWithCovarianceStamped>(
            output_topic, 20
        );
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
        config.innovation_gate =
            declare_parameter("fusion.innovation_gate", config.innovation_gate);
        if (config.history_s <= 0 || config.max_odom_gap_s <= 0 ||
            config.max_odom_step_m <= 0 || config.max_odom_step_yaw_rad <= 0 ||
            config.max_match_skew_s <= 0 || config.match_xy_stddev <= 0 ||
            config.match_yaw_stddev <= 0 || config.innovation_gate <= 0) {
            throw std::runtime_error("Fusion time, measurement noise and gate must be positive");
        }
        return config;
    }

    bool initializeExtrinsic()
    {
        if (body_to_base_) return true;
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
        filter_.reset();
        previous_odom_base_.reset();
        full_odom_history_.clear();
        anchor_odom_base_.reset();
        pending_matches_.clear();
        virtual_odom_ = {};
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
        if (previous_odom_base_) {
            const double dt = stamp - previous_odom_stamp_;
            if (dt <= 0.0 && dt >= -0.5) return;
            const Eigen::Isometry3d delta = previous_odom_base_->inverse() * odom_base;
            const double rotation = Eigen::AngleAxisd(delta.linear()).angle();
            if (dt <= 0.0 || dt > max_odom_gap_s_ ||
                delta.translation().norm() > max_odom_step_m_ ||
                rotation > max_odom_step_yaw_rad_) {
                resetMotion();
            } else {
                virtual_odom_ = gn10::integrateBodyMotion(
                    virtual_odom_, *previous_odom_base_, odom_base
                );
            }
        }
        filter_.addOdometry(stamp, virtual_odom_);
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
            if (!filter_.addMatch(it->first, it->second)) {
                recordRejection();
            } else {
                recordAcceptedMatch(it->first, it->second);
            }
            it = pending_matches_.erase(it);
        }
        if (filter_.hasPose() && anchor_odom_base_) publish(msg.header.stamp, odom_base);
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
        if (stamp > filter_.latestStamp() + 0.15) {
            pending_matches_.emplace_back(stamp, measurement);
            if (pending_matches_.size() > 100) {
                pending_matches_.pop_front();
                ++matches_rejected_;
            }
            return;
        }
        if (!filter_.addMatch(stamp, measurement)) {
            recordRejection();
        } else {
            recordAcceptedMatch(stamp, measurement);
        }
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
        const double age = current_stamp - last_accepted_stamp_;
        if (!filter_.hasPose()) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            status.message = "Waiting for FAST-LIO odometry and first accepted map match";
        } else if (!std::isfinite(age) || age < 0.0 || age > 1.0) {
            status.level = diagnostic_msgs::msg::DiagnosticStatus::WARN;
            status.message = "FAST-LIO prediction only; map match is stale";
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
        const Eigen::Isometry3d& odom_base
    )
    {
        const auto state = filter_.pose();
        const auto covariance = filter_.covariance();
        const Eigen::Isometry3d map_base = gn10::reconstructMapBase(
            state, *anchor_odom_base_, anchor_map_yaw_, odom_base
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

    gn10::PoseFusionFilter filter_;
    std::string map_frame_, base_frame_, lidar_frame_, expected_odom_frame_, expected_body_frame_;
    Eigen::Isometry3d lidar_to_imu_;
    std::optional<Eigen::Isometry3d> body_to_base_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
        match_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr publisher_;
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr diagnostics_publisher_;
    rclcpp::TimerBase::SharedPtr diagnostics_timer_;
    std::deque<std::pair<double, gn10::Pose2d>> pending_matches_;
    std::deque<std::pair<double, Eigen::Isometry3d>> full_odom_history_;
    std::optional<Eigen::Isometry3d> previous_odom_base_;
    std::optional<Eigen::Isometry3d> anchor_odom_base_;
    gn10::Pose2d virtual_odom_;
    double previous_odom_stamp_{0.0};
    double anchor_map_yaw_{0.0};
    double history_s_{5.0};
    double max_odom_gap_s_{1.0};
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
