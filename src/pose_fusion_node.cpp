#include "gn10_pointcloud_localization/pose_fusion_filter.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
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

gn10::Pose2d planarPose(const Eigen::Isometry3d& transform)
{
    return {transform.translation().x(), transform.translation().y(),
            std::atan2(transform.linear()(1, 0), transform.linear()(0, 0))};
}

class PoseFusionNode : public rclcpp::Node
{
public:
    PoseFusionNode() : Node("gn10_pose_fusion_node"), filter_(loadConfig())
    {
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
        imu_to_lidar_ = Eigen::Isometry3d::Identity();
        imu_to_lidar_.linear() = (Eigen::AngleAxisd(angles[2], Eigen::Vector3d::UnitZ()) *
                                  Eigen::AngleAxisd(angles[1], Eigen::Vector3d::UnitY()) *
                                  Eigen::AngleAxisd(angles[0], Eigen::Vector3d::UnitX()))
                                     .toRotationMatrix();
        imu_to_lidar_.translation() = Eigen::Vector3d(extrinsic[0], extrinsic[1], extrinsic[2]);

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
            body_to_base_ = imu_to_lidar_ *
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

    void onOdometry(const nav_msgs::msg::Odometry& msg)
    {
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
        const gn10::Pose2d odom_base = planarPose(
            poseToEigen(msg.pose.pose) * *body_to_base_
        );
        const double previous_stamp = filter_.latestStamp();
        filter_.addOdometry(stamp, odom_base);
        if (filter_.latestStamp() <= previous_stamp && std::isfinite(previous_stamp)) return;

        for (auto it = pending_matches_.begin(); it != pending_matches_.end();) {
            if (it->first > stamp + 0.15) {
                ++it;
                continue;
            }
            if (!filter_.addMatch(it->first, it->second)) {
                RCLCPP_WARN_THROTTLE(
                    get_logger(), *get_clock(), 2000,
                    "Map match rejected by fusion gate or timestamp alignment"
                );
            }
            it = pending_matches_.erase(it);
        }
        if (filter_.hasPose()) publish(msg.header.stamp);
    }

    void onMatch(const geometry_msgs::msg::PoseWithCovarianceStamped& msg)
    {
        if (msg.header.frame_id != map_frame_) return;
        const auto& p = msg.pose.pose;
        const gn10::Pose2d measurement{
            p.position.x, p.position.y,
            gn10::wrapYaw(2.0 * std::atan2(p.orientation.z, p.orientation.w))
        };
        const double stamp = seconds(msg.header.stamp);
        if (stamp > filter_.latestStamp() + 0.15) {
            pending_matches_.emplace_back(stamp, measurement);
            if (pending_matches_.size() > 100) pending_matches_.pop_front();
            return;
        }
        if (!filter_.addMatch(stamp, measurement)) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 2000,
                "Map match rejected by fusion gate or timestamp alignment"
            );
        }
    }

    void publish(const builtin_interfaces::msg::Time& stamp)
    {
        const auto state = filter_.pose();
        const auto covariance = filter_.covariance();
        const double half = state.yaw * 0.5;
        geometry_msgs::msg::PoseWithCovarianceStamped pose;
        pose.header.stamp = stamp;
        pose.header.frame_id = map_frame_;
        pose.pose.pose.position.x = state.x;
        pose.pose.pose.position.y = state.y;
        pose.pose.pose.orientation.z = std::sin(half);
        pose.pose.pose.orientation.w = std::cos(half);
        constexpr int axes[3] = {0, 1, 5};
        for (int row = 0; row < 3; ++row)
            for (int col = 0; col < 3; ++col)
                pose.pose.covariance[axes[row] * 6 + axes[col]] = covariance(row, col);
        publisher_->publish(pose);

        geometry_msgs::msg::TransformStamped transform;
        transform.header = pose.header;
        transform.child_frame_id = base_frame_;
        transform.transform.translation.x = state.x;
        transform.transform.translation.y = state.y;
        transform.transform.rotation = pose.pose.pose.orientation;
        broadcaster_->sendTransform(transform);
    }

    gn10::PoseFusionFilter filter_;
    std::string map_frame_, base_frame_, lidar_frame_, expected_odom_frame_, expected_body_frame_;
    Eigen::Isometry3d imu_to_lidar_;
    std::optional<Eigen::Isometry3d> body_to_base_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> broadcaster_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_subscription_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
        match_subscription_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr publisher_;
    std::deque<std::pair<double, gn10::Pose2d>> pending_matches_;
};
}  // namespace

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PoseFusionNode>());
    rclcpp::shutdown();
    return 0;
}
