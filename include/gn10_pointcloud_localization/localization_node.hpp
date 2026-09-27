#pragma once

#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <memory>
#include <Eigen/Core>
#include <nav_msgs/msg/odometry.hpp>
#include <diagnostic_msgs/msg/diagnostic_array.hpp>
#include "gn10_pointcloud_localization/motion_history.hpp"
#include "gn10_pointcloud_localization/scan_points.hpp"
#include "gn10_pointcloud_localization/custom_scan.hpp"
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/header.hpp>
#include <vector>
#include <visualization_msgs/msg/marker_array.hpp>

#include "gn10_pointcloud_localization/global_searcher.hpp"
#include "gn10_pointcloud_localization/map_loader.hpp"
#include "gn10_pointcloud_localization/pose_solver.hpp"

class LocalizationNode : public rclcpp::Node
{
public:
    LocalizationNode();
    ~LocalizationNode() override = default;

private:
    // 初期化ヘルパー
    void declareAndGetParameters();
    void setupMapData();
    void setupROSInterfaces();

    // Callbacks
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void customCloudCallback(const livox_ros_driver2::msg::CustomMsg::SharedPtr msg);
    void enqueueScan(const std_msgs::msg::Header& header, gn10::ScanPoints scan);
    void processScan(const std_msgs::msg::Header& header, gn10::ScanPoints scan);
    void drainClouds();
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void fusedPriorCallback(
        const geometry_msgs::msg::PoseWithCovarianceStamped::SharedPtr msg
    );

    // パイプライン分離ヘルパー関数
    bool getTransformAsArray(
        const std::string& frame_id, const rclcpp::Time& stamp, float out_transform[12]
    );
    void updateLostState(bool matched, float best_cost, float threshold);
    void publishPoseAndTransform(const rclcpp::Time& stamp, const PoseCandidate& pose);

    void publishCloud(
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
        const std_msgs::msg::Header& header,
        const std::vector<float>& pts
    );
    void publishFieldMapMarkers();

    // Member Objects
    std::unique_ptr<PoseSolver> solver_;
    std::unique_ptr<GlobalSearcher> global_searcher_;
    std::vector<FieldObject> map_objects_;

    // Parameters
    std::string map_frame_;
    std::string base_frame_;
    GroundFilterParams filter_params_;
    MatchingParams match_params_;

    // Global Search Area & Step Parameters
    float global_range_min_x_{-5.25f};
    float global_range_max_x_{5.25f};
    float global_range_min_y_{-5.70f};
    float global_range_max_y_{5.70f};
    float global_range_yaw_diff_{0.785f};
    float global_step_xy_{0.30f};
    float global_step_yaw_{0.2618f};
    int global_downsample_stride_{2};
    int lost_threshold_count_{5};
    bool publish_tf_{true};
    bool use_fused_prior_{false};
    double prior_max_age_s_{0.25};
    rclcpp::Time prior_stamp_;
    PoseCandidate fused_prior_;
    bool prior_received_{false};
    uint64_t timing_drops_{0}, match_accepted_{0}, match_rejected_{0};
    rclcpp::Publisher<diagnostic_msgs::msg::DiagnosticArray>::SharedPtr pub_match_diagnostics_;
    double scan_wait_s_{2.0};
    size_t scan_queue_size_{30};
    bool use_motion_{false};
    gn10::MotionHistory motion_;
    std::deque<gn10::TimedPose> prior_history_;
    struct PendingCloud { std_msgs::msg::Header header; std::chrono::steady_clock::time_point received; gn10::ScanPoints scan; };
    std::deque<PendingCloud> pending_clouds_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_motion_;
    rclcpp::TimerBase::SharedPtr motion_timer_;
    std::string motion_frame_;
    double last_cloud_stamp_{-1.0};
    Eigen::Matrix3d imu_rotation_{Eigen::Matrix3d::Identity()};
    std::string imu_frame_;
    struct YawSample { double stamp, yaw; };
    std::deque<YawSample> imu_history_;
    double integrated_yaw_{0.0}, match_integrated_yaw_{0.0};

    // Pose State & Recovery State
    std::mutex pose_mutex_;
    PoseCandidate last_known_pose_;
    PoseCandidate predicted_pose_;
    rclcpp::Time last_imu_stamp_;
    bool imu_initialized_{false};
    bool is_lost_{true};
    int lost_frame_count_{0};

    // ROS 2 Interfaces
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::Subscription<livox_ros_driver2::msg::CustomMsg>::SharedPtr sub_custom_cloud_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_cloud_filter_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>> tf_filter_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
        sub_fused_prior_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_dynamic_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_platform_pose_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_markers_;
    rclcpp::TimerBase::SharedPtr map_timer_;
};
