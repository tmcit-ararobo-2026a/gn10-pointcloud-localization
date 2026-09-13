#pragma once

#include <message_filters/subscriber.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/message_filter.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>

#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <memory>
#include <mutex>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
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
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // パイプライン分離ヘルパー関数
    bool getTransformAsArray(
        const std::string& frame_id, const rclcpp::Time& stamp, float out_transform[12]
    );
    std::vector<float> extractPointsFromMsg(const sensor_msgs::msg::PointCloud2::SharedPtr& msg);
    void updateLostState(bool matched, float best_cost);
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

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_cloud_filter_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>> tf_filter_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_dynamic_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_platform_pose_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_markers_;
    rclcpp::TimerBase::SharedPtr map_timer_;
};