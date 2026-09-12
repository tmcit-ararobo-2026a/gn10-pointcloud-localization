#pragma once

#include <cuda_runtime.h>
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

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

class LocalizationNode : public rclcpp::Node
{
public:
    LocalizationNode();
    ~LocalizationNode() override;

private:
    // --- Callbacks ---
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);

    // --- Helper Methods ---
    void publishCloud(
        const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub,
        const std_msgs::msg::Header& header,
        const float* data,
        int count
    );
    void publishFieldMapMarkers();

    // --- ROS 2 Interfaces (TF & Message Filters) ---
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub_cloud_filter_;
    std::shared_ptr<tf2_ros::MessageFilter<sensor_msgs::msg::PointCloud2>> tf_filter_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;

    // --- ROS 2 Publishers & Timers ---
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_platform_pose_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_markers_;
    rclcpp::TimerBase::SharedPtr map_timer_;

    // --- CUDA Buffers & Parameters ---
    cudaStream_t stream_;
    int max_points_{200000};

    // Device Pointers
    float* d_in_{nullptr};
    float* d_ground_{nullptr};
    float* d_obstacle_{nullptr};
    float* d_transform_{nullptr};
    int* d_ground_count_{nullptr};
    int* d_obstacle_count_{nullptr};

    // Host Pinned Memory Pointers
    float* h_in_{nullptr};
    float* h_out_ground_{nullptr};
    float* h_out_obstacle_{nullptr};

    // --- Localization State & Map Data ---
    std::mutex pose_mutex_;  // 点群コールバックとIMUコールバックの排他制御用
    PoseCandidate last_known_pose_;
    PoseCandidate predicted_pose_;

    rclcpp::Time last_imu_stamp_;
    bool imu_initialized_{false};
    bool is_initialized_{false};

    std::vector<FieldObject> map_objects_;
};