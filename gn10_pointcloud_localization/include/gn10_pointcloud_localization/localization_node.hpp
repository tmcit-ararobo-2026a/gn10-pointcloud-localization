#pragma once
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <cuda_runtime.h>
#include <tf2_ros/transform_broadcaster.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

class LocalizationNode : public rclcpp::Node {
public:
    LocalizationNode();
    ~LocalizationNode();

private:
    void cloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void publishCloud(const rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr& pub, const std_msgs::msg::Header& header, const float* data, int count);

    // ROS 2 関連
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_cloud_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_ground_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_obstacle_;
    rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr pub_platform_pose_;

    // CUDA メモリバッファ
    int max_points_;
    float *d_in_, *d_ground_, *d_obstacle_, *d_transform_;
    int *d_ground_count_, *d_obstacle_count_;
    float *h_in_, *h_out_ground_, *h_out_obstacle_;

    // 自己位置推測の状態保持
    PoseCandidate last_known_pose_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    void publishFieldMapMarkers();

    // 追加するパブリッシャー & タイマー
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_map_markers_;
    rclcpp::TimerBase::SharedPtr marker_timer_;

    bool is_initialized_{false};

    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
    
    PoseCandidate predicted_pose_;
    rclcpp::Time last_imu_stamp_;
    bool imu_initialized_{false};
    std::mutex pose_mutex_; // 点群コールバックとIMUコールバックの同期用
};