// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <rclcpp/rclcpp.hpp>
#include "gn10_pointcloud_localization/localization_node.hpp"

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LocalizationNode>();

    // Use a multi-threaded executor so that telemetry/status publishing,
    // 2D LiDAR deskewing, and high-frequency 3D LiDAR/IMU callbacks 
    // run independently without starvation or mutual latency spikes.
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 4);
    executor.add_node(node);
    executor.spin();

    rclcpp::shutdown();
    return 0;
}