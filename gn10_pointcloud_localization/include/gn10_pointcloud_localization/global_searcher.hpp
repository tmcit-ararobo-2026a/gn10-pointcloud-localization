#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vector>

#include "gn10_pointcloud_localization/pose_solver.hpp"

struct GlobalSearchConfig {
    float range_min_x{-5.25f};
    float range_max_x{5.25f};
    float range_min_y{-5.70f};
    float range_max_y{5.70f};
    float step_xy{0.30f};
    float step_yaw{0.2618f};
    int downsample_stride{2};
};

class GlobalSearcher
{
public:
    GlobalSearcher(const GlobalSearchConfig& config, rclcpp::Logger logger);
    ~GlobalSearcher() = default;

    PoseCandidate search(
        PoseSolver& solver,
        const std::vector<float>& h_raw_cloud,
        const float* h_transform,
        const GroundFilterParams& filter_params,
        const MatchingParams& match_params,
        std::vector<float>& out_ground_pts,
        std::vector<float>& out_obstacle_pts,
        float& out_best_cost
    );

private:
    GlobalSearchConfig config_;
    rclcpp::Logger logger_;
};