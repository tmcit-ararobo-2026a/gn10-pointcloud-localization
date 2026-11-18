// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <mutex>
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
    int top_k_candidates{4};
    float uniqueness_ratio{0.95f};
};

class GlobalSearcher
{
public:
    GlobalSearcher(const GlobalSearchConfig& config, rclcpp::Logger logger);
    ~GlobalSearcher() = default;

    void setConfig(const GlobalSearchConfig& config)
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        config_ = config;
    }

    GlobalSearchConfig getConfig() const
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        return config_;
    }

    /**
     * @brief Returns wall-clock duration [ms] of the most recent search() call.
     */
    double lastSearchTimeMs() const
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        return last_search_time_ms_;
    }

    PoseCandidate search(
        PoseSolver& solver,
        const std::vector<float>& h_raw_cloud,
        const float* h_transform,
        const GroundFilterParams& filter_params,
        const MatchingParams& match_params,
        float prior_yaw,
        float max_yaw_diff,
        float& out_best_cost,
        int* out_inlier_count                            = nullptr,
        float* out_inlier_cost                           = nullptr,
        const std::vector<float>* extra_base_link_points = nullptr
    );

    PoseCandidate search(
        PoseSolver& solver,
        const std::vector<TimedScanPoint>& timed_points,
        const float* h_transform,
        const DeskewParams& deskew_params,
        const std::vector<IMUSample>& imu_samples,
        int stride,
        const GroundFilterParams& filter_params,
        const MatchingParams& match_params,
        float prior_yaw,
        float max_yaw_diff,
        float& out_best_cost,
        int* out_inlier_count                            = nullptr,
        float* out_inlier_cost                           = nullptr,
        const std::vector<float>* extra_base_link_points = nullptr
    );

private:
    PoseCandidate runCoarseAndRefineSearch(
        PoseSolver& solver,
        const GlobalSearchConfig& config,
        int obstacle_count,
        const MatchingParams& match_params,
        float prior_yaw,
        float max_yaw_diff,
        float& out_best_cost,
        int* out_inlier_count,
        float* out_inlier_cost,
        std::chrono::steady_clock::time_point search_start_tp,
        double prep_time_ms
    );

    mutable std::mutex config_mutex_;
    GlobalSearchConfig config_;
    double last_search_time_ms_{0.0};
    rclcpp::Logger logger_;
};