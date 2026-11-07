// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include "gn10_pointcloud_localization/global_searcher.hpp"

#include <cmath>
#include <limits>

GlobalSearcher::GlobalSearcher(const GlobalSearchConfig& config, rclcpp::Logger logger)
    : config_(config), logger_(logger)
{
}

PoseCandidate GlobalSearcher::search(
    PoseSolver& solver,
    const std::vector<float>& h_raw_cloud,
    const float* h_transform,
    const GroundFilterParams& filter_params,
    const MatchingParams& match_params,
    float prior_yaw,
    float max_yaw_diff,
    float& out_best_cost,
    int* out_inlier_count,
    float* out_inlier_cost,
    const std::vector<float>* extra_base_link_points
)
{
    RCLCPP_INFO(logger_, "[GlobalSearch] Starting GPU-Batched Yaw-Constrained Search...");

    GlobalSearchConfig config = getConfig();

    const size_t total_points = h_raw_cloud.size() / 3;
    int stride = config.downsample_stride;
    if (stride < 1) {
        stride = 1;
    }

    // Downsample raw 3D point cloud for coarse global grid search
    std::vector<float> search_cloud;
    search_cloud.reserve((total_points / stride) * 3);

    for (size_t point_index = 0; point_index < total_points; point_index += stride) {
        search_cloud.push_back(h_raw_cloud[point_index * 3 + 0]);
        search_cloud.push_back(h_raw_cloud[point_index * 3 + 1]);
        search_cloud.push_back(h_raw_cloud[point_index * 3 + 2]);
    }

    // Prepare 3D + 2D obstacle cloud on GPU for coarse evaluation
    int obstacle_count = solver.prepareObstacleCloud(
        search_cloud, h_transform, filter_params, extra_base_link_points
    );

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Points Summary - Raw: %zu, Downsampled (stride=%d): %zu, Obstacles: %d (3D: %d, 2D: %d)",
        total_points,
        stride,
        search_cloud.size() / 3,
        obstacle_count,
        solver.obstacleCount3D(),
        solver.obstacleCount2D()
    );

    if (obstacle_count <= match_params.min_3d_points) {
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] Too few obstacle points (%d <= %d). Search aborted. "
            "Consider reducing 'filter_params.robot_radius' or 'global_search.downsample_stride', "
            "or check 'ground_z_thresh' / TF alignment.",
            obstacle_count,
            match_params.min_3d_points
        );
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // Center global search window around field midpoint and IMU prior yaw
    PoseCandidate center_pose;
    center_pose.x   = (config.range_min_x + config.range_max_x) * 0.5f;
    center_pose.y   = (config.range_min_y + config.range_max_y) * 0.5f;
    center_pose.yaw = prior_yaw;

    float range_x   = (config.range_max_x - config.range_min_x) * 0.5f;
    float range_y   = (config.range_max_y - config.range_min_y) * 0.5f;
    float range_yaw = max_yaw_diff;

    std::vector<PoseCandidate> top_coarse_poses;
    const int top_k = std::max(1, config.top_k_candidates);

    // Execute coarse global search and extract Top-K NMS candidates
    bool ok = solver.evaluateGlobalSDF(
        obstacle_count,
        center_pose,
        range_x,
        range_y,
        config.step_xy,
        range_yaw,
        config.step_yaw,
        match_params,
        top_coarse_poses,
        top_k
    );

    if (!ok || top_coarse_poses.empty()) {
        RCLCPP_ERROR(logger_, "[GlobalSearch] GPU Global Search Failed.");
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // Prepare full-resolution obstacle cloud once on GPU for refining all Top-K candidates
    int full_obstacle_count = solver.prepareObstacleCloud(
        h_raw_cloud, h_transform, filter_params, extra_base_link_points
    );

    std::vector<float> dummy_dynamic;
    PoseCandidate best_refined_pose{0.0f, 0.0f, 0.0f};
    float best_refined_cost        = std::numeric_limits<float>::max();
    float best_cost_3d             = std::numeric_limits<float>::max();
    float best_cost_2d             = std::numeric_limits<float>::max();
    int best_refined_inliers       = 0;
    float best_refined_inlier_cost = 1.0f;
    bool found_accepted            = false;

    for (const auto& coarse_candidate : top_coarse_poses) {
        PoseCandidate cand_pose{};
        float cand_cost       = std::numeric_limits<float>::max();
        int cand_inliers      = 0;
        float cand_inlier_cost = 1.0f;

        bool matched = solver.matchObstacleCloud(
            full_obstacle_count,
            match_params,
            coarse_candidate,
            dummy_dynamic,
            cand_pose,
            cand_cost,
            &cand_inliers,
            &cand_inlier_cost,
            false
        );
        if (!matched) {
            continue;
        }

        // Reject candidate if refined pose falls outside configured field bounds
        if (cand_pose.x < config.range_min_x || cand_pose.x > config.range_max_x ||
            cand_pose.y < config.range_min_y || cand_pose.y > config.range_max_y) {
            continue;
        }

        cand_pose.yaw = normalizeAngle(cand_pose.yaw);

        const bool is_accepted = match_params.isMatchAccepted(
            solver.lastCost3D(),
            solver.lastCost2D(),
            solver.obstacleCount3D(),
            solver.obstacleCount2D(),
            cand_inliers,
            cand_inlier_cost
        );
        if (is_accepted && cand_cost < best_refined_cost) {
            found_accepted           = true;
            best_refined_pose        = cand_pose;
            best_refined_cost        = cand_cost;
            best_refined_inliers     = cand_inliers;
            best_refined_inlier_cost = cand_inlier_cost;
            best_cost_3d             = solver.lastCost3D();
            best_cost_2d             = solver.lastCost2D();
        }
    }

    if (!found_accepted) {
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] None of the Top-%zu candidates satisfied strict acceptance criteria.",
            top_coarse_poses.size()
        );
        solver.setLastCosts(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    solver.setLastCosts(best_cost_3d, best_cost_2d);

    out_best_cost = best_refined_cost;
    if (out_inlier_count != nullptr) {
        *out_inlier_count = best_refined_inliers;
    }
    if (out_inlier_cost != nullptr) {
        *out_inlier_cost = best_refined_inlier_cost;
    }

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Complete (refined %zu candidates). Best Pose: (%.2f, %.2f, %.2f rad), Cost: %.4f, Inliers: %d",
        top_coarse_poses.size(),
        best_refined_pose.x,
        best_refined_pose.y,
        best_refined_pose.yaw,
        out_best_cost,
        best_refined_inliers
    );

    return best_refined_pose;
}

PoseCandidate GlobalSearcher::search(
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
    int* out_inlier_count,
    float* out_inlier_cost,
    const std::vector<float>* extra_base_link_points
)
{
    RCLCPP_INFO(logger_, "[GlobalSearch] Starting GPU-Batched Yaw-Constrained Search (with GPU Deskew)...");

    GlobalSearchConfig config = getConfig();

    const size_t total_points = timed_points.size();
    int base_stride = stride;
    if (base_stride < 1) {
        base_stride = 1;
    }
    int downsample_factor = config.downsample_stride;
    if (downsample_factor < 1) {
        downsample_factor = 1;
    }
    int combined_stride = base_stride * downsample_factor;

    // Prepare deskewed 3D + 2D obstacle cloud on GPU for coarse evaluation
    int obstacle_count = solver.prepareObstacleCloud(
        timed_points,
        h_transform,
        deskew_params,
        imu_samples,
        combined_stride,
        filter_params,
        extra_base_link_points
    );

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Points Summary - Raw: %zu, Downsampled (stride=%d): %zu, Obstacles: %d (3D: %d, 2D: %d)",
        total_points,
        combined_stride,
        total_points / combined_stride,
        obstacle_count,
        solver.obstacleCount3D(),
        solver.obstacleCount2D()
    );

    if (obstacle_count <= match_params.min_3d_points) {
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] Too few obstacle points (%d <= %d). Search aborted. "
            "Consider reducing 'filter_params.robot_radius' or 'global_search.downsample_stride', "
            "or check 'ground_z_thresh' / TF alignment.",
            obstacle_count,
            match_params.min_3d_points
        );
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // Center global search window around field midpoint and IMU prior yaw
    PoseCandidate center_pose;
    center_pose.x   = (config.range_min_x + config.range_max_x) * 0.5f;
    center_pose.y   = (config.range_min_y + config.range_max_y) * 0.5f;
    center_pose.yaw = prior_yaw;

    float range_x   = (config.range_max_x - config.range_min_x) * 0.5f;
    float range_y   = (config.range_max_y - config.range_min_y) * 0.5f;
    float range_yaw = max_yaw_diff;

    std::vector<PoseCandidate> top_coarse_poses;
    const int top_k = std::max(1, config.top_k_candidates);

    // Execute coarse global search and extract Top-K NMS candidates
    bool ok = solver.evaluateGlobalSDF(
        obstacle_count,
        center_pose,
        range_x,
        range_y,
        config.step_xy,
        range_yaw,
        config.step_yaw,
        match_params,
        top_coarse_poses,
        top_k
    );

    if (!ok || top_coarse_poses.empty()) {
        RCLCPP_ERROR(logger_, "[GlobalSearch] GPU Global Search Failed.");
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // Prepare full-resolution deskewed 3D + 2D obstacle cloud once on GPU for local refinement
    int full_obstacle_count = solver.prepareObstacleCloud(
        timed_points,
        h_transform,
        deskew_params,
        imu_samples,
        base_stride,
        filter_params,
        extra_base_link_points
    );

    std::vector<float> dummy_dynamic;
    PoseCandidate best_refined_pose{0.0f, 0.0f, 0.0f};
    float best_refined_cost        = std::numeric_limits<float>::max();
    float best_cost_3d             = std::numeric_limits<float>::max();
    float best_cost_2d             = std::numeric_limits<float>::max();
    int best_refined_inliers       = 0;
    float best_refined_inlier_cost = 1.0f;
    bool found_accepted            = false;

    for (const auto& coarse_candidate : top_coarse_poses) {
        PoseCandidate cand_pose{};
        float cand_cost        = std::numeric_limits<float>::max();
        int cand_inliers       = 0;
        float cand_inlier_cost = 1.0f;

        bool matched = solver.matchObstacleCloud(
            full_obstacle_count,
            match_params,
            coarse_candidate,
            dummy_dynamic,
            cand_pose,
            cand_cost,
            &cand_inliers,
            &cand_inlier_cost,
            false
        );
        if (!matched) {
            continue;
        }

        // Reject candidate if refined pose falls outside configured field bounds
        if (cand_pose.x < config.range_min_x || cand_pose.x > config.range_max_x ||
            cand_pose.y < config.range_min_y || cand_pose.y > config.range_max_y) {
            continue;
        }

        cand_pose.yaw = normalizeAngle(cand_pose.yaw);

        const bool is_accepted = match_params.isMatchAccepted(
            solver.lastCost3D(),
            solver.lastCost2D(),
            solver.obstacleCount3D(),
            solver.obstacleCount2D(),
            cand_inliers,
            cand_inlier_cost
        );
        if (is_accepted && cand_cost < best_refined_cost) {
            found_accepted           = true;
            best_refined_pose        = cand_pose;
            best_refined_cost        = cand_cost;
            best_refined_inliers     = cand_inliers;
            best_refined_inlier_cost = cand_inlier_cost;
            best_cost_3d             = solver.lastCost3D();
            best_cost_2d             = solver.lastCost2D();
        }
    }

    if (!found_accepted) {
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] None of the Top-%zu candidates satisfied strict acceptance criteria.",
            top_coarse_poses.size()
        );
        solver.setLastCosts(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        out_best_cost = std::numeric_limits<float>::max();
        if (out_inlier_count != nullptr) {
            *out_inlier_count = 0;
        }
        if (out_inlier_cost != nullptr) {
            *out_inlier_cost = 1.0f;
        }
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    solver.setLastCosts(best_cost_3d, best_cost_2d);

    out_best_cost = best_refined_cost;
    if (out_inlier_count != nullptr) {
        *out_inlier_count = best_refined_inliers;
    }
    if (out_inlier_cost != nullptr) {
        *out_inlier_cost = best_refined_inlier_cost;
    }

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Complete (refined %zu candidates). Best Pose: (%.2f, %.2f, %.2f rad), Cost: %.4f, Inliers: %d",
        top_coarse_poses.size(),
        best_refined_pose.x,
        best_refined_pose.y,
        best_refined_pose.yaw,
        out_best_cost,
        best_refined_inliers
    );

    return best_refined_pose;
}
