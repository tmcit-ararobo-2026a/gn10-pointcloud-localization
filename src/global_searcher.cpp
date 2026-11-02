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
    float* out_inlier_cost
)
{
    RCLCPP_INFO(logger_, "[GlobalSearch] Starting GPU-Batched Yaw-Constrained Search...");

    GlobalSearchConfig config = getConfig();

    const size_t total_points = h_raw_cloud.size() / 3;
    int stride = config.downsample_stride;
    if (stride < 1) {
        stride = 1;
    }

    // 点群のダウンサンプリング
    std::vector<float> search_cloud;
    search_cloud.reserve((total_points / stride) * 3);

    for (size_t point_index = 0; point_index < total_points; point_index += stride) {
        search_cloud.push_back(h_raw_cloud[point_index * 3 + 0]);
        search_cloud.push_back(h_raw_cloud[point_index * 3 + 1]);
        search_cloud.push_back(h_raw_cloud[point_index * 3 + 2]);
    }

    // GPU 上に障害物点群を展開
    int obstacle_count = solver.prepareObstacleCloud(search_cloud, h_transform, filter_params);

    // 点数ログの出力
    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Points Summary - Raw: %zu, Downsampled (stride=%d): %zu, Obstacles: %d",
        total_points,
        stride,
        search_cloud.size() / 3,
        obstacle_count
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
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // IMUの事前Yaw角を中心に、指定された許容範囲(max_yaw_diff)のみを探索窓に設定
    PoseCandidate center_pose;
    center_pose.x   = (config.range_min_x + config.range_max_x) * 0.5f;
    center_pose.y   = (config.range_min_y + config.range_max_y) * 0.5f;
    center_pose.yaw = prior_yaw;

    float range_x   = (config.range_max_x - config.range_min_x) * 0.5f;
    float range_y   = (config.range_max_y - config.range_min_y) * 0.5f;
    float range_yaw = max_yaw_diff;

    PoseCandidate best_coarse_pose;
    float coarse_cost = std::numeric_limits<float>::max();

    // GPU 探索実行
    bool ok = solver.evaluateGlobalSDF(
        obstacle_count,
        center_pose,
        range_x,
        range_y,
        config.step_xy,
        range_yaw,
        config.step_yaw,
        match_params,
        best_coarse_pose,
        coarse_cost
    );

    if (!ok) {
        RCLCPP_ERROR(logger_, "[GlobalSearch] GPU Global Search Failed.");
        out_best_cost = std::numeric_limits<float>::max();
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // Raw 点群による局所リファイン処理
    std::vector<float> dummy_dynamic;
    PoseCandidate refined_pose;
    solver.processPointCloud(
        h_raw_cloud,
        h_transform,
        filter_params,
        match_params,
        best_coarse_pose,
        dummy_dynamic,
        refined_pose,
        out_best_cost,
        out_inlier_count,
        out_inlier_cost
    );

    // 範囲外チェック
    if (refined_pose.x < config.range_min_x || refined_pose.x > config.range_max_x ||
        refined_pose.y < config.range_min_y || refined_pose.y > config.range_max_y) {
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] Refined pose (%.2f, %.2f) is outside configured area [%.2f~%.2f, %.2f~%.2f]. Rejecting pose.",
            refined_pose.x,
            refined_pose.y,
            config.range_min_x,
            config.range_max_x,
            config.range_min_y,
            config.range_max_y
        );
        out_best_cost = std::numeric_limits<float>::max();
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    refined_pose.yaw = normalizeAngle(refined_pose.yaw);

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Complete. Best Pose: (%.2f, %.2f, %.2f rad), Cost: %.4f",
        refined_pose.x,
        refined_pose.y,
        refined_pose.yaw,
        out_best_cost
    );

    return refined_pose;
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
    float* out_inlier_cost
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

    // GPU 上に歪み補正済み障害物点群を直接展開
    int obstacle_count = solver.prepareObstacleCloud(
        timed_points, h_transform, deskew_params, imu_samples, combined_stride, filter_params
    );

    // 点数ログの出力
    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Points Summary - Raw: %zu, Downsampled (stride=%d): %zu, Obstacles: %d",
        total_points,
        combined_stride,
        total_points / combined_stride,
        obstacle_count
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
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // IMUの事前Yaw角を中心に、指定された許容範囲(max_yaw_diff)のみを探索窓に設定
    PoseCandidate center_pose;
    center_pose.x   = (config.range_min_x + config.range_max_x) * 0.5f;
    center_pose.y   = (config.range_min_y + config.range_max_y) * 0.5f;
    center_pose.yaw = prior_yaw;

    float range_x   = (config.range_max_x - config.range_min_x) * 0.5f;
    float range_y   = (config.range_max_y - config.range_min_y) * 0.5f;
    float range_yaw = max_yaw_diff;

    PoseCandidate best_coarse_pose;
    float coarse_cost = std::numeric_limits<float>::max();

    // GPU 探索実行
    bool ok = solver.evaluateGlobalSDF(
        obstacle_count,
        center_pose,
        range_x,
        range_y,
        config.step_xy,
        range_yaw,
        config.step_yaw,
        match_params,
        best_coarse_pose,
        coarse_cost
    );

    if (!ok) {
        RCLCPP_ERROR(logger_, "[GlobalSearch] GPU Global Search Failed.");
        out_best_cost = std::numeric_limits<float>::max();
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // 局所リファイン処理 (GPU Deskew版)
    std::vector<float> dummy_dynamic;
    PoseCandidate refined_pose;
    solver.processPointCloud(
        timed_points,
        h_transform,
        deskew_params,
        imu_samples,
        base_stride,
        filter_params,
        match_params,
        best_coarse_pose,
        dummy_dynamic,
        refined_pose,
        out_best_cost,
        out_inlier_count,
        out_inlier_cost
    );

    // 範囲外チェック
    if (refined_pose.x < config.range_min_x || refined_pose.x > config.range_max_x ||
        refined_pose.y < config.range_min_y || refined_pose.y > config.range_max_y) {
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] Refined pose (%.2f, %.2f) is outside configured area [%.2f~%.2f, %.2f~%.2f]. Rejecting pose.",
            refined_pose.x,
            refined_pose.y,
            config.range_min_x,
            config.range_max_x,
            config.range_min_y,
            config.range_max_y
        );
        out_best_cost = std::numeric_limits<float>::max();
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    refined_pose.yaw = normalizeAngle(refined_pose.yaw);

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Complete. Best Pose: (%.2f, %.2f, %.2f rad), Cost: %.4f",
        refined_pose.x,
        refined_pose.y,
        refined_pose.yaw,
        out_best_cost
    );

    return refined_pose;
}
