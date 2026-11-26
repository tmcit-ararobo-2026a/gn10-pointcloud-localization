// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include "gn10_pointcloud_localization/global_searcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

bool hasSufficientGlobalSearchPoints(const PoseSolver& solver, const MatchingParams& match_params)
{
    return (solver.obstacleCount3D() >= match_params.min_3d_points) ||
           (solver.obstacleCount2D() >= match_params.min_2d_points);
}

void setSearchFailureOutputs(
    float& out_best_cost, int* out_inlier_count, float* out_inlier_cost
)
{
    out_best_cost = std::numeric_limits<float>::max();
    if (out_inlier_count != nullptr) {
        *out_inlier_count = 0;
    }
    if (out_inlier_cost != nullptr) {
        *out_inlier_cost = 1.0f;
    }
}

}  // namespace

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
    const auto search_start_tp = std::chrono::steady_clock::now();
    RCLCPP_INFO(logger_, "[GlobalSearch] Starting GPU-Batched Yaw-Constrained Search...");

    GlobalSearchConfig config = getConfig();

    const size_t total_points = h_raw_cloud.size() / 3;

    // Prepare 3D + 2D obstacle cloud on GPU (voxel-equalized when voxel_size > 0)
    int obstacle_count = solver.prepareObstacleCloud(
        h_raw_cloud, h_transform, filter_params, extra_base_link_points
    );
    const double prep_time_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - search_start_tp
                                ).count();

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Points Summary - Raw: %zu, Obstacles: %d (3D: %d, 2D: %d)",
        total_points,
        obstacle_count,
        solver.obstacleCount3D(),
        solver.obstacleCount2D()
    );

    if (!hasSufficientGlobalSearchPoints(solver, match_params)) {
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - search_start_tp
                                  ).count();
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            last_search_time_ms_ = elapsed_ms;
        }
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] Too few obstacle points (3D: %d < %d, 2D: %d < %d, elapsed: %.2f ms). Search aborted. "
            "Consider reducing 'filter_params.robot_radius' or check 'ground_z_thresh' / TF alignment.",
            solver.obstacleCount3D(),
            match_params.min_3d_points,
            solver.obstacleCount2D(),
            match_params.min_2d_points,
            elapsed_ms
        );
        setSearchFailureOutputs(out_best_cost, out_inlier_count, out_inlier_cost);
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    return runCoarseAndRefineSearch(
        solver,
        config,
        obstacle_count,
        match_params,
        prior_yaw,
        max_yaw_diff,
        out_best_cost,
        out_inlier_count,
        out_inlier_cost,
        search_start_tp,
        prep_time_ms
    );
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
    const auto search_start_tp = std::chrono::steady_clock::now();
    RCLCPP_INFO(logger_, "[GlobalSearch] Starting GPU-Batched Yaw-Constrained Search (with GPU Deskew)...");

    GlobalSearchConfig config = getConfig();

    const size_t total_points = timed_points.size();
    const int base_stride     = std::max(1, stride);

    // Prepare deskewed 3D + 2D obstacle cloud on GPU
    int obstacle_count = solver.prepareObstacleCloud(
        timed_points,
        h_transform,
        deskew_params,
        imu_samples,
        base_stride,
        filter_params,
        extra_base_link_points
    );
    const double prep_time_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - search_start_tp
                                ).count();

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Points Summary - Raw: %zu, Obstacles: %d (3D: %d, 2D: %d)",
        total_points,
        obstacle_count,
        solver.obstacleCount3D(),
        solver.obstacleCount2D()
    );

    if (!hasSufficientGlobalSearchPoints(solver, match_params)) {
        const double elapsed_ms = std::chrono::duration<double, std::milli>(
                                      std::chrono::steady_clock::now() - search_start_tp
                                  ).count();
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            last_search_time_ms_ = elapsed_ms;
        }
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] Too few obstacle points (3D: %d < %d, 2D: %d < %d, elapsed: %.2f ms). Search aborted. "
            "Consider reducing 'filter_params.robot_radius' or check 'ground_z_thresh' / TF alignment.",
            solver.obstacleCount3D(),
            match_params.min_3d_points,
            solver.obstacleCount2D(),
            match_params.min_2d_points,
            elapsed_ms
        );
        setSearchFailureOutputs(out_best_cost, out_inlier_count, out_inlier_cost);
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    return runCoarseAndRefineSearch(
        solver,
        config,
        obstacle_count,
        match_params,
        prior_yaw,
        max_yaw_diff,
        out_best_cost,
        out_inlier_count,
        out_inlier_cost,
        search_start_tp,
        prep_time_ms
    );
}

PoseCandidate GlobalSearcher::runCoarseAndRefineSearch(
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
)
{
    const auto coarse_start_tp = std::chrono::steady_clock::now();
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
    const auto refine_start_tp = std::chrono::steady_clock::now();
    const double coarse_time_ms =
        std::chrono::duration<double, std::milli>(refine_start_tp - coarse_start_tp).count();

    if (!ok || top_coarse_poses.empty()) {
        const double total_time_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - search_start_tp
                                     ).count();
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            last_search_time_ms_ = total_time_ms;
        }
        RCLCPP_ERROR(
            logger_, "[GlobalSearch] GPU Global Search Failed. (elapsed: %.2f ms)", total_time_ms
        );
        setSearchFailureOutputs(out_best_cost, out_inlier_count, out_inlier_cost);
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    struct RefinedCandidate {
        PoseCandidate pose;
        float cost;
        float cost_3d;
        float cost_2d;
        int inliers;
        float inlier_cost;
        bool accepted;
    };

    std::vector<float> dummy_dynamic;
    std::vector<RefinedCandidate> refined_candidates;
    refined_candidates.reserve(top_coarse_poses.size());
    int best_idx = -1;

    for (const auto& coarse_candidate : top_coarse_poses) {
        PoseCandidate cand_pose{};
        float cand_cost        = std::numeric_limits<float>::max();
        int cand_inliers       = 0;
        float cand_inlier_cost = 1.0f;

        bool matched = solver.matchObstacleCloud(
            obstacle_count,
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

        // Enforce 3D LiDAR primacy during global recovery: when 3D obstacle points are available,
        // normalize any candidate where 2D failed acceptance back to 3D cost scale.
        if (solver.obstacleCount3D() >= match_params.min_3d_points &&
            (solver.lastWeight2D() >= 1.0f ||
             solver.lastCost2D() >= match_params.cost_threshold_2d)) {
            cand_cost = solver.lastCost3D();
        }
        const bool is_accepted =
            (solver.obstacleCount3D() >= match_params.min_3d_points)
                ? match_params.isMatchAccepted(
                      solver.lastCost3D(),
                      std::numeric_limits<float>::max(),
                      solver.obstacleCount3D(),
                      0
                  )
                : match_params.isMatchAccepted(
                      solver.lastCost3D(),
                      solver.lastCost2D(),
                      solver.obstacleCount3D(),
                      solver.obstacleCount2D()
                  );

        refined_candidates.push_back({
            cand_pose,
            cand_cost,
            solver.lastCost3D(),
            solver.lastCost2D(),
            cand_inliers,
            cand_inlier_cost,
            is_accepted,
        });

        if (is_accepted) {
            if (best_idx < 0 || cand_cost < refined_candidates[best_idx].cost) {
                best_idx = static_cast<int>(refined_candidates.size()) - 1;
            }
        }
    }

    if (best_idx < 0) {
        const auto search_end_tp = std::chrono::steady_clock::now();
        const double refine_time_ms =
            std::chrono::duration<double, std::milli>(search_end_tp - refine_start_tp).count();
        const double total_time_ms =
            std::chrono::duration<double, std::milli>(search_end_tp - search_start_tp).count();
        {
            std::lock_guard<std::mutex> lock(config_mutex_);
            last_search_time_ms_ = total_time_ms;
        }
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] None of the Top-%zu candidates satisfied strict acceptance criteria. "
            "(total: %.2f ms, prep: %.2f ms, coarse: %.2f ms, refine: %.2f ms)",
            top_coarse_poses.size(),
            total_time_ms,
            prep_time_ms,
            coarse_time_ms,
            refine_time_ms
        );
        solver.setLastCosts(std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        setSearchFailureOutputs(out_best_cost, out_inlier_count, out_inlier_cost);
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    const RefinedCandidate& best = refined_candidates[best_idx];

    // Check Top-1 vs Top-2 uniqueness margin across distinct pose modes.
    float second_best_cost = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < refined_candidates.size(); ++i) {
        if (i == static_cast<size_t>(best_idx)) {
            continue;
        }
        const auto& cand    = refined_candidates[i];
        const float dist_xy = std::hypot(cand.pose.x - best.pose.x, cand.pose.y - best.pose.y);
        const float dyaw    = std::abs(normalizeAngle(cand.pose.yaw - best.pose.yaw));
        if (dist_xy > 0.30f || dyaw > 0.25f) {
            second_best_cost = std::min(second_best_cost, cand.cost);
        }
    }

    const bool fails_uniqueness =
        (config.uniqueness_ratio < 1.0f) && std::isfinite(second_best_cost) &&
        (best.cost > config.uniqueness_ratio * second_best_cost);

    const auto search_end_tp = std::chrono::steady_clock::now();
    const double refine_time_ms =
        std::chrono::duration<double, std::milli>(search_end_tp - refine_start_tp).count();
    const double total_time_ms =
        std::chrono::duration<double, std::milli>(search_end_tp - search_start_tp).count();
    {
        std::lock_guard<std::mutex> lock(config_mutex_);
        last_search_time_ms_ = total_time_ms;
    }

    if (fails_uniqueness) {
        RCLCPP_WARN(
            logger_,
            "[GlobalSearch] Top-1 candidate rejected by uniqueness check "
            "(best: %.4f, second: %.4f, ratio: %.3f/%.3f, total: %.2f ms, prep: %.2f ms, coarse: %.2f ms, refine: %.2f ms).",
            best.cost,
            second_best_cost,
            best.cost / std::max(1e-6f, second_best_cost),
            config.uniqueness_ratio,
            total_time_ms,
            prep_time_ms,
            coarse_time_ms,
            refine_time_ms
        );
        solver.setLastCosts(
            std::numeric_limits<float>::max(), std::numeric_limits<float>::max()
        );
        setSearchFailureOutputs(out_best_cost, out_inlier_count, out_inlier_cost);
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    solver.setLastCosts(best.cost_3d, best.cost_2d);

    out_best_cost = best.cost;
    if (out_inlier_count != nullptr) {
        *out_inlier_count = best.inliers;
    }
    if (out_inlier_cost != nullptr) {
        *out_inlier_cost = best.inlier_cost;
    }

    RCLCPP_INFO(
        logger_,
        "[GlobalSearch] Complete (refined %zu candidates, total: %.2f ms, prep: %.2f ms, coarse: %.2f ms, refine: %.2f ms). "
        "Best Pose: (%.2f, %.2f, %.2f rad), Cost: %.4f, Inliers: %d",
        top_coarse_poses.size(),
        total_time_ms,
        prep_time_ms,
        coarse_time_ms,
        refine_time_ms,
        best.pose.x,
        best.pose.y,
        best.pose.yaw,
        out_best_cost,
        best.inliers
    );

    return best.pose;
}
