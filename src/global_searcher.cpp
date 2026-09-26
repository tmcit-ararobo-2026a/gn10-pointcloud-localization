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
    float& out_best_cost
)
{
    RCLCPP_INFO(logger_, "[GlobalSearch] Starting GPU-Batched Yaw-Constrained Search...");

    // 点群のダウンサンプリング
    std::vector<float> search_cloud;
    const size_t total_points = h_raw_cloud.size() / 3;
    const int stride          = std::max(1, config_.downsample_stride);
    search_cloud.reserve((total_points / stride) * 3);

    for (size_t i = 0; i < total_points; i += stride) {
        search_cloud.push_back(h_raw_cloud[i * 3 + 0]);
        search_cloud.push_back(h_raw_cloud[i * 3 + 1]);
        search_cloud.push_back(h_raw_cloud[i * 3 + 2]);
    }

    // GPU 上に障害物点群を展開
    int obstacle_count = solver.prepareObstacleCloud(search_cloud, h_transform, filter_params);
    if (obstacle_count <= 50) {
        RCLCPP_WARN(logger_, "[GlobalSearch] Too few obstacle points. Search aborted.");
        out_best_cost = std::numeric_limits<float>::max();
        return PoseCandidate{0.0f, 0.0f, 0.0f};
    }

    // IMUの事前Yaw角を中心に、指定された許容範囲(max_yaw_diff)のみを探索窓に設定
    PoseCandidate center_pose;
    center_pose.x   = (config_.range_min_x + config_.range_max_x) * 0.5f;
    center_pose.y   = (config_.range_min_y + config_.range_max_y) * 0.5f;
    center_pose.yaw = prior_yaw;  // 事前情報をセット

    float range_x   = (config_.range_max_x - config_.range_min_x) * 0.5f;
    float range_y   = (config_.range_max_y - config_.range_min_y) * 0.5f;
    float range_yaw = max_yaw_diff;  // 全周(PI)ではなく制限範囲を設定

    PoseCandidate best_coarse_pose;
    float coarse_cost = std::numeric_limits<float>::max();

    // GPU 探索実行
    bool ok = solver.evaluateGlobalSDF(
        obstacle_count,
        center_pose,
        range_x,
        range_y,
        config_.step_xy,
        range_yaw,
        config_.step_yaw,
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
        out_best_cost
    );

    if (refined_pose.x < config_.range_min_x || refined_pose.x > config_.range_max_x ||
        refined_pose.y < config_.range_min_y || refined_pose.y > config_.range_max_y) {
        RCLCPP_WARN(logger_, "[GlobalSearch] Refined pose is outside the configured start area.");
        out_best_cost = std::numeric_limits<float>::max();
    }

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
