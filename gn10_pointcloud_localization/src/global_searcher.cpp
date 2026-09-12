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
    std::vector<float>& out_ground_pts,
    std::vector<float>& out_obstacle_pts,
    float& out_best_cost
)
{
    RCLCPP_INFO(logger_, "[GlobalSearch] Executing sync global localization...");

    // 間引き（インターリーブ抽出）
    std::vector<float> search_cloud;
    const size_t total_points = h_raw_cloud.size() / 3;
    const int stride          = std::max(1, config_.downsample_stride);
    search_cloud.reserve((total_points / stride) * 3);

    for (size_t i = 0; i < total_points; i += stride) {
        search_cloud.push_back(h_raw_cloud[i * 3 + 0]);
        search_cloud.push_back(h_raw_cloud[i * 3 + 1]);
        search_cloud.push_back(h_raw_cloud[i * 3 + 2]);
    }

    float min_cost = std::numeric_limits<float>::max();
    PoseCandidate best_coarse_pose{0.0f, 0.0f, 0.0f};

    MatchingParams global_match_params = match_params;
    global_match_params.range_xy       = 0.00f;  // ピンポイント評価
    global_match_params.range_yaw      = 0.00f;

    std::vector<float> tmp_ground, tmp_obstacle, tmp_dynamic;
    PoseCandidate tmp_pose;
    float tmp_cost = 0.0f;

    for (float x = config_.range_min_x; x <= config_.range_max_x; x += config_.step_xy) {
        for (float y = config_.range_min_y; y <= config_.range_max_y; y += config_.step_xy) {
            for (float yaw = -M_PI; yaw < M_PI; yaw += config_.step_yaw) {
                PoseCandidate candidate_pose{x, y, yaw};
                bool ok = solver.processPointCloud(
                    search_cloud,
                    h_transform,
                    filter_params,
                    global_match_params,
                    candidate_pose,
                    tmp_ground,
                    tmp_obstacle,
                    tmp_dynamic,
                    tmp_pose,
                    tmp_cost
                );

                if (ok && tmp_cost < min_cost) {
                    min_cost         = tmp_cost;
                    best_coarse_pose = candidate_pose;
                }
            }
        }
    }

    // 元の Raw 点群を使用してリファイン
    PoseCandidate refined_pose;
    solver.processPointCloud(
        h_raw_cloud,
        h_transform,
        filter_params,
        match_params,
        best_coarse_pose,
        out_ground_pts,
        out_obstacle_pts,
        tmp_dynamic,
        refined_pose,
        out_best_cost
    );

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