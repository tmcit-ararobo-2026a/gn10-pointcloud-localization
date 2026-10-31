// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"
#include "gn10_pointcloud_localization/cuda/esdf_matcher.cuh"
#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"
#include "gn10_pointcloud_localization/esdf_map.hpp"

inline float normalizeAngle(float angle)
{
    if (!std::isfinite(angle)) {
        return 0.0f;
    }
    float a = std::fmod(angle + static_cast<float>(M_PI), static_cast<float>(2.0 * M_PI));
    if (a < 0.0f) {
        a += static_cast<float>(2.0 * M_PI);
    }
    return a - static_cast<float>(M_PI);
}

struct GroundFilterParams {
    float range_max{12.0f};
    float robot_radius{0.6f};
    float robot_height_min{0.0f};
    float robot_height_max{1.2f};
    float ground_z_thresh{0.08f};
};

struct MatchingParams {
    float range_xy{0.30f};
    float step_xy{0.03f};
    float range_yaw{0.15f};
    float step_yaw{0.02f};
    bool fine_refine{true};
    int fine_refine_levels{2};
    float max_dist_thresh{0.20f};
    float cost_threshold{0.165f};
    float dynamic_dist_thresh{0.15f};
    float field_min_x{-5.5f};
    float field_max_x{5.5f};
    float field_min_y{-6.0f};
    float field_max_y{6.0f};
    float inlier_dist_thresh{0.08f};
    int min_inliers{60};
    float inlier_cost_thresh{0.05f};
    float weight_2d_lidar{0.40f};
    int min_2d_points{30};
    int target_2d_points{200};

    void useESDFBounds(const ESDFHeader& header)
    {
        field_min_x = header.min_x;
        field_max_x = header.min_x + header.size_x * header.resolution;
        field_min_y = header.min_y;
        field_max_y = header.min_y + header.size_y * header.resolution;
    }
};

class PoseSolver
{
public:
    explicit PoseSolver(int max_points = 200000);
    ~PoseSolver();

    void setMap(const std::vector<FieldObject>& host_map);
    void setESDFMap(const ESDFMap& esdf_map);

    // 通常追従用（GroundFilter + SDF Matcher + Dynamic Point 抽出）
    bool processPointCloud(
        const std::vector<float>& h_raw_cloud,
        const float h_transform[12],
        const GroundFilterParams& filter_params,
        const MatchingParams& match_params,
        const PoseCandidate& search_base_pose,
        std::vector<float>& out_dynamic_pts,
        PoseCandidate& out_best_pose,
        float& out_best_cost,
        int* out_inlier_count                            = nullptr,
        float* out_inlier_cost                           = nullptr,
        const std::vector<float>* extra_base_link_points = nullptr
    );

    // Deskew 統合通常追従用（GPU Deskew + GroundFilter + SDF Matcher + Dynamic Point 抽出）
    bool processPointCloud(
        const std::vector<TimedScanPoint>& timed_points,
        const float h_transform[12],
        const DeskewParams& deskew_params,
        const std::vector<IMUSample>& imu_samples,
        int stride,
        const GroundFilterParams& filter_params,
        const MatchingParams& match_params,
        const PoseCandidate& search_base_pose,
        std::vector<float>& out_dynamic_pts,
        PoseCandidate& out_best_pose,
        float& out_best_cost,
        int* out_inlier_count                            = nullptr,
        float* out_inlier_cost                           = nullptr,
        const std::vector<float>* extra_base_link_points = nullptr
    );

    // 確定姿勢を用いた動的点群の抽出 (Pose探索をスキップ)
    bool extractDynamicPointsOnly(
        const std::vector<float>& h_raw_cloud,
        const PoseCandidate& confirmed_pose,
        float dynamic_dist_thresh,
        float field_min_x,
        float field_max_x,
        float field_min_y,
        float field_max_y,
        std::vector<float>& out_dynamic_pts,
        const float* tracking_sphere = nullptr
    );

    // 点群の前処理（GroundFilter）のみを実行し GPU 上の d_obstacle_ に保持する
    int prepareObstacleCloud(
        const std::vector<float>& h_raw_cloud,
        const float h_transform[12],
        const GroundFilterParams& filter_params
    );

    // 点群の前処理（GPU Deskew + GroundFilter）を実行し GPU 上の d_obstacle_ に保持する
    int prepareObstacleCloud(
        const std::vector<TimedScanPoint>& timed_points,
        const float h_transform[12],
        const DeskewParams& deskew_params,
        const std::vector<IMUSample>& imu_samples,
        int stride,
        const GroundFilterParams& filter_params
    );

    // Copy the filtered clouds for RViz only when a subscriber needs them.
    void copyFilteredClouds(std::vector<float>* ground, std::vector<float>* obstacle);

    // 既に d_obstacle_ に保持されている点群に対して、全域候補の SDF Matcher を一括起動する
    bool evaluateGlobalSDF(
        int obstacle_count,
        const PoseCandidate& base_pose,
        float range_x,
        float range_y,
        float step_xy,
        float range_yaw,
        float step_yaw,
        const MatchingParams& match_params,
        PoseCandidate& out_best_pose,
        float& out_best_cost
    );

    struct ProfileMetrics {
        double time_ground_filter_ms{0.0};
        double time_matching_ms{0.0};
        double time_dynamic_extraction_ms{0.0};
        size_t d2h_transfer_bytes{0};
    };
    const ProfileMetrics& lastProfileMetrics() const { return last_metrics_; }
    int obstacleCount() const { return obstacle_count_; }
    int obstacleCount3D() const { return num_3d_obstacles_; }
    int obstacleCount2D() const { return num_2d_obstacles_; }
    float lastCost3D() const { return last_cost_3d_; }
    float lastCost2D() const { return last_cost_2d_; }
    float lastWeight2D() const { return last_weight_2d_; }

private:
    int max_points_;
    cudaStream_t stream_{nullptr};

    // Device Memory
    float* d_in_{nullptr};
    TimedScanPoint* d_timed_in_{nullptr};
    IMUSample* d_imu_samples_{nullptr};
    float* d_ground_{nullptr};
    float* d_obstacle_{nullptr};
    float* d_transform_{nullptr};
    int* d_ground_count_{nullptr};
    int* d_obstacle_count_{nullptr};

    // Host Pinned Memory
    float* h_in_{nullptr};
    int ground_count_{0};
    int obstacle_count_{0};
    int num_3d_obstacles_{0};
    int num_2d_obstacles_{0};
    float last_cost_3d_{0.0f};
    float last_cost_2d_{0.0f};
    float last_weight_2d_{0.0f};
    bool use_esdf_{false};
    ProfileMetrics last_metrics_{};
    bool matchObstacleCloud(
        int h_obstacle_count,
        const MatchingParams& match_params,
        const PoseCandidate& search_base_pose,
        std::vector<float>& out_dynamic_pts,
        PoseCandidate& out_best_pose,
        float& out_best_cost,
        int* out_inlier_count,
        float* out_inlier_cost
    );
};
