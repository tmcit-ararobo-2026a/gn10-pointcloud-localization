#pragma once

#include <cuda_runtime.h>

#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

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
    float max_dist_thresh{0.20f};
    float cost_threshold{0.165f};
    float dynamic_dist_thresh{0.15f};
    float field_min_x{-5.5f};
    float field_max_x{5.5f};
    float field_min_y{-6.0f};
    float field_max_y{6.0f};
};

class PoseSolver
{
public:
    explicit PoseSolver(int max_points = 200000);
    ~PoseSolver();

    void setMap(const std::vector<FieldObject>& host_map);

    // 通常追従用（GroundFilter + SDF Matcher + Dynamic Point 抽出）
    bool processPointCloud(
        const std::vector<float>& h_raw_cloud,
        const float h_transform[12],
        const GroundFilterParams& filter_params,
        const MatchingParams& match_params,
        const PoseCandidate& search_base_pose,
        std::vector<float>& out_dynamic_pts,
        PoseCandidate& out_best_pose,
        float& out_best_cost
    );

    // 点群の前処理（GroundFilter）のみを実行し GPU 上の d_obstacle_ に保持する
    int prepareObstacleCloud(
        const std::vector<float>& h_raw_cloud,
        const float h_transform[12],
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

private:
    int max_points_;
    cudaStream_t stream_{nullptr};

    // Device Memory
    float* d_in_{nullptr};
    float* d_ground_{nullptr};
    float* d_obstacle_{nullptr};
    float* d_transform_{nullptr};
    int* d_ground_count_{nullptr};
    int* d_obstacle_count_{nullptr};

    // Host Pinned Memory
    float* h_in_{nullptr};
    int ground_count_{0};
    int obstacle_count_{0};
};
