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
    float max_dist_thresh{0.20f};
    float cost_threshold{0.20f};
};

class PoseSolver
{
public:
    explicit PoseSolver(int max_points = 200000);
    ~PoseSolver();

    void setMap(const std::vector<FieldObject>& host_map);

    bool processPointCloud(
        const std::vector<float>& h_raw_cloud,
        const float h_transform[12],
        const GroundFilterParams& filter_params,
        const MatchingParams& match_params,
        const PoseCandidate& search_base_pose,
        std::vector<float>& out_ground_pts,
        std::vector<float>& out_obstacle_pts,
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
    float* h_out_ground_{nullptr};
    float* h_out_obstacle_{nullptr};
};