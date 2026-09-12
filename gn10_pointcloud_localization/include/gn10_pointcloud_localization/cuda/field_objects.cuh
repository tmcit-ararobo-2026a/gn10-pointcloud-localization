#pragma once
#include <cuda_runtime.h>

#include <vector>

enum ObjectType { CYLINDER, BOX };

struct FieldObject {
    ObjectType type;
    float center_x, center_y;
    float z_min, z_max;
    float param1, param2;
};

struct PoseCandidate {
    float x, y, yaw;
};

// C++ / NVCC 間のシンボル不一致を防ぐための extern "C" 宣言
#ifdef __cplusplus
extern "C" {
#endif

void uploadFieldMapToGPU(const std::vector<FieldObject>& host_map);

bool launchFieldSDFMatcher(
    cudaStream_t stream,
    const float* d_obstacle_cloud,
    int num_points,
    const PoseCandidate& base_pose,
    float range_xy,
    float step_xy,
    float range_yaw,
    float step_yaw,
    float max_dist_thresh,
    PoseCandidate& out_best_pose,
    float& out_best_cost
);

#ifdef __cplusplus
}
#endif