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
    float dynamic_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    PoseCandidate& out_best_pose,
    float& out_best_cost,
    std::vector<float>& out_dynamic_pts,
    bool extract_dynamic = true  // false の場合は動的点群抽出カーネルとD2H転送を一切実行しない
);

#ifdef __cplusplus
}
#endif