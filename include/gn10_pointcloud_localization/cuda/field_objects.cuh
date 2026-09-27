#pragma once
#include <cuda_runtime.h>

#include <vector>

enum ObjectType { CYLINDER, BOX, VISUAL_BOX };

struct FieldObject {
    ObjectType type;
    float center_x, center_y;
    float z_min, z_max;
    float param1, param2;
};

struct FieldMatchStats { int support_count{0}; float support_ratio{0}; int sectors{0}; int axis_x{0}, axis_y{0}; float residual{0}; float ranking_cost{0}; float initial_residual{0}; bool refined{false}; };

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
    float range_x,
    float range_y,
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
    bool extract_dynamic = true,
    float robust_distance = 0.0f,
    float min_support_ratio = 0.0f,
    int min_support_count = 0,
    int min_support_sectors = 0,
    FieldMatchStats* stats = nullptr,
    int min_axis_support = 0
);

#ifdef __cplusplus
}
#endif
