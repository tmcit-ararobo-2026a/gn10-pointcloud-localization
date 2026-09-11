#pragma once
#include <cuda_runtime.h>

struct PlatformConstraint {
    float center_x;
    float center_y;
    float distance_m;
    float yaw_error_rad;
    bool valid;
};
void launchPlatformDetector(
    const float* d_obstacle_cloud,
    int num_obstacle_points,
    float z_min,
    float z_max,
    int max_ransac_iter,
    float ransac_thresh,
    PlatformConstraint& out_constraint
);