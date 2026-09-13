#pragma once
#include <cuda_runtime.h>

void launchGroundFilter(
    cudaStream_t stream,
    const float* d_in,
    float* d_ground,
    float* d_obstacle,
    const float* d_transform,
    int num_points,
    float range_max,
    float robot_radius,
    float robot_height_min,
    float robot_height_max,
    float ground_z_thresh,
    int* d_ground_count,
    int* d_obstacle_count,
    int* h_ground_count,
    int* h_obstacle_count
);