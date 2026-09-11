#pragma once
#include <cuda_runtime.h>

// CPU (C++) 側から呼び出すためのラッパー関数の宣言
void launchGroundFilter(
    const float* d_in,
    float* d_ground,
    float* d_obstacle,
    int num_points,
    float range_max,
    float ground_z_thresh,
    int* d_ground_count,
    int* d_obstacle_count,
    int* h_ground_count,
    int* h_obstacle_count
);