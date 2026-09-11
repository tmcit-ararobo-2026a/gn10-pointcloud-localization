#include <cuda_runtime.h>

#include <iostream>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

// GPU 上のメモリ保持用（再利用して malloc のオーバーヘッドを削減）
static FieldObject* d_map_objects = nullptr;
static int g_num_map_objects      = 0;

static PoseCandidate* d_candidates = nullptr;
static float* d_costs              = nullptr;
static int g_max_candidates        = 0;

// マップオブジェクトの GPU メモリへの初回ロード
void uploadFieldMapToGPU(const std::vector<FieldObject>& host_map)
{
    if (d_map_objects) cudaFree(d_map_objects);

    g_num_map_objects = host_map.size();
    size_t bytes      = g_num_map_objects * sizeof(FieldObject);
    cudaMalloc(&d_map_objects, bytes);
    cudaMemcpy(d_map_objects, host_map.data(), bytes, cudaMemcpyHostToDevice);
}

// グリッドサーチ実行関数
PoseCandidate runGPUGridSearch(
    const float* d_obstacle_cloud,
    int num_points,
    const std::vector<PoseCandidate>& host_candidates,
    float max_dist_thresh
)
{
    int num_candidates = host_candidates.size();
    if (num_candidates == 0 || num_points == 0) {
        return {0.0f, 0.0f, 0.0f};
    }

    // 候補姿勢用バッファの再確保判定
    if (num_candidates > g_max_candidates) {
        if (d_candidates) cudaFree(d_candidates);
        if (d_costs) cudaFree(d_costs);
        g_max_candidates = num_candidates;
        cudaMalloc(&d_candidates, g_max_candidates * sizeof(PoseCandidate));
        cudaMalloc(&d_costs, g_max_candidates * sizeof(float));
    }

    // 候補姿勢の転送
    cudaMemcpy(
        d_candidates,
        host_candidates.data(),
        num_candidates * sizeof(PoseCandidate),
        cudaMemcpyHostToDevice
    );

    // 1ブロック = 1つの候補姿勢、256スレッドで点群を分散評価
    int threads_per_block = 256;
    int blocks_per_grid   = num_candidates;

    evaluateFieldSDFKernel<<<blocks_per_grid, threads_per_block>>>(
        d_obstacle_cloud,
        num_points,
        d_map_objects,
        g_num_map_objects,
        d_candidates,
        d_costs,
        max_dist_thresh
    );

    // コストの読み出し
    std::vector<float> host_costs(num_candidates);
    cudaMemcpy(host_costs.data(), d_costs, num_candidates * sizeof(float), cudaMemcpyDeviceToHost);

    // 最小コストの候補を探索
    int best_idx   = 0;
    float min_cost = host_costs[0];
    for (int i = 1; i < num_candidates; ++i) {
        if (host_costs[i] < min_cost) {
            min_cost = host_costs[i];
            best_idx = i;
        }
    }

    return host_candidates[best_idx];
}