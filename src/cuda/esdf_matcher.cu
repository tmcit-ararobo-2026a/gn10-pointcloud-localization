// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <iostream>
#include <vector>
#include <stdexcept>

#include "gn10_pointcloud_localization/cuda/esdf_matcher.cuh"

// 3D Texture Memory リソース
static cudaArray_t g_esdf_array       = nullptr;
static cudaTextureObject_t g_esdf_tex = 0;
__constant__ ESDFHeader c_esdf_header;
static bool g_esdf_loaded = false;

// マッチング用バッファ
static PoseCandidate* d_candidates = nullptr;
static float* d_costs              = nullptr;
static int g_max_candidates        = 0;

static uint8_t* d_is_dynamic = nullptr;
static int g_max_dynamic_pts = 0;

struct Point3D {
    float x, y, z;
};
static_assert(sizeof(Point3D) == 3 * sizeof(float), "Point3D must be packed XYZ");
static Point3D* d_compact_dynamic_pts = nullptr;
static int* d_compact_dynamic_count   = nullptr;
static void* d_compact_temp_storage   = nullptr;
static size_t g_compact_temp_bytes    = 0;

static cub::KeyValuePair<int, float>* d_out_argmin = nullptr;
static void* d_temp_storage                        = nullptr;
static size_t temp_storage_bytes                   = 0;

static int* d_inlier_count      = nullptr;
static float* d_inlier_dist_sum = nullptr;
static float* d_cost_3d_sum     = nullptr;
static float* d_cost_2d_sum     = nullptr;

// 3D Texture からハードウェア Trilinear 補間を用いて距離を取得するデバイス関数
// tex3D は非正規化座標モードの場合、中心が 0.5f オフセットとなります
__device__ inline float queryESDF(
    cudaTextureObject_t tex, const ESDFHeader& header, float wx, float wy, float wz
)
{
    // グリッド範囲内か判定
    if (wx < header.min_x || wy < header.min_y || wz < header.min_z) {
        return header.max_dist_thresh;
    }
    // Builders sample at min + (index + 0.5) * resolution. That world
    // position already maps to the CUDA texel center index + 0.5.
    float u = (wx - header.min_x) / header.resolution;
    float v = (wy - header.min_y) / header.resolution;
    float w = (wz - header.min_z) / header.resolution;

    if (u >= static_cast<float>(header.size_x) || v >= static_cast<float>(header.size_y) ||
        w >= static_cast<float>(header.size_z)) {
        return header.max_dist_thresh;
    }

    return tex3D<float>(tex, u, v, w);
}

// 全姿勢候補の残差計算カーネル (3D Texture 版: O(1) 参照)
// 3D 点群 (MID-360) と 2D 点群 (Lakibeam) で正規化平均コストを別々に計算し、重み付け合成
__global__ void evaluateESDFKernel(
    cudaTextureObject_t tex,
    const float* __restrict__ cloud,
    int num_points,
    int num_3d_points,
    int num_2d_points,
    float weight_2d,
    const PoseCandidate* __restrict__ candidates,
    float* __restrict__ out_costs,
    float max_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y
)
{
    int pose_idx = blockIdx.x;
    int tid      = threadIdx.x;

    float rx   = candidates[pose_idx].x;
    float ry   = candidates[pose_idx].y;
    float ryaw = candidates[pose_idx].yaw;

    float cos_y = cosf(ryaw);
    float sin_y = sinf(ryaw);

    __shared__ float s_cost_3d[256];
    __shared__ float s_cost_2d[256];
    s_cost_3d[tid] = 0.0f;
    s_cost_2d[tid] = 0.0f;

    for (int i = tid; i < num_points; i += blockDim.x) {
        float lx = cloud[i * 3 + 0];
        float ly = cloud[i * 3 + 1];
        float lz = cloud[i * 3 + 2];

        // map 座標系へ変換
        float wx = cos_y * lx - sin_y * ly + rx;
        float wy = sin_y * lx + cos_y * ly + ry;
        float wz = lz;

        float cost_i = max_dist_thresh;
        if (wx >= field_min_x && wx <= field_max_x && wy >= field_min_y && wy <= field_max_y) {
            // 3D Texture Memory 参照 (O(1) & ハードウェアトライリニア補間)
            float d = queryESDF(tex, c_esdf_header, wx, wy, wz);
            cost_i = fminf(d, max_dist_thresh);
        }

        if (i < num_3d_points) {
            s_cost_3d[tid] += cost_i;
        } else {
            s_cost_2d[tid] += cost_i;
        }
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_cost_3d[tid] += s_cost_3d[tid + s];
            s_cost_2d[tid] += s_cost_2d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        float mean_3d = (num_3d_points > 0) ? (s_cost_3d[0] / static_cast<float>(num_3d_points)) : max_dist_thresh;
        float mean_2d = (num_2d_points > 0) ? (s_cost_2d[0] / static_cast<float>(num_2d_points)) : max_dist_thresh;

        if (num_2d_points > 0 && num_3d_points > 0) {
            out_costs[pose_idx] = (1.0f - weight_2d) * mean_3d + weight_2d * mean_2d;
        } else if (num_2d_points > 0) {
            out_costs[pose_idx] = mean_2d;
        } else {
            out_costs[pose_idx] = mean_3d;
        }
    }
}

__global__ void filterDynamicPointsESDFKernel(
    cudaTextureObject_t tex,
    const float* __restrict__ cloud,
    int num_points,
    PoseCandidate best_pose,
    float dynamic_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    float4 keep_sphere,
    uint8_t* __restrict__ out_is_dynamic
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    float rx   = best_pose.x;
    float ry   = best_pose.y;
    float ryaw = best_pose.yaw;

    float cos_y = cosf(ryaw);
    float sin_y = sinf(ryaw);

    float lx = cloud[i * 3 + 0];
    float ly = cloud[i * 3 + 1];
    float lz = cloud[i * 3 + 2];

    const float dx=lx-keep_sphere.x, dy=ly-keep_sphere.y, dz=lz-keep_sphere.z;
    if (keep_sphere.w>0 && dx*dx+dy*dy+dz*dz<=keep_sphere.w*keep_sphere.w) {
        // 床面近傍の点は保護対象から除外 (床面ノイズの固定化を防止)
        if (lz > 0.02f) {
            out_is_dynamic[i]=1; return;
        }
    }
    float wx = cos_y * lx - sin_y * ly + rx;
    float wy = sin_y * lx + cos_y * ly + ry;
    float wz = lz;

    if (wx < field_min_x || wx > field_max_x || wy < field_min_y || wy > field_max_y) {
        out_is_dynamic[i] = 0;
        return;
    }

    float d           = queryESDF(tex, c_esdf_header, wx, wy, wz);
    out_is_dynamic[i] = (d >= dynamic_dist_thresh) ? 1 : 0;
}

__global__ void computeBestPoseMetricsESDFKernel(
    cudaTextureObject_t tex,
    const float* __restrict__ cloud,
    int num_points,
    int num_3d_points,
    int num_2d_points,
    PoseCandidate best_pose,
    float inlier_dist_thresh,
    float max_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    int* __restrict__ out_inlier_count,
    float* __restrict__ out_inlier_dist_sum,
    float* __restrict__ out_cost_3d_sum,
    float* __restrict__ out_cost_2d_sum
)
{
    __shared__ int s_count[256];
    __shared__ float s_dist[256];
    __shared__ float s_cost_3d[256];
    __shared__ float s_cost_2d[256];

    int tid      = threadIdx.x;
    s_count[tid] = 0;
    s_dist[tid]  = 0.0f;
    s_cost_3d[tid] = 0.0f;
    s_cost_2d[tid] = 0.0f;

    float rx   = best_pose.x;
    float ry   = best_pose.y;
    float ryaw = best_pose.yaw;

    float cos_y = cosf(ryaw);
    float sin_y = sinf(ryaw);

    for (int i = blockIdx.x * blockDim.x + tid; i < num_points; i += gridDim.x * blockDim.x) {
        float lx = cloud[i * 3 + 0];
        float ly = cloud[i * 3 + 1];
        float lz = cloud[i * 3 + 2];

        float wx = cos_y * lx - sin_y * ly + rx;
        float wy = sin_y * lx + cos_y * ly + ry;
        float wz = lz;

        float cost_i = max_dist_thresh;
        if (wx >= field_min_x && wx <= field_max_x && wy >= field_min_y && wy <= field_max_y) {
            float d = queryESDF(tex, c_esdf_header, wx, wy, wz);
            cost_i = fminf(d, max_dist_thresh);
            if (d < inlier_dist_thresh) {
                s_count[tid] += 1;
                s_dist[tid] += d;
            }
        }

        if (i < num_3d_points) {
            s_cost_3d[tid] += cost_i;
        } else {
            s_cost_2d[tid] += cost_i;
        }
    }
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_count[tid] += s_count[tid + s];
            s_dist[tid] += s_dist[tid + s];
            s_cost_3d[tid] += s_cost_3d[tid + s];
            s_cost_2d[tid] += s_cost_2d[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out_inlier_count, s_count[0]);
        atomicAdd(out_inlier_dist_sum, s_dist[0]);
        if (out_cost_3d_sum) atomicAdd(out_cost_3d_sum, s_cost_3d[0]);
        if (out_cost_2d_sum) atomicAdd(out_cost_2d_sum, s_cost_2d[0]);
    }
}

static void checkDynamicCuda(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

static void extractDynamicPointsGPU(
    cudaStream_t stream,
    const float* d_cloud,
    int num_points,
    const PoseCandidate& pose,
    float dynamic_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    std::vector<float>& out_dynamic_pts,
    const float* tracking_sphere = nullptr
)
{
    out_dynamic_pts.clear();
    if (num_points <= 0 || !g_esdf_loaded) return;

    if (!d_compact_dynamic_pts || !d_is_dynamic || num_points > g_max_dynamic_pts) {
        if (d_is_dynamic) cudaFree(d_is_dynamic);
        if (d_compact_dynamic_pts) cudaFree(d_compact_dynamic_pts);
        g_max_dynamic_pts = num_points * 2;
        checkDynamicCuda(cudaMalloc(&d_is_dynamic, g_max_dynamic_pts * sizeof(uint8_t)));
        checkDynamicCuda(cudaMalloc(&d_compact_dynamic_pts, g_max_dynamic_pts * sizeof(Point3D)));
    }
    if (!d_compact_dynamic_count) {
        checkDynamicCuda(cudaMalloc(&d_compact_dynamic_count, sizeof(int)));
    }

    int dyn_threads = 256;
    int dyn_blocks  = (num_points + dyn_threads - 1) / dyn_threads;
    filterDynamicPointsESDFKernel<<<dyn_blocks, dyn_threads, 0, stream>>>(
        g_esdf_tex,
        d_cloud,
        num_points,
        pose,
        dynamic_dist_thresh,
        field_min_x,
        field_max_x,
        field_min_y,
        field_max_y,
        tracking_sphere ? make_float4(tracking_sphere[0],tracking_sphere[1],tracking_sphere[2],tracking_sphere[3]) : make_float4(0,0,0,0),
        d_is_dynamic
    );

    checkDynamicCuda(cudaGetLastError());

    const Point3D* d_in_pts = reinterpret_cast<const Point3D*>(d_cloud);

    // CUB DeviceSelect::Flagged による GPU 内ストリームコンパクション
    size_t required_temp_bytes = 0;
    checkDynamicCuda(cub::DeviceSelect::Flagged(
        nullptr, required_temp_bytes,
        d_in_pts, d_is_dynamic, d_compact_dynamic_pts, d_compact_dynamic_count,
        num_points, stream
    ));

    if (required_temp_bytes > g_compact_temp_bytes) {
        if (d_compact_temp_storage) cudaFree(d_compact_temp_storage);
        g_compact_temp_bytes = required_temp_bytes * 2;
        checkDynamicCuda(cudaMalloc(&d_compact_temp_storage, g_compact_temp_bytes));
    }

    checkDynamicCuda(cub::DeviceSelect::Flagged(
        d_compact_temp_storage, g_compact_temp_bytes,
        d_in_pts, d_is_dynamic, d_compact_dynamic_pts, d_compact_dynamic_count,
        num_points, stream
    ));

    int h_selected_count = 0;
    checkDynamicCuda(cudaMemcpyAsync(
        &h_selected_count, d_compact_dynamic_count, sizeof(int), cudaMemcpyDeviceToHost, stream
    ));
    checkDynamicCuda(cudaStreamSynchronize(stream));

    if (h_selected_count < 0 || h_selected_count > num_points) throw std::runtime_error("Invalid compacted point count");
    if (h_selected_count > 0) {
        out_dynamic_pts.resize(h_selected_count * 3);
        checkDynamicCuda(cudaMemcpyAsync(
            out_dynamic_pts.data(),
            d_compact_dynamic_pts,
            h_selected_count * sizeof(Point3D),
            cudaMemcpyDeviceToHost,
            stream
        ));
        checkDynamicCuda(cudaStreamSynchronize(stream));
    }
}

extern "C" {

void freeESDFMapFromGPU()
{
    if (g_esdf_tex) {
        cudaDestroyTextureObject(g_esdf_tex);
        g_esdf_tex = 0;
    }
    if (g_esdf_array) {
        cudaFreeArray(g_esdf_array);
        g_esdf_array = nullptr;
    }
    if (d_is_dynamic) {
        cudaFree(d_is_dynamic);
        d_is_dynamic    = nullptr;
        g_max_dynamic_pts = 0;
    }
    if (d_compact_dynamic_pts) {
        cudaFree(d_compact_dynamic_pts);
        d_compact_dynamic_pts = nullptr;
    }
    if (d_compact_dynamic_count) {
        cudaFree(d_compact_dynamic_count);
        d_compact_dynamic_count = nullptr;
    }
    if (d_compact_temp_storage) {
        cudaFree(d_compact_temp_storage);
        d_compact_temp_storage = nullptr;
        g_compact_temp_bytes = 0;
    }
    if (d_inlier_count) {
        cudaFree(d_inlier_count);
        d_inlier_count = nullptr;
    }
    if (d_inlier_dist_sum) {
        cudaFree(d_inlier_dist_sum);
        d_inlier_dist_sum = nullptr;
    }
    if (d_cost_3d_sum) {
        cudaFree(d_cost_3d_sum);
        d_cost_3d_sum = nullptr;
    }
    if (d_cost_2d_sum) {
        cudaFree(d_cost_2d_sum);
        d_cost_2d_sum = nullptr;
    }
    if (d_candidates) {
        cudaFree(d_candidates);
        d_candidates = nullptr;
    }
    if (d_costs) {
        cudaFree(d_costs);
        d_costs = nullptr;
    }
    if (d_out_argmin) {
        cudaFree(d_out_argmin);
        d_out_argmin = nullptr;
    }
    if (d_temp_storage) {
        cudaFree(d_temp_storage);
        d_temp_storage = nullptr;
        temp_storage_bytes = 0;
    }
    g_max_candidates = 0;
    g_esdf_loaded = false;
}

void uploadESDFMapToGPU(const ESDFHeader& header, const float* h_grid_data)
{
    freeESDFMapFromGPU();

    cudaMemcpyToSymbol(c_esdf_header, &header, sizeof(ESDFHeader));

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(header.size_x, header.size_y, header.size_z);

    cudaError_t err = cudaMalloc3DArray(&g_esdf_array, &channelDesc, extent);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ESDF] cudaMalloc3DArray failed: " << cudaGetErrorString(err)
                  << std::endl;
        return;
    }

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr            = make_cudaPitchedPtr(
        const_cast<float*>(h_grid_data), header.size_x * sizeof(float), header.size_x, header.size_y
    );
    copyParams.dstArray = g_esdf_array;
    copyParams.extent   = extent;
    copyParams.kind     = cudaMemcpyHostToDevice;

    err = cudaMemcpy3D(&copyParams);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ESDF] cudaMemcpy3D failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // テクスチャオブジェクトの設定 (Trilinear 補間 + Clamp)
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType         = cudaResourceTypeArray;
    resDesc.res.array.array = g_esdf_array;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0]   = cudaAddressModeClamp;
    texDesc.addressMode[1]   = cudaAddressModeClamp;
    texDesc.addressMode[2]   = cudaAddressModeClamp;
    texDesc.filterMode       = cudaFilterModeLinear;  // ハードウェアトライリニア補間
    texDesc.readMode         = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;  // 非正規化座標 (ボクセルインデックス基準)

    err = cudaCreateTextureObject(&g_esdf_tex, &resDesc, &texDesc, nullptr);
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ESDF] cudaCreateTextureObject failed: " << cudaGetErrorString(err)
                  << std::endl;
        return;
    }

    g_esdf_loaded = true;
    std::cout << "[CUDA ESDF] 3D Texture memory successfully allocated and loaded ("
              << header.size_x << "x" << header.size_y << "x" << header.size_z << ")" << std::endl;
}

bool launchESDFMatcher(
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
    bool extract_dynamic,
    int* out_inlier_count,
    float* out_inlier_cost,
    float inlier_dist_thresh,
    int num_3d_points,
    int num_2d_points,
    float weight_2d,
    float* out_cost_3d,
    float* out_cost_2d
)
{
    if (num_points <= 0 || !g_esdf_loaded) return false;

    const int actual_3d   = (num_3d_points >= 0) ? num_3d_points : num_points;
    const int actual_2d   = (num_3d_points >= 0) ? num_2d_points : 0;
    const float actual_w2d = (num_3d_points >= 0) ? weight_2d : 0.0f;

    // 姿勢候補の生成 (Host)
    std::vector<PoseCandidate> h_candidates;
    for (float dx = -range_x; dx <= range_x + 1e-5f; dx += step_xy) {
        for (float dy = -range_y; dy <= range_y + 1e-5f; dy += step_xy) {
            for (float dyaw = -range_yaw; dyaw <= range_yaw + 1e-5f; dyaw += step_yaw) {
                h_candidates.push_back({base_pose.x + dx, base_pose.y + dy, base_pose.yaw + dyaw});
            }
        }
    }

    int num_candidates = static_cast<int>(h_candidates.size());
    if (num_candidates == 0) return false;

    // バッファ確保
    if (num_candidates > g_max_candidates) {
        if (d_candidates) cudaFree(d_candidates);
        if (d_costs) cudaFree(d_costs);
        if (d_out_argmin) cudaFree(d_out_argmin);
        if (d_temp_storage) cudaFree(d_temp_storage);

        g_max_candidates = num_candidates * 2;
        cudaMalloc(&d_candidates, g_max_candidates * sizeof(PoseCandidate));
        cudaMalloc(&d_costs, g_max_candidates * sizeof(float));
        cudaMalloc(&d_out_argmin, sizeof(cub::KeyValuePair<int, float>));

        d_temp_storage     = nullptr;
        temp_storage_bytes = 0;
        cub::DeviceReduce::ArgMin(
            d_temp_storage, temp_storage_bytes, d_costs, d_out_argmin, num_candidates, stream
        );
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
    }

    if (!d_inlier_count) {
        cudaMalloc(&d_inlier_count, sizeof(int));
        cudaMalloc(&d_inlier_dist_sum, sizeof(float));
    }
    if (!d_cost_3d_sum) {
        cudaMalloc(&d_cost_3d_sum, sizeof(float));
        cudaMalloc(&d_cost_2d_sum, sizeof(float));
    }

    // H2D
    cudaMemcpyAsync(
        d_candidates,
        h_candidates.data(),
        num_candidates * sizeof(PoseCandidate),
        cudaMemcpyHostToDevice,
        stream
    );

    // ESDF 評価カーネル (3D Texture O(1)) - 3D/2D 独立残差項 & 重み付け合成
    int sdf_threads = 256;
    int sdf_blocks  = num_candidates;
    evaluateESDFKernel<<<sdf_blocks, sdf_threads, 0, stream>>>(
        g_esdf_tex,
        d_obstacle_cloud,
        num_points,
        actual_3d,
        actual_2d,
        actual_w2d,
        d_candidates,
        d_costs,
        max_dist_thresh,
        field_min_x,
        field_max_x,
        field_min_y,
        field_max_y
    );

    // CUB ArgMin
    cub::DeviceReduce::ArgMin(
        d_temp_storage, temp_storage_bytes, d_costs, d_out_argmin, num_candidates, stream
    );

    cub::KeyValuePair<int, float> h_argmin;
    cudaMemcpyAsync(
        &h_argmin,
        d_out_argmin,
        sizeof(cub::KeyValuePair<int, float>),
        cudaMemcpyDeviceToHost,
        stream
    );

    cudaStreamSynchronize(stream);

    int best_idx  = h_argmin.key;
    out_best_cost = h_argmin.value;
    out_best_pose = h_candidates[best_idx];

    // Inlier & Component Cost メトリクスの計算
    if (out_inlier_count || out_inlier_cost || out_cost_3d || out_cost_2d) {
        cudaMemsetAsync(d_inlier_count, 0, sizeof(int), stream);
        cudaMemsetAsync(d_inlier_dist_sum, 0, sizeof(float), stream);
        if (out_cost_3d || out_cost_2d) {
            cudaMemsetAsync(d_cost_3d_sum, 0, sizeof(float), stream);
            cudaMemsetAsync(d_cost_2d_sum, 0, sizeof(float), stream);
        }
        int met_threads = 256;
        int met_blocks  = (num_points + met_threads - 1) / met_threads;
        if (met_blocks > 64) met_blocks = 64;
        computeBestPoseMetricsESDFKernel<<<met_blocks, met_threads, 0, stream>>>(
            g_esdf_tex,
            d_obstacle_cloud,
            num_points,
            actual_3d,
            actual_2d,
            out_best_pose,
            inlier_dist_thresh,
            max_dist_thresh,
            field_min_x,
            field_max_x,
            field_min_y,
            field_max_y,
            d_inlier_count,
            d_inlier_dist_sum,
            (out_cost_3d || out_cost_2d) ? d_cost_3d_sum : nullptr,
            (out_cost_3d || out_cost_2d) ? d_cost_2d_sum : nullptr
        );
        int h_count         = 0;
        float h_dist_sum    = 0.0f;
        float h_cost_3d_sum = 0.0f;
        float h_cost_2d_sum = 0.0f;
        cudaMemcpyAsync(&h_count, d_inlier_count, sizeof(int), cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(
            &h_dist_sum, d_inlier_dist_sum, sizeof(float), cudaMemcpyDeviceToHost, stream
        );
        if (out_cost_3d || out_cost_2d) {
            cudaMemcpyAsync(&h_cost_3d_sum, d_cost_3d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(&h_cost_2d_sum, d_cost_2d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream);
        }
        cudaStreamSynchronize(stream);
        if (out_inlier_count) *out_inlier_count = h_count;
        if (out_inlier_cost)
            *out_inlier_cost = (h_count > 0) ? (h_dist_sum / static_cast<float>(h_count)) : 1.0f;
        if (out_cost_3d)
            *out_cost_3d = (actual_3d > 0) ? (h_cost_3d_sum / static_cast<float>(actual_3d)) : 0.0f;
        if (out_cost_2d)
            *out_cost_2d = (actual_2d > 0) ? (h_cost_2d_sum / static_cast<float>(actual_2d)) : 0.0f;
    }

    // 動的点群抽出
    out_dynamic_pts.clear();
    if (extract_dynamic) {
        extractDynamicPointsGPU(
            stream,
            d_obstacle_cloud,
            num_points,
            out_best_pose,
            dynamic_dist_thresh,
            field_min_x,
            field_max_x,
            field_min_y,
            field_max_y,
            out_dynamic_pts
        );
    }

    return true;
}

bool launchESDFDynamicFilterOnly(
    cudaStream_t stream,
    const float* d_cloud,
    int num_points,
    const PoseCandidate& confirmed_pose,
    float dynamic_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    std::vector<float>& out_dynamic_pts,
    const float* tracking_sphere
)
{
    if (num_points <= 0 || !g_esdf_loaded) return false;
    extractDynamicPointsGPU(
        stream,
        d_cloud,
        num_points,
        confirmed_pose,
        dynamic_dist_thresh,
        field_min_x,
        field_max_x,
        field_min_y,
        field_max_y,
        out_dynamic_pts,
        tracking_sphere
    );
    return true;
}

}  // extern "C"
