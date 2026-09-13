#include <cuda_runtime.h>

#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cub/cub.cuh>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

__constant__ FieldObject c_cylinders[128];
__constant__ FieldObject c_boxes[128];

static int g_num_cylinders = 0;
static int g_num_boxes     = 0;

static PoseCandidate* d_candidates = nullptr;
static float* d_costs              = nullptr;
static int g_max_candidates        = 0;

static uint8_t* d_is_dynamic = nullptr;
static int g_max_dynamic_pts = 0;

static cub::KeyValuePair<int, float>* d_out_argmin = nullptr;
static void* d_temp_storage                        = nullptr;
static size_t temp_storage_bytes                   = 0;

__device__ __forceinline__ float distToBox2D(
    float px, float py, float cx, float cy, float half_w, float half_d
)
{
    float dx = fabsf(px - cx) - half_w;
    float dy = fabsf(py - cy) - half_d;
    float ax = fmaxf(dx, 0.0f);
    float ay = fmaxf(dy, 0.0f);
    return sqrtf(ax * ax + ay * ay) + fminf(fmaxf(dx, dy), 0.0f);
}

__device__ __forceinline__ float distToCylinder2D(
    float px, float py, float cx, float cy, float radius
)
{
    float dx          = px - cx;
    float dy          = py - cy;
    float dist_center = sqrtf(dx * dx + dy * dy);
    return fabsf(dist_center - radius);
}

// 全姿勢候補の残差計算カーネル
__global__ void evaluateFieldSDFKernel(
    const float* __restrict__ cloud,
    int num_points,
    int num_cylinders,
    int num_boxes,
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

    __shared__ float s_cost[256];
    __shared__ int s_valid_count[256];
    s_cost[tid]        = 0.0f;
    s_valid_count[tid] = 0;

    for (int i = tid; i < num_points; i += blockDim.x) {
        float lx = cloud[i * 3 + 0];
        float ly = cloud[i * 3 + 1];
        float lz = cloud[i * 3 + 2];

        // map 座標系へ変換
        float wx = cos_y * lx - sin_y * ly + rx;
        float wy = sin_y * lx + cos_y * ly + ry;
        float wz = lz;

        if (wx < field_min_x || wx > field_max_x || wy < field_min_y || wy > field_max_y) {
            continue;
        }

        s_valid_count[tid] += 1;
        float min_d = max_dist_thresh;

        // --- CYLINDER の判定ループ ---
        for (int o = 0; o < num_cylinders; ++o) {
            const FieldObject obj = c_cylinders[o];
            if (wz >= (obj.z_min - 0.1f) && wz <= (obj.z_max + 0.1f)) {
                float d = distToCylinder2D(wx, wy, obj.center_x, obj.center_y, obj.param1);
                if (d < min_d) min_d = d;
            }
        }

        // --- BOX の判定ループ ---
        for (int o = 0; o < num_boxes; ++o) {
            const FieldObject obj = c_boxes[o];
            if (wz >= (obj.z_min - 0.1f) && wz <= (obj.z_max + 0.1f)) {
                float d = distToBox2D(wx, wy, obj.center_x, obj.center_y, obj.param1, obj.param2);
                if (d < min_d) min_d = d;
            }
        }

        s_cost[tid] += (min_d < max_dist_thresh) ? min_d : max_dist_thresh;
    }
    __syncthreads();

    // Block Reduce
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_cost[tid] += s_cost[tid + s];
            s_valid_count[tid] += s_valid_count[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        int total_valid = s_valid_count[0];
        out_costs[pose_idx] =
            (total_valid > 0) ? (s_cost[0] / static_cast<float>(total_valid)) : max_dist_thresh;
    }
}

__global__ void filterDynamicPointsKernel(
    const float* __restrict__ cloud,
    int num_points,
    int num_cylinders,
    int num_boxes,
    PoseCandidate best_pose,
    float dynamic_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
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

    // map 座標系へ変換
    float wx = cos_y * lx - sin_y * ly + rx;
    float wy = sin_y * lx + cos_y * ly + ry;
    float wz = lz;

    if (wx < field_min_x || wx > field_max_x || wy < field_min_y || wy > field_max_y) {
        out_is_dynamic[i] = 0;
        return;
    }

    float min_d = dynamic_dist_thresh;

    // --- CYLINDER の判定ループ ---
    for (int o = 0; o < num_cylinders; ++o) {
        const FieldObject obj = c_cylinders[o];
        if (wz >= (obj.z_min - 0.1f) && wz <= (obj.z_max + 0.1f)) {
            float d = distToCylinder2D(wx, wy, obj.center_x, obj.center_y, obj.param1);
            if (d < min_d) min_d = d;
        }
    }

    // --- BOX の判定ループ ---
    for (int o = 0; o < num_boxes; ++o) {
        const FieldObject obj = c_boxes[o];
        if (wz >= (obj.z_min - 0.1f) && wz <= (obj.z_max + 0.1f)) {
            float d = distToBox2D(wx, wy, obj.center_x, obj.center_y, obj.param1, obj.param2);
            if (d < min_d) min_d = d;
        }
    }

    out_is_dynamic[i] = (min_d >= dynamic_dist_thresh) ? 1 : 0;
}

extern "C" {

void uploadFieldMapToGPU(const std::vector<FieldObject>& host_map)
{
    std::vector<FieldObject> cylinders;
    std::vector<FieldObject> boxes;

    for (const auto& obj : host_map) {
        if (obj.type == CYLINDER) {
            cylinders.push_back(obj);
        } else if (obj.type == BOX) {
            boxes.push_back(obj);
        }
    }

    g_num_cylinders = static_cast<int>(cylinders.size());
    g_num_boxes     = static_cast<int>(boxes.size());

    if (g_num_cylinders > 128) g_num_cylinders = 128;
    if (g_num_boxes > 128) g_num_boxes = 128;

    if (g_num_cylinders > 0) {
        cudaMemcpyToSymbol(c_cylinders, cylinders.data(), g_num_cylinders * sizeof(FieldObject));
    }
    if (g_num_boxes > 0) {
        cudaMemcpyToSymbol(c_boxes, boxes.data(), g_num_boxes * sizeof(FieldObject));
    }
}

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
    bool extract_dynamic
)
{
    if (num_points <= 0 || (g_num_cylinders == 0 && g_num_boxes == 0)) return false;

    // 姿勢候補の生成 (Host)
    std::vector<PoseCandidate> h_candidates;
    for (float dx = -range_xy; dx <= range_xy + 1e-5f; dx += step_xy) {
        for (float dy = -range_xy; dy <= range_xy + 1e-5f; dy += step_xy) {
            for (float dyaw = -range_yaw; dyaw <= range_yaw + 1e-5f; dyaw += step_yaw) {
                h_candidates.push_back({base_pose.x + dx, base_pose.y + dy, base_pose.yaw + dyaw});
            }
        }
    }

    int num_candidates = static_cast<int>(h_candidates.size());
    if (num_candidates == 0) return false;

    // メモリ確保・再確保チェック
    if (num_candidates > g_max_candidates) {
        if (d_candidates) cudaFree(d_candidates);
        if (d_costs) cudaFree(d_costs);
        if (d_out_argmin) cudaFree(d_out_argmin);

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

    if (num_points > g_max_dynamic_pts) {
        if (d_is_dynamic) cudaFree(d_is_dynamic);
        g_max_dynamic_pts = num_points * 2;
        cudaMalloc(&d_is_dynamic, g_max_dynamic_pts * sizeof(uint8_t));
    }

    // H2D 転送
    cudaMemcpyAsync(
        d_candidates,
        h_candidates.data(),
        num_candidates * sizeof(PoseCandidate),
        cudaMemcpyHostToDevice,
        stream
    );

    // 全姿勢候補のSDF評価 (GPU)
    int sdf_threads = 256;
    int sdf_blocks  = num_candidates;
    evaluateFieldSDFKernel<<<sdf_blocks, sdf_threads, 0, stream>>>(
        d_obstacle_cloud,
        num_points,
        g_num_cylinders,
        g_num_boxes,
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

    // 動的点群の抽出
    out_dynamic_pts.clear();

    if (extract_dynamic) {
        int dyn_threads = 256;
        int dyn_blocks  = (num_points + dyn_threads - 1) / dyn_threads;
        filterDynamicPointsKernel<<<dyn_blocks, dyn_threads, 0, stream>>>(
            d_obstacle_cloud,
            num_points,
            g_num_cylinders,
            g_num_boxes,
            out_best_pose,
            dynamic_dist_thresh,
            field_min_x,
            field_max_x,
            field_min_y,
            field_max_y,
            d_is_dynamic
        );

        std::vector<uint8_t> h_is_dynamic(num_points);
        std::vector<float> h_raw_cloud(num_points * 3);

        cudaMemcpyAsync(
            h_is_dynamic.data(),
            d_is_dynamic,
            num_points * sizeof(uint8_t),
            cudaMemcpyDeviceToHost,
            stream
        );
        cudaMemcpyAsync(
            h_raw_cloud.data(),
            d_obstacle_cloud,
            num_points * 3 * sizeof(float),
            cudaMemcpyDeviceToHost,
            stream
        );

        cudaStreamSynchronize(stream);

        out_dynamic_pts.reserve(num_points * 3);
        for (int i = 0; i < num_points; ++i) {
            if (h_is_dynamic[i]) {
                out_dynamic_pts.push_back(h_raw_cloud[i * 3 + 0]);
                out_dynamic_pts.push_back(h_raw_cloud[i * 3 + 1]);
                out_dynamic_pts.push_back(h_raw_cloud[i * 3 + 2]);
            }
        }
    }

    return true;
}

}  // extern "C"