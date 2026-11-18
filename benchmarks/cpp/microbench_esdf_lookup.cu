// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <omp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"
#include "gn10_pointcloud_localization/esdf_map.hpp"
#include "gn10_pointcloud_localization/map_loader.hpp"

namespace
{

constexpr int kMaxBenchObjects = 128;
__constant__ FieldObject c_bench_objs[kMaxBenchObjects];

inline void cudaCheck(cudaError_t err, const char* what)
{
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__device__ inline float evaluateWelschOrL1(
    float distance_m, float max_dist_thresh, int loss_type, float sigma
)
{
    if (distance_m >= max_dist_thresh) {
        return max_dist_thresh;
    }
    if (loss_type == 1 && sigma > 1e-5f) {
        const float ratio = distance_m / sigma;
        return max_dist_thresh * (1.0f - expf(-0.5f * ratio * ratio));
    }
    return distance_m;
}

__device__ inline float queryTex3D(
    cudaTextureObject_t tex, const ESDFHeader& header, float wx, float wy, float wz
)
{
    if (wx < header.min_x || wy < header.min_y || wz < header.min_z) {
        return header.max_dist_thresh;
    }
    const float u = (wx - header.min_x) / header.resolution;
    const float v = (wy - header.min_y) / header.resolution;
    const float w = (wz - header.min_z) / header.resolution;
    if (u >= static_cast<float>(header.size_x) || v >= static_cast<float>(header.size_y) ||
        w >= static_cast<float>(header.size_z)) {
        return header.max_dist_thresh;
    }
    return tex3D<float>(tex, u, v, w);
}

__device__ inline float fetchGlobalClamped(
    const float* __restrict__ grid, const ESDFHeader& header, int ix, int iy, int iz
)
{
    ix = max(0, min(header.size_x - 1, ix));
    iy = max(0, min(header.size_y - 1, iy));
    iz = max(0, min(header.size_z - 1, iz));
    return grid[(static_cast<size_t>(iz) * header.size_y + iy) * header.size_x + ix];
}

__device__ inline float queryGlobalTrilinear(
    const float* __restrict__ grid, const ESDFHeader& header, float wx, float wy, float wz
)
{
    if (wx < header.min_x || wy < header.min_y || wz < header.min_z) {
        return header.max_dist_thresh;
    }
    const float max_x = header.min_x + static_cast<float>(header.size_x) * header.resolution;
    const float max_y = header.min_y + static_cast<float>(header.size_y) * header.resolution;
    const float max_z = header.min_z + static_cast<float>(header.size_z) * header.resolution;
    if (wx >= max_x || wy >= max_y || wz >= max_z) {
        return header.max_dist_thresh;
    }

    const float u = fminf(
        fmaxf((wx - header.min_x) / header.resolution - 0.5f, 0.0f),
        static_cast<float>(header.size_x - 1)
    );
    const float v = fminf(
        fmaxf((wy - header.min_y) / header.resolution - 0.5f, 0.0f),
        static_cast<float>(header.size_y - 1)
    );
    const float w = fminf(
        fmaxf((wz - header.min_z) / header.resolution - 0.5f, 0.0f),
        static_cast<float>(header.size_z - 1)
    );

    const int x0   = static_cast<int>(floorf(u));
    const int y0   = static_cast<int>(floorf(v));
    const int z0   = static_cast<int>(floorf(w));
    const float fx = u - static_cast<float>(x0);
    const float fy = v - static_cast<float>(y0);
    const float fz = w - static_cast<float>(z0);

    const float c000 = fetchGlobalClamped(grid, header, x0, y0, z0);
    const float c100 = fetchGlobalClamped(grid, header, x0 + 1, y0, z0);
    const float c010 = fetchGlobalClamped(grid, header, x0, y0 + 1, z0);
    const float c110 = fetchGlobalClamped(grid, header, x0 + 1, y0 + 1, z0);
    const float c001 = fetchGlobalClamped(grid, header, x0, y0, z0 + 1);
    const float c101 = fetchGlobalClamped(grid, header, x0 + 1, y0, z0 + 1);
    const float c011 = fetchGlobalClamped(grid, header, x0, y0 + 1, z0 + 1);
    const float c111 = fetchGlobalClamped(grid, header, x0 + 1, y0 + 1, z0 + 1);

    const float c00 = c000 + fx * (c100 - c000);
    const float c10 = c010 + fx * (c110 - c010);
    const float c01 = c001 + fx * (c101 - c001);
    const float c11 = c011 + fx * (c111 - c011);
    const float c0  = c00 + fy * (c10 - c00);
    const float c1  = c01 + fy * (c11 - c01);
    return c0 + fz * (c1 - c0);
}

__device__ inline float queryGlobalNearest(
    const float* __restrict__ grid, const ESDFHeader& header, float wx, float wy, float wz
)
{
    if (wx < header.min_x || wy < header.min_y || wz < header.min_z) {
        return header.max_dist_thresh;
    }
    const int ix = static_cast<int>(floorf((wx - header.min_x) / header.resolution));
    const int iy = static_cast<int>(floorf((wy - header.min_y) / header.resolution));
    const int iz = static_cast<int>(floorf((wz - header.min_z) / header.resolution));
    if (ix >= header.size_x || iy >= header.size_y || iz >= header.size_z) {
        return header.max_dist_thresh;
    }
    return grid[(static_cast<size_t>(iz) * header.size_y + iy) * header.size_x + ix];
}

__device__ inline float queryAnalyticFieldSDF(
    int num_objects, float wx, float wy, float wz, float max_dist_thresh
)
{
    float min_d = max_dist_thresh;
    for (int obj_idx = 0; obj_idx < num_objects; ++obj_idx) {
        const FieldObject obj = c_bench_objs[obj_idx];
        if (wz >= (obj.z_min - 0.1f) && wz <= (obj.z_max + 0.1f)) {
            float d = max_dist_thresh;
            if (obj.type == CYLINDER) {
                const float dx = wx - obj.center_x;
                const float dy = wy - obj.center_y;
                d              = fabsf(sqrtf(dx * dx + dy * dy) - obj.param1);
            } else if (obj.type == BOX) {
                const float dx = fabsf(wx - obj.center_x) - obj.param1;
                const float dy = fabsf(wy - obj.center_y) - obj.param2;
                const float ax = fmaxf(dx, 0.0f);
                const float ay = fmaxf(dy, 0.0f);
                d              = fabsf(sqrtf(ax * ax + ay * ay) + fminf(fmaxf(dx, dy), 0.0f));
            }
            if (d < min_d) {
                min_d = d;
            }
        }
    }
    return min_d;
}

enum class LookupBackend : int
{
    kTex3DTrilinear     = 0,
    kGlobalTrilinear    = 1,
    kGlobalNearest      = 2,
    kAnalyticPrimitives = 3,
};

template <LookupBackend kBackend>
__global__ void benchLookupKernel(
    cudaTextureObject_t tex,
    const float* __restrict__ global_grid,
    ESDFHeader header,
    int num_objects,
    const float* __restrict__ cloud,
    int num_points,
    SearchGrid grid,
    float* __restrict__ out_costs,
    float max_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    int loss_type,
    float sigma
)
{
    const int pose_idx = blockIdx.x;
    const int tid      = threadIdx.x;
    __shared__ float s_pose[4];
    __shared__ float s_sum[256];

    if (tid == 0) {
        const PoseCandidate cand = grid.indexToPose(pose_idx);
        s_pose[0]                = cand.x;
        s_pose[1]                = cand.y;
        s_pose[2]                = cosf(cand.yaw);
        s_pose[3]                = sinf(cand.yaw);
    }
    s_sum[tid] = 0.0f;
    __syncthreads();

    const float rx    = s_pose[0];
    const float ry    = s_pose[1];
    const float cos_y = s_pose[2];
    const float sin_y = s_pose[3];

    for (int point_idx = tid; point_idx < num_points; point_idx += blockDim.x) {
        const float lx = cloud[point_idx * 3 + 0];
        const float ly = cloud[point_idx * 3 + 1];
        const float lz = cloud[point_idx * 3 + 2];

        const float wx = cos_y * lx - sin_y * ly + rx;
        const float wy = sin_y * lx + cos_y * ly + ry;
        const float wz = lz;

        float cost_i = max_dist_thresh;
        if (wx >= field_min_x && wx <= field_max_x && wy >= field_min_y && wy <= field_max_y) {
            float d = max_dist_thresh;
            if constexpr (kBackend == LookupBackend::kTex3DTrilinear) {
                d = queryTex3D(tex, header, wx, wy, wz);
            } else if constexpr (kBackend == LookupBackend::kGlobalTrilinear) {
                d = queryGlobalTrilinear(global_grid, header, wx, wy, wz);
            } else if constexpr (kBackend == LookupBackend::kGlobalNearest) {
                d = queryGlobalNearest(global_grid, header, wx, wy, wz);
            } else if constexpr (kBackend == LookupBackend::kAnalyticPrimitives) {
                d = queryAnalyticFieldSDF(num_objects, wx, wy, wz, max_dist_thresh);
            }
            cost_i = evaluateWelschOrL1(d, max_dist_thresh, loss_type, sigma);
        }
        s_sum[tid] += cost_i;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_sum[tid] += s_sum[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        out_costs[pose_idx] = s_sum[0] / static_cast<float>(num_points);
    }
}

float timeGpuBackend(
    LookupBackend backend,
    cudaTextureObject_t tex,
    const float* d_grid,
    const ESDFHeader& header,
    int num_objects,
    const float* d_cloud,
    int num_points,
    const SearchGrid& grid,
    float* d_out_costs,
    float max_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    int loss_type,
    float sigma,
    int warmup,
    int iters
)
{
    const int blocks  = grid.totalCandidates();
    const int threads = 256;

    auto launch_once = [&]() {
        switch (backend) {
            case LookupBackend::kTex3DTrilinear:
                benchLookupKernel<LookupBackend::kTex3DTrilinear><<<blocks, threads>>>(
                    tex,
                    d_grid,
                    header,
                    num_objects,
                    d_cloud,
                    num_points,
                    grid,
                    d_out_costs,
                    max_dist_thresh,
                    field_min_x,
                    field_max_x,
                    field_min_y,
                    field_max_y,
                    loss_type,
                    sigma
                );
                break;
            case LookupBackend::kGlobalTrilinear:
                benchLookupKernel<LookupBackend::kGlobalTrilinear><<<blocks, threads>>>(
                    tex,
                    d_grid,
                    header,
                    num_objects,
                    d_cloud,
                    num_points,
                    grid,
                    d_out_costs,
                    max_dist_thresh,
                    field_min_x,
                    field_max_x,
                    field_min_y,
                    field_max_y,
                    loss_type,
                    sigma
                );
                break;
            case LookupBackend::kGlobalNearest:
                benchLookupKernel<LookupBackend::kGlobalNearest><<<blocks, threads>>>(
                    tex,
                    d_grid,
                    header,
                    num_objects,
                    d_cloud,
                    num_points,
                    grid,
                    d_out_costs,
                    max_dist_thresh,
                    field_min_x,
                    field_max_x,
                    field_min_y,
                    field_max_y,
                    loss_type,
                    sigma
                );
                break;
            case LookupBackend::kAnalyticPrimitives:
                benchLookupKernel<LookupBackend::kAnalyticPrimitives><<<blocks, threads>>>(
                    tex,
                    d_grid,
                    header,
                    num_objects,
                    d_cloud,
                    num_points,
                    grid,
                    d_out_costs,
                    max_dist_thresh,
                    field_min_x,
                    field_max_x,
                    field_min_y,
                    field_max_y,
                    loss_type,
                    sigma
                );
                break;
        }
    };

    for (int i = 0; i < warmup; ++i) {
        launch_once();
    }
    cudaCheck(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t ev_start{};
    cudaEvent_t ev_stop{};
    cudaCheck(cudaEventCreate(&ev_start), "cudaEventCreate start");
    cudaCheck(cudaEventCreate(&ev_stop), "cudaEventCreate stop");

    cudaCheck(cudaEventRecord(ev_start), "cudaEventRecord start");
    for (int i = 0; i < iters; ++i) {
        launch_once();
    }
    cudaCheck(cudaEventRecord(ev_stop), "cudaEventRecord stop");
    cudaCheck(cudaEventSynchronize(ev_stop), "cudaEventSynchronize stop");

    float total_ms = 0.0f;
    cudaCheck(cudaEventElapsedTime(&total_ms, ev_start, ev_stop), "cudaEventElapsedTime");

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);

    return total_ms / static_cast<float>(iters);
}

double timeCpuTrilinear(
    const ESDFMap& esdf_map,
    const std::vector<float>& cloud_xyz,
    int num_points,
    const SearchGrid& grid,
    float max_dist_thresh,
    float field_min_x,
    float field_max_x,
    float field_min_y,
    float field_max_y,
    int loss_type,
    float sigma,
    int cpu_iters
)
{
    const int num_candidates = grid.totalCandidates();
    std::vector<float> host_costs(static_cast<size_t>(num_candidates), 0.0f);

    const auto eval_once = [&]() {
#pragma omp parallel for schedule(static)
        for (int pose_idx = 0; pose_idx < num_candidates; ++pose_idx) {
            const PoseCandidate cand = grid.indexToPose(pose_idx);
            const float cos_y        = std::cos(cand.yaw);
            const float sin_y        = std::sin(cand.yaw);
            float sum                = 0.0f;
            for (int point_idx = 0; point_idx < num_points; ++point_idx) {
                const float lx = cloud_xyz[point_idx * 3 + 0];
                const float ly = cloud_xyz[point_idx * 3 + 1];
                const float lz = cloud_xyz[point_idx * 3 + 2];
                const float wx = cos_y * lx - sin_y * ly + cand.x;
                const float wy = sin_y * lx + cos_y * ly + cand.y;
                float cost_i   = max_dist_thresh;
                if (wx >= field_min_x && wx <= field_max_x && wy >= field_min_y &&
                    wy <= field_max_y) {
                    const float d = esdf_map.getDistance(wx, wy, lz);
                    if (d < max_dist_thresh) {
                        if (loss_type == 1 && sigma > 1e-5f) {
                            const float ratio = d / sigma;
                            cost_i = max_dist_thresh * (1.0f - std::exp(-0.5f * ratio * ratio));
                        } else {
                            cost_i = d;
                        }
                    }
                }
                sum += cost_i;
            }
            host_costs[pose_idx] = sum / static_cast<float>(num_points);
        }
    };

    eval_once();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < cpu_iters; ++i) {
        eval_once();
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    return total_ms / static_cast<double>(cpu_iters);
}

/**
 * @brief Generates deterministic synthetic obstacle surface returns in base_link frame.
 */
std::vector<float> buildSyntheticScan(
    const std::vector<FieldObject>& objects,
    int target_points,
    float sensor_x,
    float sensor_y,
    float sensor_yaw
)
{
    std::vector<float> pts;
    pts.reserve(static_cast<size_t>(target_points) * 3);

    const float cos_inv = std::cos(-sensor_yaw);
    const float sin_inv = std::sin(-sensor_yaw);

    // Sample perimeter points along solid BOX and CYLINDER objects above ground
    std::vector<float> world_pts;
    for (const auto& obj : objects) {
        if (obj.type == VISUAL_BOX || obj.z_max < 0.08f) {
            continue;
        }
        const float z_mid = std::clamp(0.5f * (obj.z_min + obj.z_max), 0.10f, 1.50f);
        if (obj.type == CYLINDER) {
            const int samples = 48;
            for (int k = 0; k < samples; ++k) {
                const float angle = 2.0f * static_cast<float>(M_PI) * k / samples;
                world_pts.push_back(obj.center_x + obj.param1 * std::cos(angle));
                world_pts.push_back(obj.center_y + obj.param1 * std::sin(angle));
                world_pts.push_back(z_mid);
            }
        } else {
            const int edge_samples = 40;
            for (int k = 0; k < edge_samples; ++k) {
                const float u = -1.0f + 2.0f * static_cast<float>(k) / (edge_samples - 1);
                world_pts.push_back(obj.center_x + u * obj.param1);
                world_pts.push_back(obj.center_y + obj.param2);
                world_pts.push_back(z_mid);
                world_pts.push_back(obj.center_x + u * obj.param1);
                world_pts.push_back(obj.center_y - obj.param2);
                world_pts.push_back(z_mid);
            }
        }
    }

    if (world_pts.empty()) {
        throw std::runtime_error("Map produced zero obstacle surface samples for benchmark.");
    }

    const size_t total_world = world_pts.size() / 3;
    for (int i = 0; i < target_points; ++i) {
        const size_t src = (static_cast<size_t>(i) * 7919) % total_world;
        const float dx   = world_pts[src * 3 + 0] - sensor_x;
        const float dy   = world_pts[src * 3 + 1] - sensor_y;
        const float wz   = world_pts[src * 3 + 2];
        const float lx   = cos_inv * dx - sin_inv * dy;
        const float ly   = sin_inv * dx + cos_inv * dy;
        pts.push_back(lx);
        pts.push_back(ly);
        pts.push_back(wz);
    }
    return pts;
}

struct WorkloadSpec
{
    std::string name;
    float range_x;
    float range_y;
    float step_xy;
    float range_yaw;
    float step_yaw;
    int num_points;
    int loss_type;
};

void printUsage(const char* prog)
{
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --map <path>         Path to field JSON map (default: map/nhk2026_map.json)\n"
              << "  --esdf <path>        Optional prebuilt .esdf binary instead of JSON\n"
              << "  --res <float>        ESDF voxel resolution [m] (default: 0.05)\n"
              << "  --max-dist <float>   ESDF truncation ceiling [m] (default: 0.50)\n"
              << "  --iters <int>        GPU timing iterations per workload (default: 30)\n"
              << "  --warmup <int>       GPU warmup iterations (default: 5)\n"
              << "  --include-cpu        Also measure OpenMP CPU trilinear (DLL-style baseline)\n"
              << "  --cpu-iters <int>    CPU timing iterations (default: 3)\n"
              << "  --json-out <path>    Save structured JSON report to file\n";
}

}  // namespace

int main(int argc, char** argv)
{
    std::string map_path = "map/nhk2026_map.json";
    std::string esdf_path;
    float resolution  = 0.05f;
    float max_dist    = 0.50f;
    int gpu_iters     = 30;
    int warmup        = 5;
    bool include_cpu  = false;
    int cpu_iters     = 3;
    std::string json_out;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--map" && i + 1 < argc) {
            map_path = argv[++i];
        } else if (arg == "--esdf" && i + 1 < argc) {
            esdf_path = argv[++i];
        } else if (arg == "--res" && i + 1 < argc) {
            resolution = std::stof(argv[++i]);
        } else if (arg == "--max-dist" && i + 1 < argc) {
            max_dist = std::stof(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            gpu_iters = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--include-cpu") {
            include_cpu = true;
        } else if (arg == "--cpu-iters" && i + 1 < argc) {
            cpu_iters = std::stoi(argv[++i]);
        } else if (arg == "--json-out" && i + 1 < argc) {
            json_out = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    cudaDeviceProp prop{};
    cudaCheck(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");

    ESDFMap esdf_map;
    std::vector<FieldObject> raw_objects;
    std::vector<FieldObject> analytic_objects;

    if (!esdf_path.empty()) {
        if (!esdf_map.loadBinary(esdf_path)) {
            std::cerr << "Failed to load ESDF binary: " << esdf_path << "\n";
            return 1;
        }
    } else {
        raw_objects = MapLoader::loadFromJSON(map_path);
        if (raw_objects.empty()) {
            std::cerr << "Failed to load JSON map: " << map_path << "\n";
            return 1;
        }
        for (const auto& obj : raw_objects) {
            if (obj.type != VISUAL_BOX) {
                analytic_objects.push_back(obj);
            }
        }
        if (analytic_objects.size() > static_cast<size_t>(kMaxBenchObjects)) {
            std::cerr << "Too many objects for analytic constant memory benchmark: "
                      << analytic_objects.size() << "\n";
            return 1;
        }
        if (!esdf_map.buildFromFieldObjects(
                raw_objects, resolution, -5.5f, 5.5f, -6.0f, 6.0f, -0.20f, 4.00f, max_dist
            )) {
            std::cerr << "Failed to build ESDFMap from field objects\n";
            return 1;
        }
    }

    const ESDFHeader& header = esdf_map.header();
    const float field_min_x  = header.min_x;
    const float field_max_x  = header.min_x + static_cast<float>(header.size_x) * header.resolution;
    const float field_min_y  = header.min_y;
    const float field_max_y  = header.min_y + static_cast<float>(header.size_y) * header.resolution;
    const bool has_analytic  = !analytic_objects.empty();

    if (has_analytic) {
        cudaCheck(
            cudaMemcpyToSymbol(
                c_bench_objs,
                analytic_objects.data(),
                analytic_objects.size() * sizeof(FieldObject)
            ),
            "cudaMemcpyToSymbol"
        );
    }

    // Upload ESDF to CUDA 3D array + Texture Object
    cudaChannelFormatDesc chan_desc = cudaCreateChannelDesc<float>();
    cudaExtent extent              = make_cudaExtent(header.size_x, header.size_y, header.size_z);
    cudaArray_t d_array            = nullptr;
    cudaCheck(cudaMalloc3DArray(&d_array, &chan_desc, extent), "cudaMalloc3DArray");

    cudaMemcpy3DParms copy_params{};
    copy_params.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(esdf_map.data().data()),
        static_cast<size_t>(header.size_x) * sizeof(float),
        header.size_x,
        header.size_y
    );
    copy_params.dstArray = d_array;
    copy_params.extent   = extent;
    copy_params.kind     = cudaMemcpyHostToDevice;
    cudaCheck(cudaMemcpy3D(&copy_params), "cudaMemcpy3D");

    cudaResourceDesc res_desc{};
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = d_array;

    cudaTextureDesc tex_desc{};
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.addressMode[2]   = cudaAddressModeClamp;
    tex_desc.filterMode       = cudaFilterModeLinear;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    cudaCheck(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr), "cudaCreateTextureObject");

    // Upload identical ESDF array to linear Global Memory
    const size_t total_voxels = esdf_map.data().size();
    float* d_global_grid      = nullptr;
    cudaCheck(cudaMalloc(&d_global_grid, total_voxels * sizeof(float)), "cudaMalloc d_global_grid");
    cudaCheck(
        cudaMemcpy(
            d_global_grid,
            esdf_map.data().data(),
            total_voxels * sizeof(float),
            cudaMemcpyHostToDevice
        ),
        "cudaMemcpy d_global_grid"
    );

    const std::vector<WorkloadSpec> workloads = {
        {"Local_Fine_5x5x5", 0.010f, 0.010f, 0.005f, 0.0024f, 0.0012f, 2146, 1},
        {"Local_Coarse_Voxelized", 0.30f, 0.30f, 0.05f, 0.15f, 0.03f, 1073, 1},
        {"Local_Coarse_RawDense", 0.30f, 0.30f, 0.05f, 0.15f, 0.03f, 4013, 1},
        {"Global_Search_IMU_Window", 5.25f, 5.70f, 0.30f, 0.40f, 0.30f, 1073, 0},
        {"Global_Search_Full_Yaw", 5.25f, 5.70f, 0.30f, 3.1415f, 0.30f, 1073, 0},
    };

    const std::vector<float> h_cloud = has_analytic
                                           ? buildSyntheticScan(analytic_objects, 4013, -3.5f, 2.0f, 0.4f)
                                           : std::vector<float>(4013 * 3, 1.0f);

    float* d_cloud = nullptr;
    cudaCheck(cudaMalloc(&d_cloud, h_cloud.size() * sizeof(float)), "cudaMalloc d_cloud");
    cudaCheck(
        cudaMemcpy(
            d_cloud, h_cloud.data(), h_cloud.size() * sizeof(float), cudaMemcpyHostToDevice
        ),
        "cudaMemcpy d_cloud"
    );

    int max_candidates = 0;
    for (const auto& w : workloads) {
        const SearchGrid g = createSearchGrid(
            PoseCandidate{0.0f, 0.0f, 0.0f}, w.range_x, w.range_y, w.step_xy, w.range_yaw, w.step_yaw
        );
        max_candidates = std::max(max_candidates, g.totalCandidates());
    }
    float* d_out_costs = nullptr;
    cudaCheck(cudaMalloc(&d_out_costs, static_cast<size_t>(max_candidates) * sizeof(float)), "cudaMalloc d_out_costs");

    std::cout << "Device: " << prop.name << " (sm_" << prop.major << prop.minor
              << ", SMs=" << prop.multiProcessorCount
              << ", VRAM=" << (prop.totalGlobalMem / (1024 * 1024)) << " MiB)\n";
    std::cout << "ESDF Grid: " << header.size_x << "x" << header.size_y << "x" << header.size_z
              << " (res=" << header.resolution << " m, voxels=" << total_voxels
              << ", analytic_objs=" << analytic_objects.size() << ")\n\n";

    nlohmann::json report;
    report["device"] = {
        {"name", prop.name},
        {"compute_capability", std::to_string(prop.major) + "." + std::to_string(prop.minor)},
        {"sm_count", prop.multiProcessorCount},
        {"total_vram_mib", prop.totalGlobalMem / (1024 * 1024)},
    };
    report["grid"] = {
        {"size_x", header.size_x},
        {"size_y", header.size_y},
        {"size_z", header.size_z},
        {"resolution_m", header.resolution},
        {"max_dist_m", header.max_dist_thresh},
        {"analytic_objects", analytic_objects.size()},
    };
    report["workloads"] = nlohmann::json::array();

    std::cout << std::left << std::setw(26) << "Workload" << std::right << std::setw(7) << "K"
              << std::setw(7) << "N" << std::setw(12) << "tex3D[ms]" << std::setw(14) << "Glob8[ms]"
              << std::setw(12) << "Near[ms]" << std::setw(14) << "Analytic[ms]";
    if (include_cpu) {
        std::cout << std::setw(14) << "CPU_OMP[ms]";
    }
    std::cout << "\n";

    for (const auto& w : workloads) {
        const SearchGrid grid = createSearchGrid(
            PoseCandidate{0.0f, 0.0f, 0.0f}, w.range_x, w.range_y, w.step_xy, w.range_yaw, w.step_yaw
        );
        const int num_cand = grid.totalCandidates();
        const float t_tex = timeGpuBackend(
            LookupBackend::kTex3DTrilinear,
            tex,
            d_global_grid,
            header,
            static_cast<int>(analytic_objects.size()),
            d_cloud,
            w.num_points,
            grid,
            d_out_costs,
            0.25f,
            field_min_x,
            field_max_x,
            field_min_y,
            field_max_y,
            w.loss_type,
            0.06f,
            warmup,
            gpu_iters
        );
        const float t_g8 = timeGpuBackend(
            LookupBackend::kGlobalTrilinear,
            tex,
            d_global_grid,
            header,
            static_cast<int>(analytic_objects.size()),
            d_cloud,
            w.num_points,
            grid,
            d_out_costs,
            0.25f,
            field_min_x,
            field_max_x,
            field_min_y,
            field_max_y,
            w.loss_type,
            0.06f,
            warmup,
            gpu_iters
        );
        const float t_near = timeGpuBackend(
            LookupBackend::kGlobalNearest,
            tex,
            d_global_grid,
            header,
            static_cast<int>(analytic_objects.size()),
            d_cloud,
            w.num_points,
            grid,
            d_out_costs,
            0.25f,
            field_min_x,
            field_max_x,
            field_min_y,
            field_max_y,
            w.loss_type,
            0.06f,
            warmup,
            gpu_iters
        );
        const float t_ana =
            has_analytic
                ? timeGpuBackend(
                      LookupBackend::kAnalyticPrimitives,
                      tex,
                      d_global_grid,
                      header,
                      static_cast<int>(analytic_objects.size()),
                      d_cloud,
                      w.num_points,
                      grid,
                      d_out_costs,
                      0.25f,
                      field_min_x,
                      field_max_x,
                      field_min_y,
                      field_max_y,
                      w.loss_type,
                      0.06f,
                      warmup,
                      gpu_iters
                  )
                : -1.0f;

        double t_cpu = -1.0;
        if (include_cpu) {
            t_cpu = timeCpuTrilinear(
                esdf_map,
                h_cloud,
                w.num_points,
                grid,
                0.25f,
                field_min_x,
                field_max_x,
                field_min_y,
                field_max_y,
                w.loss_type,
                0.06f,
                cpu_iters
            );
        }

        std::cout << std::left << std::setw(26) << w.name << std::right << std::setw(7) << num_cand
                  << std::setw(7) << w.num_points << std::fixed << std::setprecision(4)
                  << std::setw(12) << t_tex << std::setw(14) << t_g8 << std::setw(12) << t_near
                  << std::setw(14) << t_ana;
        if (include_cpu) {
            std::cout << std::setw(14) << t_cpu;
        }
        std::cout << "\n";

        nlohmann::json item = {
            {"workload", w.name},
            {"candidates", num_cand},
            {"points", w.num_points},
            {"tex3d_trilinear_ms", t_tex},
            {"global_8fetch_trilinear_ms", t_g8},
            {"global_nearest_ms", t_near},
            {"analytic_fieldsdf_ms", t_ana},
            {"speedup_global8_over_tex3d", t_g8 / std::max(1e-6f, t_tex)},
            {"speedup_analytic_over_tex3d", (t_ana > 0.0f) ? (t_ana / std::max(1e-6f, t_tex)) : -1.0f},
        };
        if (include_cpu) {
            item["cpu_omp_trilinear_ms"]   = t_cpu;
            item["cpu_omp_threads"]        = omp_get_max_threads();
            item["speedup_cpu_over_tex3d"] = t_cpu / std::max(1e-6f, t_tex);
        }
        report["workloads"].push_back(item);
    }

    if (!json_out.empty()) {
        std::ofstream ofs(json_out);
        if (!ofs.is_open()) {
            std::cerr << "Failed to write JSON report: " << json_out << "\n";
            return 1;
        }
        ofs << report.dump(2) << "\n";
        std::cout << "\nSaved microbenchmark JSON: " << json_out << "\n";
    }

    cudaFree(d_out_costs);
    cudaFree(d_cloud);
    cudaFree(d_global_grid);
    cudaDestroyTextureObject(tex);
    cudaFreeArray(d_array);
    return 0;
}
