#include "gn10_pointcloud_localization/cuda/esdf_matcher.cuh"
#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"
#include <cassert>
#include <iostream>
#include <vector>

static void testGroundFilterBoundsCheck(cudaStream_t stream) {
    constexpr int max_points = 8;
    constexpr int num_points = 32;  // 16 ground + 16 obstacle > max_points
    constexpr float sentinel = -999.0f;

    std::vector<float> h_in(num_points * 3, 0.0f);
    std::vector<TimedScanPoint> h_timed(num_points);
    for (int i = 0; i < num_points; ++i) {
        float x = 2.0f;
        float y = 0.0f;
        float z = (i % 2 == 0) ? 0.0f : 0.5f;  // half ground, half obstacle
        h_in[i * 3 + 0] = x;
        h_in[i * 3 + 1] = y;
        h_in[i * 3 + 2] = z;
        h_timed[i] = TimedScanPoint{x, y, z, 1000000000LL + i * 1000LL};
    }

    const float h_tf[12] = {1,0,0,0, 0,1,0,0, 0,0,1,0};
    float *d_in = nullptr, *d_tf = nullptr, *d_ground = nullptr, *d_obstacle = nullptr;
    TimedScanPoint* d_timed = nullptr;
    int *d_g_cnt = nullptr, *d_o_cnt = nullptr;
    assert(cudaMalloc(&d_in, num_points * 3 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_timed, num_points * sizeof(TimedScanPoint)) == cudaSuccess);
    assert(cudaMalloc(&d_tf, 12 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_ground, (max_points + 4) * 3 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_obstacle, (max_points + 4) * 3 * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc(&d_g_cnt, sizeof(int)) == cudaSuccess);
    assert(cudaMalloc(&d_o_cnt, sizeof(int)) == cudaSuccess);

    std::vector<float> init_buf((max_points + 4) * 3, sentinel);
    assert(cudaMemcpy(d_in, h_in.data(), num_points * 3 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_timed, h_timed.data(), num_points * sizeof(TimedScanPoint), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_tf, h_tf, 12 * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);

    int h_g_cnt = 0, h_o_cnt = 0;

    // 1. Standard filterGroundKernel bounds check
    assert(cudaMemcpy(d_ground, init_buf.data(), init_buf.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_obstacle, init_buf.data(), init_buf.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    launchGroundFilter(stream, d_in, d_ground, d_obstacle, d_tf, num_points, max_points,
                       10.0f, 0.5f, 0.0f, 1.0f, 0.1f, d_g_cnt, d_o_cnt, &h_g_cnt, &h_o_cnt);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);

    std::vector<float> out_g(init_buf.size()), out_o(init_buf.size());
    assert(cudaMemcpy(out_g.data(), d_ground, out_g.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(out_o.data(), d_obstacle, out_o.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = max_points * 3; i < static_cast<int>(out_g.size()); ++i) {
        assert(out_g[i] == sentinel);
        assert(out_o[i] == sentinel);
    }

    // 2. Deskew filterGroundWithDeskewKernel bounds check
    assert(cudaMemcpy(d_ground, init_buf.data(), init_buf.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(cudaMemcpy(d_obstacle, init_buf.data(), init_buf.size() * sizeof(float), cudaMemcpyHostToDevice) == cudaSuccess);
    DeskewParams dp{};
    dp.enabled = false;
    launchGroundFilterWithDeskew(stream, d_timed, d_ground, d_obstacle, d_tf, num_points, 1, max_points,
                                 dp, nullptr, 0, 10.0f, 0.5f, 0.0f, 1.0f, 0.1f,
                                 d_g_cnt, d_o_cnt, &h_g_cnt, &h_o_cnt);
    assert(cudaStreamSynchronize(stream) == cudaSuccess);

    assert(cudaMemcpy(out_g.data(), d_ground, out_g.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    assert(cudaMemcpy(out_o.data(), d_obstacle, out_o.size() * sizeof(float), cudaMemcpyDeviceToHost) == cudaSuccess);
    for (int i = max_points * 3; i < static_cast<int>(out_g.size()); ++i) {
        assert(out_g[i] == sentinel);
        assert(out_o[i] == sentinel);
    }

    cudaFree(d_in); cudaFree(d_timed); cudaFree(d_tf);
    cudaFree(d_ground); cudaFree(d_obstacle);
    cudaFree(d_g_cnt); cudaFree(d_o_cnt);
    std::cout << "Ground filter bounds check passed\n";
}
int main() {
    int devices=0;
    if (cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) return 77;
    ESDFHeader grid{0.1f,4,4,4,0,0,0,0.5f};
    std::vector<float> sdf(64,0.5f);
    uploadESDFMapToGPU(grid,sdf.data());
    cudaStream_t stream; assert(cudaStreamCreate(&stream)==cudaSuccess);
    float* points=nullptr; assert(cudaMalloc(&points,4096*3*sizeof(float))==cudaSuccess);
    for (int count : {100,800,32,4096,1,4000}) {
        std::vector<float> input(count*3,0.15f),output;
        assert(cudaMemcpy(points,input.data(),input.size()*sizeof(float),cudaMemcpyHostToDevice)==cudaSuccess);
        PoseCandidate best{}; float cost=0;
        // Exercise matcher dynamic points extraction and capacity growth.
        assert(launchESDFMatcher(stream,points,count,{},0,0,.1f,0,.1f,.5f,.1f,-1,1,-1,1,best,cost,output));
        assert(output.size()==input.size());
        assert(cudaDeviceSynchronize()==cudaSuccess);
    }
    testGroundFilterBoundsCheck(stream);
    cudaFree(points); cudaStreamDestroy(stream); freeESDFMapFromGPU();
    std::cout<<"Matcher dynamic filter and capacity growth passed\n";
}
