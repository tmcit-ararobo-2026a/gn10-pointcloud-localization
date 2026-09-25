#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition)
{
    if (!condition) throw std::runtime_error("field matcher check failed");
}

void checkCuda(cudaError_t result)
{
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

struct DeviceCloud {
    explicit DeviceCloud(const std::vector<float>& points)
    {
        checkCuda(cudaMalloc(&data, points.size() * sizeof(float)));
        checkCuda(cudaMemcpy(
            data, points.data(), points.size() * sizeof(float), cudaMemcpyHostToDevice
        ));
    }
    ~DeviceCloud() { cudaFree(data); }
    float* data{nullptr};
};

float matchCost(
    cudaStream_t stream, const std::vector<float>& points, float range_x,
    PoseCandidate& pose
)
{
    DeviceCloud cloud(points);
    float cost = 1.0f;
    std::vector<float> dynamic;
    require(launchFieldSDFMatcher(
        stream, cloud.data, static_cast<int>(points.size() / 3),
        {0.0f, 0.0f, 0.0f}, range_x, 0.0f, 0.2f, 0.0f, 0.1f,
        0.2f, 0.15f, -1.0f, 1.0f, -1.0f, 1.0f,
        pose, cost, dynamic, false
    ));
    return cost;
}
}  // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return 77;
    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream));
    uploadFieldMapToGPU({{BOX, 0.0f, 0.0f, -0.1f, 0.1f, 0.5f, 0.5f}});

    PoseCandidate pose{};
    const float surface_cost = matchCost(
        stream,
        {-0.5f, 0.0f, 0.0f, 0.5f, 0.0f, 0.0f,
         0.0f, -0.5f, 0.0f, 0.0f, 0.5f, 0.0f},
        0.2f, pose
    );
    require(std::abs(pose.x) < 1e-4f);
    require(surface_cost < 1e-4f);

    const float outside_cost = matchCost(
        stream, {0.5f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f}, 0.0f, pose
    );
    require(std::abs(outside_cost - 0.1f) < 1e-4f);
    checkCuda(cudaStreamDestroy(stream));
    return 0;
}
