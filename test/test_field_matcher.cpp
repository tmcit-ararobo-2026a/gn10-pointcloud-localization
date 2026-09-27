#include "gn10_pointcloud_localization/cuda/field_objects.cuh"
#include "gn10_pointcloud_localization/cuda/ground_filter.cuh"

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

    const float top_cost=matchCost(stream,{0,0,.1f,.2f,.2f,.1f},0,pose);
    require(top_cost<1e-6f); // Exact horizontal top returns belong to the static map.
    DeviceCloud top_only({.1f,.1f,.1f,-.1f,.1f,.1f,-.1f,-.1f,.1f,.1f,-.1f,.1f});
    FieldMatchStats top_quality;float top_robust;std::vector<float> top_dynamic;
    require(!launchFieldSDFMatcher(stream,top_only.data,4,{0,0,0},0,0,.1,0,.1,
        .2,.15,-1,1,-1,1,pose,top_robust,top_dynamic,false,.08,.1,1,3,&top_quality,1));
    require(top_quality.support_count==0&&top_quality.axis_x==0&&top_quality.axis_y==0); // A tabletop alone cannot localize XY/yaw.

    const float outside_cost = matchCost(
        stream, {0.5f, 0.0f, 0.0f, 3.0f, 0.0f, 0.0f}, 0.0f, pose
    );
    require(std::abs(outside_cost - 0.1f) < 1e-4f);

    DeviceCloud shifted({
        -0.517f, 0.0f, 0.0f, 0.483f, 0.0f, 0.0f,
        -0.017f, -0.5f, 0.0f, -0.017f, 0.5f, 0.0f
    });
    float coarse_cost = 1.0f;
    float fine_cost = 1.0f;
    PoseCandidate coarse_pose{};
    PoseCandidate fine_pose{};
    std::vector<float> dynamic;
    require(launchFieldSDFMatcher(
        stream, shifted.data, 4, {0.0f, 0.0f, 0.0f},
        0.05f, 0.0f, 0.05f, 0.0f, 0.01f,
        0.2f, 0.15f, -1.0f, 1.0f, -1.0f, 1.0f,
        coarse_pose, coarse_cost, dynamic, false
    ));
    require(launchFieldSDFMatcher(
        stream, shifted.data, 4, coarse_pose,
        0.02f, 0.0f, 0.01f, 0.0f, 0.01f,
        0.2f, 0.15f, -1.0f, 1.0f, -1.0f, 1.0f,
        fine_pose, fine_cost, dynamic, false
    ));
    require(std::abs(coarse_pose.x) < 1e-4f);
    require(std::abs(fine_pose.x - 0.02f) < 1e-4f);
    require(fine_cost < coarse_cost - 0.005f);
    // Dense moving objects must not raise the accepted map-support residual.
    std::vector<float> occluded;
    for (int i=0;i<20;++i) {
        const float y=-0.2f+i*0.02f;
        occluded.insert(occluded.end(),{-0.5f,y,0.0f,0.5f,y,0.0f,
            y,-0.5f,0.0f,y,0.5f,0.0f});
    }
    for(int i=0;i<400;++i) occluded.insert(occluded.end(),{0.0f,0.0f,0.0f});
    DeviceCloud clutter(occluded);
    FieldMatchStats quality;
    float robust_cost;
    require(launchFieldSDFMatcher(stream,clutter.data,480,{0,0,0},
        0.2f,0.0f,0.02f,0.0f,0.01f,0.2f,0.15f,-1,1,-1,1,
        pose,robust_cost,dynamic,false,0.08f,0.1f,50,3,&quality,10));
    require(std::abs(pose.x)<1e-4f && robust_cost<1e-4f);
    require(quality.support_count==80 && quality.sectors>=3);
    // A single visible surface cannot validate the whole pose.
    std::vector<float> cluster;
    for(int i=0;i<100;++i)cluster.insert(cluster.end(),{0.5f,0.0f,0.0f});
    DeviceCloud single(cluster);
    require(!launchFieldSDFMatcher(stream,single.data,100,{0,0,0},
        0,0,0.1f,0,0.1f,0.2f,0.15f,-1,1,-1,1,
        pose,robust_cost,dynamic,false,0.08f,0.1f,50,3));
    // No mapped surface evidence: never accept a low inlier-only cost.
    DeviceCloud outside({3,3,0,4,4,0});
    require(!launchFieldSDFMatcher(stream,outside.data,2,{0,0,0},
        0,0,0.1f,0,0.1f,0.2f,0.15f,-1,1,-1,1,
        pose,robust_cost,dynamic,false,0.08f,0.1f,1,1));
    // Many directions on one wall still cannot constrain motion along that wall.
    std::vector<float> wall;
    for(int i=0;i<100;++i)wall.insert(wall.end(),{-0.4f+i*0.008f,0.5f,0.0f});
    DeviceCloud one_wall(wall);
    require(!launchFieldSDFMatcher(stream,one_wall.data,100,{0,0,0},
        0,0,0.1f,0,0.1f,0.2f,0.15f,-1,1,-1,1,
        pose,robust_cost,dynamic,false,0.08f,0.1f,50,2,nullptr,10));
    // A real tilted floor above z=4cm is removed by plane distance, while
    // output coordinates stay unchanged and a point 5.5cm above it remains.
    DeviceCloud floor_returns({0,-5,.045f,0,-5,.1f,0,0,.1f});
    DeviceCloud floor_output(std::vector<float>(9)),obstacle_output(std::vector<float>(9));
    DeviceCloud transform({1,0,0,0,0,1,0,0,0,0,1,0});
    int *ground_count,*obstacle_count;checkCuda(cudaMalloc(&ground_count,sizeof(int)));
    checkCuda(cudaMalloc(&obstacle_count,sizeof(int)));int ground_n=0,obstacle_n=0;
    launchGroundFilter(stream,floor_returns.data,floor_output.data,obstacle_output.data,transform.data,
        3,12,.8,0,1.2,.04,ground_count,obstacle_count,&ground_n,&obstacle_n,0,-.007,.01);
    checkCuda(cudaStreamSynchronize(stream));require(ground_n==1&&obstacle_n==1);
    float ground_xyz[3];checkCuda(cudaMemcpy(ground_xyz,floor_output.data,sizeof(ground_xyz),cudaMemcpyDeviceToHost));
    require(std::abs(ground_xyz[1]+5)<1e-6&&std::abs(ground_xyz[2]-.045)<1e-6);
    checkCuda(cudaFree(ground_count));checkCuda(cudaFree(obstacle_count));
    checkCuda(cudaStreamDestroy(stream));
    return 0;
}
