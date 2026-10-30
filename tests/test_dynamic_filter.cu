#include "gn10_pointcloud_localization/cuda/esdf_matcher.cuh"
#include <cassert>
#include <iostream>
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
        // Exercise BOTH the existing matcher and new filter, including capacity growth.
        assert(launchESDFMatcher(stream,points,count,{},0,0,.1f,0,.1f,.5f,.1f,-1,1,-1,1,best,cost,output));
        assert(output.size()==input.size());
        assert(launchESDFDynamicFilterOnly(stream,points,count,{},.1f,-1,1,-1,1,output));
        assert(output==input);
        assert(launchESDFDynamicFilterOnly(stream,points,count,{},.6f,-1,1,-1,1,output));
        assert(output.empty());
        float sphere[4]{.15f,.15f,.15f,.02f};
        assert(launchESDFDynamicFilterOnly(stream,points,count,{},.6f,-1,1,-1,1,output,sphere));
        assert(output==input); // Prediction protection preserves a cloth close to mapped geometry.
        assert(cudaDeviceSynchronize()==cudaSuccess);
    }
    cudaFree(points); cudaStreamDestroy(stream); freeESDFMapFromGPU();
    std::cout<<"Matcher/filter capacity growth, small clouds, all-selected and empty results passed\n";
}
