#pragma once
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
namespace gn10 {
struct FloorPlane {float a{0},b{0},c{0},rms{0};int support{0};bool valid{false};};
// Constrained near-horizontal RANSAC. Estimate classification geometry only;
// keep the original base coordinates and sensor extrinsic transform unchanged.
inline FloorPlane estimateFloor(const std::vector<float>& cloud,const float T[12],double max_range=8.0) {
    std::vector<Eigen::Vector3d> points;points.reserve(1024);
    const size_t stride=std::max(size_t(1),(cloud.size()/3+1023)/1024);
    for(size_t i=0;i<cloud.size()/3;i+=stride) {
        Eigen::Vector3d p;
        for(int r=0;r<3;++r)p[r]=T[r*4]*cloud[i*3]+T[r*4+1]*cloud[i*3+1]+T[r*4+2]*cloud[i*3+2]+T[r*4+3];
        const double radius=p.head<2>().norm();
        if(p.allFinite() && radius>1 && radius<max_range && p.z()>-.15 && p.z()<.20)points.push_back(p);
    }
    if(points.size()<100)return {};
    Eigen::Vector3d best=Eigen::Vector3d::Zero();int best_count=0;
    uint32_t state=2654435761U;
    auto sample=[&](){state^=state<<13;state^=state>>17;state^=state<<5;return state%points.size();};
    auto plausible=[](const Eigen::Vector3d& p){return p.allFinite() && p.head<2>().norm()<=.05 && std::abs(p.z())<=.10;};
    for(int iteration=0;iteration<48;++iteration) {
        const auto& p=points[sample()];const auto d1=points[sample()]-p,d2=points[sample()]-p;
        const double determinant=d1.x()*d2.y()-d2.x()*d1.y();if(std::abs(determinant)<.2)continue;
        const double a=(d1.z()*d2.y()-d2.z()*d1.y())/determinant;
        const double b=(d1.x()*d2.z()-d2.x()*d1.z())/determinant;
        Eigen::Vector3d model(a,b,p.z()-a*p.x()-b*p.y());if(!plausible(model))continue;
        int count=0;for(const auto& point:points)count+=std::abs(point.z()-model.x()*point.x()-model.y()*point.y()-model.z())<.02;
        if(count>best_count){best_count=count;best=model;}
    }
    if(best_count<100 || best_count<.65*points.size())return {};
    Eigen::Matrix3d H=Eigen::Matrix3d::Zero();Eigen::Vector3d rhs=Eigen::Vector3d::Zero();
    Eigen::Vector2d minimum=Eigen::Vector2d::Constant(1e9),maximum=-minimum;
    for(const auto& p:points) {
        if(std::abs(p.z()-best.x()*p.x()-best.y()*p.y()-best.z())>=.02)continue;
        Eigen::Vector3d v(p.x(),p.y(),1);H.noalias()+=v*v.transpose();rhs.noalias()+=v*p.z();
        minimum=minimum.cwiseMin(p.head<2>());maximum=maximum.cwiseMax(p.head<2>());
    }
    if((maximum-minimum).minCoeff()<1.5)return {};
    const Eigen::Vector2d centre=H.topRightCorner<2,1>()/H(2,2);
    const Eigen::Matrix2d coverage=H.topLeftCorner<2,2>()/H(2,2)-centre*centre.transpose();
    if(Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>(coverage).eigenvalues().minCoeff()<.25)return {};
    const auto solver=H.ldlt();if(solver.info()!=Eigen::Success || !solver.isPositive())return {};
    const Eigen::Vector3d model=solver.solve(rhs);if(!plausible(model))return {};
    double sum=0;int count=0;
    for(const auto& p:points){const double residual=p.z()-model.x()*p.x()-model.y()*p.y()-model.z();if(std::abs(residual)<.02){sum+=residual*residual;++count;}}
    if(count<100 || count<.65*points.size())return {};
    const double rms=std::sqrt(sum/count);if(rms>.012)return {};
    return {float(model.x()),float(model.y()),float(model.z()),float(rms),count,true};
}
} // namespace gn10
