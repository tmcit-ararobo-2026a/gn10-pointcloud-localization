#pragma once
#include "gn10_pointcloud_localization/cuda/field_objects.cuh"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>

namespace gn10 {
// Signed point-to-surface residual and outward normal, consistent with the CUDA
// unsigned box/cylinder distance. Movable/visual geometry is never introduced.
inline bool surfaceResidual(const FieldObject& o, double x, double y, double z,
                            double& residual, Eigen::Vector2d& normal) {
    if (o.type==VISUAL_BOX || z<o.z_min-.1 || z>o.z_max+.1) return false;
    const double dx=x-o.center_x,dy=y-o.center_y;
    if (o.type==CYLINDER) {
        const double r=std::hypot(dx,dy);
        if (r<1e-9) return false;
        normal={dx/r,dy/r}; residual=r-o.param1; return true;
    }
    const double ax=std::abs(dx)-o.param1,ay=std::abs(dy)-o.param2;
    if (ax>0 || ay>0) {
        const double nx=std::copysign(std::max(0.0,ax),dx);
        const double ny=std::copysign(std::max(0.0,ay),dy);
        residual=std::hypot(nx,ny); normal={nx/residual,ny/residual};
    } else if (ax>ay) {
        residual=ax;normal={std::copysign(1.0,dx),0};
    } else {
        residual=ay;normal={0,std::copysign(1.0,dy)};
    }
    return true;
}

// Small, bounded Gauss-Newton refinement after the CUDA search. All candidates
// still undergo the CUDA support tests; this does not relax acceptance gates.
inline bool refineSurfaces(const std::vector<float>& cloud,
                           const std::vector<FieldObject>& map, double support_distance,
                           const PoseCandidate& initial, PoseCandidate& result, bool* observable=nullptr) {
    if (observable) *observable=false;
    Eigen::Vector3d pose(initial.x,initial.y,initial.yaw);
    const size_t count=cloud.size()/3, stride=std::max(size_t(1),(count+4095)/4096);
    for (int iteration=0;iteration<3;++iteration) {
        Eigen::Matrix3d H=Eigen::Matrix3d::Zero();Eigen::Vector3d b=Eigen::Vector3d::Zero();
        const double c=std::cos(pose.z()),s=std::sin(pose.z());int supported=0;
        for (size_t i=0;i<count;i+=stride) {
            const double lx=cloud[3*i],ly=cloud[3*i+1],z=cloud[3*i+2];
            const double x=c*lx-s*ly+pose.x(),y=s*lx+c*ly+pose.y();
            double best=support_distance,residual=0;Eigen::Vector2d normal=Eigen::Vector2d::Zero();
            for (const auto& o:map) {
                double r;Eigen::Vector2d n;
                if (surfaceResidual(o,x,y,z,r,n) && std::abs(r)<best) {
                    best=std::abs(r);residual=r;normal=n;
                }
            }
            if (best>=support_distance) continue;
            Eigen::Vector3d J(normal.x(),normal.y(),normal.x()*(-s*lx-c*ly)+normal.y()*(c*lx-s*ly));
            // Normalize yaw to a two-metre lever arm for the observability check.
            J.z()/=2.0;
            const double weight=std::min(1.0,.03/std::max(best,1e-12));
            H.noalias()+=weight*J*J.transpose();b.noalias()+=weight*J*residual;++supported;
        }
        if (supported<30) return false;
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigen(H);
        if (eigen.info()!=Eigen::Success || eigen.eigenvalues().minCoeff()<1e-4*eigen.eigenvalues().maxCoeff())
            return false; // One wall or a rotationally symmetric cluster cannot constrain SE(2).
        if (observable) *observable=true;
        Eigen::Vector3d step=-H.ldlt().solve(b);step.z()/=2.0;
        if (!step.allFinite()) return false;
        const double scale=std::min({1.0,.02/std::max(step.head<2>().norm(),1e-12),.02/std::max(std::abs(step.z()),1e-12)});
        pose+=scale*step;
        if ((pose.head<2>()-Eigen::Vector2d(initial.x,initial.y)).norm()>.04 || std::abs(pose.z()-initial.yaw)>.04)
            return false;
        if (step.norm()<1e-5) break;
    }
    result={static_cast<float>(pose.x()),static_cast<float>(pose.y()),static_cast<float>(pose.z())};return true;
}
} // namespace gn10
