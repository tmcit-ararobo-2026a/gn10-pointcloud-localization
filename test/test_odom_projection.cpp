#include "gn10_pointcloud_localization/odom_projection.hpp"

#include <cmath>
#include <stdexcept>

namespace {
void require(bool condition) { if (!condition) throw std::runtime_error("odom projection failed"); }
bool near(double a, double b, double tolerance = 1e-9)
{
    return std::abs(a - b) < tolerance;
}
}  // namespace

int main()
{
    Eigen::Isometry3d first = Eigen::Isometry3d::Identity();
    first.linear() = (
        Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(-0.25, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(3.13, Eigen::Vector3d::UnitX())
    ).toRotationMatrix();
    first.translation() = Eigen::Vector3d(4.0, -3.0, 2.0);

    Eigen::Isometry3d local_step = Eigen::Isometry3d::Identity();
    local_step.linear() = Eigen::AngleAxisd(0.15, Eigen::Vector3d::UnitZ())
                              .toRotationMatrix();
    local_step.translation() = Eigen::Vector3d(1.0, 0.2, 0.05);
    const Eigen::Isometry3d second = first * local_step;

    const auto planar = gn10::integrateBodyMotion({}, first, second);
    require(near(planar.x, 1.0));
    require(near(planar.y, 0.2));
    require(near(planar.yaw, 0.15));

    const auto anchored = gn10::reconstructMapBase({5.0, 2.0, 1.0}, first, 1.0, first);
    require(near(anchored.translation().z(), 0.0));
    require(near(std::atan2(anchored.linear()(1, 0), anchored.linear()(0, 0)), 1.0));

    const auto fused = gn10::reconstructMapBase({5.3, 2.4, 1.2}, first, 1.0, second);
    require(near(fused.translation().x(), 5.3));
    require(near(fused.translation().y(), 2.4));
    require(near(fused.translation().z(), 0.05));
    require(near(std::atan2(fused.linear()(1, 0), fused.linear()(0, 0)), 1.2));
    require(near(fused.linear().determinant(), 1.0));
    return 0;
}
