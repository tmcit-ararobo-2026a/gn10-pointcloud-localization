#include "gn10_pointcloud_localization/odom_projection.hpp"

#include <cmath>

namespace gn10 {

Pose2d integrateBodyMotion(
    Pose2d previous_planar,
    const Eigen::Isometry3d& previous_odom_base,
    const Eigen::Isometry3d& current_odom_base
)
{
    const Eigen::Isometry3d delta = previous_odom_base.inverse() * current_odom_base;
    const double c = std::cos(previous_planar.yaw);
    const double s = std::sin(previous_planar.yaw);
    const double dx = delta.translation().x();
    const double dy = delta.translation().y();
    return {
        previous_planar.x + c * dx - s * dy,
        previous_planar.y + s * dx + c * dy,
        wrapYaw(previous_planar.yaw +
                std::atan2(delta.linear()(1, 0), delta.linear()(0, 0)))
    };
}

Eigen::Isometry3d reconstructMapBase(
    Pose2d fused,
    const Eigen::Isometry3d& anchor_odom_base,
    double anchor_map_yaw,
    const Eigen::Isometry3d& current_odom_base
)
{
    const Eigen::Isometry3d relative = anchor_odom_base.inverse() * current_odom_base;
    const Eigen::Matrix3d anchored_rotation =
        Eigen::AngleAxisd(anchor_map_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
        relative.linear();
    const double anchored_yaw =
        std::atan2(anchored_rotation(1, 0), anchored_rotation(0, 0));
    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.linear() =
        Eigen::AngleAxisd(wrapYaw(fused.yaw - anchored_yaw), Eigen::Vector3d::UnitZ())
            .toRotationMatrix() * anchored_rotation;
    result.translation() = Eigen::Vector3d(
        fused.x, fused.y, relative.translation().z()
    );
    return result;
}

}  // namespace gn10
