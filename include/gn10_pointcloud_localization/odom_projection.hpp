#pragma once

#include "gn10_pointcloud_localization/pose_fusion_filter.hpp"

#include <Eigen/Geometry>

namespace gn10 {

// Integrate the full SE(3) odometry increment in the previous base frame.
// The result is a level SE(2) odometry track for the map-matching filter.
Pose2d integrateBodyMotion(
    Pose2d previous_planar,
    const Eigen::Isometry3d& previous_odom_base,
    const Eigen::Isometry3d& current_odom_base
);

// Anchor FAST-LIO's arbitrary 3D world to a level map pose at the first match.
// Subsequent x, y and yaw come from the fusion filter; relative height and tilt
// come from FAST-LIO, so the map TF can transform the raw LiDAR cloud correctly.
Eigen::Isometry3d reconstructMapBase(
    Pose2d fused,
    const Eigen::Isometry3d& anchor_odom_base,
    double anchor_map_yaw,
    const Eigen::Isometry3d& current_odom_base
);

}  // namespace gn10
