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
// x, y and yaw come from fusion. With constrain_to_floor, z/roll/pitch are zero;
// otherwise FAST-LIO relative height and tilt are retained without map correction.
Eigen::Isometry3d reconstructMapBase(
    Pose2d fused,
    const Eigen::Isometry3d& anchor_odom_base,
    double anchor_map_yaw,
    const Eigen::Isometry3d& current_odom_base,
    bool constrain_to_floor = false
);

}  // namespace gn10
