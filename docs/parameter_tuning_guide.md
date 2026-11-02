# GN10 Point Cloud Localization Parameter Tuning Guide

This document explains the architecture, processing stages, parameter effects, and practical tuning procedures for `gn10-pointcloud-localization`.

---

## 1. Architecture & Flow Overview

The localization pipeline consists of sensor ingestion, deskew preprocessing, motion prediction, ESDF grid evaluation with independent residual weighting, and global recovery.

The PlantUML diagrams in this directory visualize the complete dataflow:
- [pipeline_flow_and_parameters.pu](pipeline_flow_and_parameters.pu): Full pipeline dataflow from sensors to publishers, mapped to the parameters controlling each stage.
- [matching_cost_and_weighting.pu](matching_cost_and_weighting.pu): Mathematical process of the 3D/2D independent residual cost and inlier convergence check.

---

## 2. Parameter Category Reference

### 2.1 Scan Accumulation (`scan_accumulation.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `scan_accumulation.window_s` | `0.10` | double | Livox LiDAR packet accumulation time window [s]. For Livox MID360 (10Hz / 100ms full rotation), set to `0.10` to get a dense scan. |
| `scan_accumulation.timestamp_field` | `"timestamp"` | string | PointCloud2 field name for per-point timestamps (Livox driver uses `"timestamp"`). |
| `scan_accumulation.timestamp_scale` | `1.0e-9` | double | Multiplier to convert timestamp field values into seconds (e.g. `1e-9` for nanoseconds). |
| `scan_accumulation.timestamp_relative` | `false` | bool | Whether point timestamps are relative offsets from `header.stamp` or absolute nanoseconds. |

---

### 2.2 Point Cloud Filtering (`filter_params.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `filter_params.ground_z_thresh` | `0.04` ~ `0.08` | double | Ground plane removal height [m]. Points with $z \le \text{ground\_z\_thresh}$ in `base_link` coordinates are treated as floor. Increase if carpet/field seam points cause false obstacles. |
| `filter_params.robot_radius` | `0.80` | double | Radius around the robot center [m] to filter out chassis, bumpers, wheels, and payloads. Points with $x^2 + y^2 \le r^2$ are rejected. |
| `filter_params.robot_height_min` | `0.0` | double | Minimum relative Z height [m] for obstacle consideration. |
| `filter_params.robot_height_max` | `1.2` | double | Maximum relative Z height [m]. Discards ceilings and elevated venue lighting. |
| `filter_params.max_range` | `12.0` | double | Maximum sensor distance [m]. Cuts long-distance noisy returns. |
| `filter_params.max_points` | `200000` | int | Maximum point count allocated in GPU memory buffers. |

---

### 2.3 Motion Prediction & Velocity Estimation (`velocity_estimation.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `velocity_estimation.enable` | `true` | bool | Enables translation estimation and constant-velocity motion extrapolation between scans. Prevents search window lag during rapid acceleration. |
| `velocity_estimation.window_sec` | `0.20` | double | Time window [s] over which historical poses are retained to differentiate linear velocity. Mitigates high-frequency sensor noise. |
| `velocity_estimation.min_dt_sec` | `0.15` | double | Minimum duration [s] before computing $\Delta \mathbf{p} / \Delta t$. Prevents noisy velocity spikes when frames arrive in rapid succession. |
| `velocity_estimation.max_dt_sec` | `0.50` | double | Maximum time span [s] allowed for velocity extrapolation. If the robot loses localization or drops frames longer than this, velocity dead-reckoning is safely paused. |
| `velocity_estimation.max_linear_velocity` | `4.0` | double | Upper limit clamp [m/s] on estimated translational speed. Any higher value is flagged as an outlier and ignored. |

---

### 2.4 ESDF Local Matching & Residual Weighting (`matching_params.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `matching_params.search_range_xy` | `0.30` | double | Local search grid half-width [m] along X and Y axes around predicted pose. |
| `matching_params.search_step_xy` | `0.03` ~ `0.05` | double | Coarse search step resolution [m] along X and Y. Smaller values increase accuracy but require more GPU compute. |
| `matching_params.search_range_yaw` | `0.15` ~ `0.60` | double | Base yaw search half-width [rad]. |
| `matching_params.search_step_yaw` | `0.02` ~ `0.05` | double | Angular search step resolution [rad]. |
| `matching_params.dynamic_yaw_expansion` | `true` | bool | Automatically expands yaw search window when rotating rapidly. |
| `matching_params.dynamic_yaw_omega_thresh` | `0.50` | double | Angular velocity threshold [rad/s] above which dynamic yaw search range expansion activates. |
| `matching_params.dynamic_yaw_factor` | `0.15` | double | Proportional factor [s] multiplied by $|\omega_z|$ to expand yaw range: $\text{range\_yaw} = \max(\text{base}, |\omega_z| \times \text{factor})$. |
| `matching_params.dynamic_yaw_max_range` | `0.35` | double | Upper bound clamp [rad] on dynamic yaw expansion window. |
| `matching_params.fine_refine` | `true` | bool | Perform hierarchical sub-grid refinement around the coarse minimum. |
| `matching_params.fine_refine_levels` | `2` | int | Refinement levels (1 to 3). |
| `matching_params.max_dist_thresh` | `0.20` ~ `0.25` | double | Maximum ESDF truncation distance [m]. Points further than this distance from map surfaces are saturated to this cost. |
| `matching_params.weight_2d_lidar` | `0.40` ~ `0.50` | double | Baseline weighting factor $w_{2D}$ for the 2D planar LiDAR residual. |
| `matching_params.min_2d_points` | `30` | int | Minimum number of valid 2D LiDAR points required to include 2D cost. |
| `matching_params.target_2d_points` | `200` | int | Point threshold for full FoV weighting: $\text{fov\_scale} = \min(1.0, N_{2D} / \text{target\_2d\_points})$. |
| `matching_params.min_3d_points` | `50` | int | Minimum 3D obstacle point count. If 3D falls below this (e.g. occluded) and 2D is valid, weight falls back to $w_{2D} = 1.0$. |
| `matching_params.cost_threshold` | `0.165` | double | Average normalized distance cost acceptance threshold [m]. Matches below this cost are accepted. |
| `matching_params.inlier_dist_thresh` | `0.08` | double | Distance threshold [m] for considering an obstacle point as an inlier to the map surface. |
| `matching_params.min_inliers` | `60` | int | Fallback inlier count threshold. |
| `matching_params.inlier_cost_thresh` | `0.05` | double | Fallback average inlier distance threshold [m]. |

---

### 2.5 2D Planar LiDAR (`lidar_2d.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `lidar_2d.enable` | `true` | bool | Enable Lakibeam 1 2D planar LiDAR integration. |
| `lidar_2d.deskew` | `true` | bool | Compensate for robot rotation during scan sweep using IMU integration (`time_increment`). |
| `lidar_2d.topic` | `"/lakibeam/scan"` | string | LaserScan input topic. |
| `lidar_2d.frame_id` | `"lakibeam_frame"` | string | Fallback TF frame ID if the incoming message has an empty `header.frame_id`. |
| `lidar_2d.tf_wait_sec` | `0.05` | double | Timeout [s] for TF buffer lookup (`base_link` $\to$ `frame_id`). |

---

### 2.6 Global Search Recovery (`global_search.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `global_search.range_min_x / max_x` | `-5.25` / `5.25` | double | X bounds of field recovery search [m]. |
| `global_search.range_min_y / max_y` | `-5.70` / `5.70` | double | Y bounds of field recovery search [m]. |
| `global_search.range_yaw_diff` | `0.785` ~ `3.1415` | double | Yaw constraint around IMU prior [rad]. |
| `global_search.step_xy` | `0.30` | double | Coarse grid resolution [m] for GPU global search. |
| `global_search.step_yaw` | `0.2618` ~ `0.30` | double | Coarse angular resolution [rad]. |
| `global_search.downsample_stride` | `2` | int | Point cloud decimation stride to speed up coarse GPU evaluation. |
| `global_search.lost_count_thresh` | `5` ~ `10` | int | Number of consecutive matching failures before entering global recovery mode. |

---

## 3. Dynamic Reconfiguration

Parameters marked as mutable can be updated dynamically at runtime via ROS 2 parameters:
```bash
ros2 param set /gn10_localization_node matching_params.weight_2d_lidar 0.60
ros2 param set /gn10_localization_node velocity_estimation.enable false
ros2 param set /gn10_localization_node global_search.step_xy 0.25
```

When a parameter is changed, the node outputs a confirmation log:
```
[INFO] [gn10_localization_node]: [DynamicReconfigure] 'matching_params.weight_2d_lidar' updated to: 0.600000
```
If `global_search.*` parameters are reconfigured, `GlobalSearcher` is automatically updated and synchronized:
```
[INFO] [gn10_localization_node]: [DynamicReconfigure] GlobalSearcher config synchronized.
```

Immutable parameters (such as topic names, ESDF map file path, and allocation sizes) are safely guarded against runtime modification and will return an error if attempted.
