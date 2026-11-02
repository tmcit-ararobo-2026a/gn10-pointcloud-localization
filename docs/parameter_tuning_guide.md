# GN10 Point Cloud Localization Parameter Tuning Guide

This document explains the architecture, processing stages, parameter effects, and practical tuning procedures for `gn10-pointcloud-localization`.

---

## 1. Architecture & Flow Overview

The localization pipeline consists of sensor ingestion, deskew preprocessing, motion prediction with EMA velocity smoothing, GPU ESDF matching with independent residual weighting, strict inlier acceptance gating, and Top-K global recovery search.

The PlantUML diagrams in this directory visualize the complete dataflow:
- [pipeline_flow_and_parameters.pu](pipeline_flow_and_parameters.pu): Full pipeline dataflow from sensors to publishers, mapped to the parameters controlling each stage.
- [matching_cost_and_weighting.pu](matching_cost_and_weighting.pu): Mathematical process of the 3D/2D independent residual cost, SearchGrid evaluation, and strict inlier acceptance check.

---

## 2. Parameter Category Reference

### 2.1 Scan Accumulation (`scan_accumulation.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `scan_accumulation.window_s` | `0.10` | double | Livox LiDAR packet accumulation time window [s]. For Livox MID360 (10Hz full rotation), set to `0.10` for dense point distribution. |
| `scan_accumulation.timestamp_field` | `"timestamp"` | string | PointCloud2 field name for per-point timestamps (Livox driver uses `"timestamp"`). |
| `scan_accumulation.timestamp_scale` | `1.0e-9` | double | Multiplier to convert timestamp field values into seconds (e.g. `1e-9` for nanoseconds). |
| `scan_accumulation.timestamp_relative` | `false` | bool | Whether point timestamps are relative offsets from `header.stamp` or absolute nanoseconds. |

---

### 2.2 Point Cloud Filtering (`filter_params.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `filter_params.ground_z_thresh` | `0.06` | double | Ground plane removal height [m]. Points with $z \le \text{ground\_z\_thresh}$ in `base_link` coordinates are treated as floor. Increase if carpet seams or field ramps cause false obstacles. |
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
| `velocity_estimation.window_sec` | `0.20` | double | Time window [s] over which historical poses are retained to differentiate linear velocity. |
| `velocity_estimation.min_dt_sec` | `0.15` | double | Minimum duration [s] before computing $\Delta \mathbf{p} / \Delta t$. Prevents noisy velocity spikes when frames arrive in rapid succession. |
| `velocity_estimation.max_dt_sec` | `0.50` | double | Maximum time span [s] allowed for velocity extrapolation. If localization drops longer than this, velocity dead-reckoning is safely paused. |
| `velocity_estimation.max_linear_velocity` | `4.0` | double | Upper limit clamp [m/s] on estimated translational speed. Higher speeds are treated as outliers. |

An exponential moving average (EMA) filter with coefficient $\alpha = 0.6$ is applied to the estimated linear velocity to eliminate single-frame matching jitter:
$$\mathbf{v}_k = 0.6 \, \mathbf{v}_{\text{raw}} + 0.4 \, \mathbf{v}_{k-1}$$

Additionally, for high-speed motion ($\ge 2.0$ m/s), the translational search window $\text{range\_xy}$ is adaptively expanded up to 1.5x based on the current speed:
$$\text{scale} = \min\left(1.5, 1.0 + 0.5 \times \frac{\|\mathbf{v}\| - 2.0}{v_{\max} - 2.0}\right)$$

---

### 2.4 ESDF Local Matching & Residual Weighting (`matching_params.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `matching_params.search_range_xy` | `0.30` | double | Local search grid half-width [m] along X and Y axes around predicted pose. |
| `matching_params.search_step_xy` | `0.05` | double | Coarse search step resolution [m] along X and Y. |
| `matching_params.search_range_yaw` | `0.15` | double | Base yaw search half-width [rad] (approx. $\pm 8.6^\circ$). Keeps search focused and avoids false symmetric locking. |
| `matching_params.search_step_yaw` | `0.03` | double | Angular search step resolution [rad] (approx. $1.7^\circ$). |
| `matching_params.dynamic_yaw_expansion` | `true` | bool | Automatically expands yaw search window when rotating rapidly. |
| `matching_params.dynamic_yaw_omega_thresh` | `0.50` | double | Angular velocity threshold [rad/s] above which dynamic yaw search range expansion activates. |
| `matching_params.dynamic_yaw_factor` | `0.15` | double | Proportional factor [s] multiplied by $|\omega_z|$ to expand yaw range. |
| `matching_params.dynamic_yaw_max_range` | `0.40` | double | Upper bound clamp [rad] on dynamic yaw expansion window. Clamped with $\max(\text{range\_yaw}, \text{dynamic\_yaw\_max\_range})$. |
| `matching_params.fine_refine` | `true` | bool | Perform hierarchical sub-grid refinement around coarse minimum. |
| `matching_params.fine_refine_levels` | `2` | int | Refinement levels (1 to 3). With step 0.05m, 2 levels achieve millimeter sub-voxel precision. |
| `matching_params.max_dist_thresh` | `0.25` | double | Maximum ESDF truncation distance [m]. |
| `matching_params.weight_2d_lidar` | `0.50` | double | Baseline weighting factor $w_{2D}$ for 2D planar LiDAR residual. |
| `matching_params.min_2d_points` | `30` | int | Minimum number of valid 2D LiDAR points required to activate 2D residual. |
| `matching_params.target_2d_points` | `200` | int | Point threshold for full FoV weighting: $\text{fov\_scale} = \min(1.0, N_{2D} / \text{target\_2d\_points})$. |
| `matching_params.min_3d_points` | `50` | int | Minimum 3D obstacle point count. If 3D falls below this and 2D is valid, weight falls back to $w_{2D} = 1.0$. |
| `matching_params.cost_threshold` | `0.165` | double | Average normalized distance cost acceptance threshold [m]. |
| `matching_params.inlier_dist_thresh` | `0.08` | double | Distance threshold [m] for considering an obstacle point as an inlier to the map surface. |
| `matching_params.min_inliers` | `60` | int | Fallback minimum inlier count. |
| `matching_params.inlier_cost_thresh` | `0.01` | double | Fallback average inlier distance threshold [m]. |
| `matching_params.min_inlier_ratio` | `0.35` | double | Minimum ratio of inlier points to total obstacles ($N_{\text{inlier}} / N_{\text{total}} \ge 0.35$). Prevents partial-wall false locks. |
| `matching_params.max_inlier_accept_cost` | `0.22` | double | Upper total composite cost guard for inlier fallback acceptance. |

#### Match Acceptance Logic
A pose match is accepted if either:
1. $C_{\text{total}} < \text{cost\_threshold}$
2. OR ALL of the following 4 conditions are met simultaneously:
   - $N_{\text{inlier}} \ge \text{min\_inliers}$
   - $C_{\text{inlier}} < \text{inlier\_cost\_thresh}$
   - $N_{\text{inlier}} / N_{\text{total}} \ge \text{min\_inlier\_ratio}$
   - $C_{\text{total}} < \text{max\_inlier\_accept\_cost}$

---

### 2.5 2D Planar LiDAR (`lidar_2d.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `lidar_2d.enable` | `true` | bool | Enable Lakibeam 1 2D planar LiDAR integration. |
| `lidar_2d.deskew` | `true` | bool | Compensate for robot rotation during scan sweep using IMU integration (`time_increment`). |
| `lidar_2d.topic` | `"/lakibeam/scan"` | string | LaserScan input topic. |
| `lidar_2d.frame_id` | `"lakibeam_frame"` | string | Fallback TF frame ID if message has an empty `header.frame_id`. |
| `lidar_2d.tf_wait_sec` | `0.05` | double | Timeout [s] for TF buffer lookup (`base_link` $\to$ `frame_id`). |

---

### 2.6 Global Search Recovery (`global_search.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `global_search.range_min_x / max_x` | `-5.25` / `5.25` | double | X bounds of field recovery search [m]. |
| `global_search.range_min_y / max_y` | `-5.70` / `5.70` | double | Y bounds of field recovery search [m]. |
| `global_search.range_yaw_diff` | `3.1415` | double | Yaw search half-width around IMU prior [rad]. |
| `global_search.step_xy` | `0.30` | double | Coarse grid resolution [m] for GPU global search. |
| `global_search.step_yaw` | `0.30` | double | Coarse angular resolution [rad]. |
| `global_search.downsample_stride` | `2` | int | Point cloud decimation stride to speed up coarse GPU evaluation. |
| `global_search.top_k_candidates` | `4` | int | Number of spatially/angularly distinct candidates extracted via Non-Maximum Suppression (NMS) for local refinement. |
| `global_search.lost_count_thresh` | `10` | int | Number of consecutive matching failures before entering global recovery mode. |

---

### 2.7 Telemetry & Diagnostics (`telemetry.*`, `topics.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `telemetry.enable` | `true` | bool | Enables ~50Hz telemetry publishing for real-time analysis and plotting. |
| `topics.output_telemetry` | `"/localization/telemetry"` | string | Topic name for `gn10_pointcloud_localization/msg/LocalizationTelemetry`. |

---

## 3. Practical Tuning Workflow with Telemetry

### 3.1 Telemetry Data Fields
The node publishes `gn10_pointcloud_localization/msg/LocalizationTelemetry` at the cloud callback frequency (~50Hz):

- `best_cost`: Total composite matching cost (3D + 2D weighted).
- `cost_3d`, `cost_2d`: Individual normalized residual distance costs [m].
- `weight_2d`: Dynamic FoV-scaled weight applied to 2D LiDAR residual.
- `inlier_count`, `inlier_cost`, `inlier_ratio`: Number, average distance, and fraction of inlier points.
- `obstacle_count_3d`, `obstacle_count_2d`, `obstacle_count_total`: Obstacle points evaluated.
- `linear_speed`, `velocity_x`, `velocity_y`: Estimated velocity components and magnitude [m/s].
- `search_range_xy`, `search_range_yaw`: Active local search bounds [m, rad].
- `match_accepted`, `is_lost`, `lost_frame_count`, `global_recovered`: Health and recovery flags.

### 3.2 Visualizing with PlotJuggler
Run PlotJuggler to graph costs, inlier ratios, and search windows in real time:
```bash
ros2 run plotjuggler plotjuggler
```
Select `/localization/telemetry` and plot:
- Curve 1: `best_cost` vs `cost_threshold` (0.165) and `max_inlier_accept_cost` (0.220)
- Curve 2: `inlier_ratio` vs `min_inlier_ratio` (0.35)
- Curve 3: `linear_speed` vs `search_range_xy`

### 3.3 Visualizing with `scripts/plot_telemetry.py`
The package provides a built-in Python plotter:

```bash
# Live visualization during robot motion:
ros2 run gn10_pointcloud_localization plot_telemetry.py

# Offline plotting from a recorded CSV:
ros2 topic echo /localization/telemetry --csv > telemetry.csv
python3 scripts/plot_telemetry.py --csv telemetry.csv --save telemetry_analysis.png
```

### 3.4 Key Tuning Scenarios

1. **Robot loses tracking during sharp omnidirectional turns:**
   - Check `search_range_yaw` and `dynamic_yaw_expansion`. If angular speed is high, verify that `dynamic_yaw_max_range` expands up to `0.40` rad.
   - Verify that Lakibeam 2D deskew is active (`lidar_2d.deskew: true`).

2. **Robot locks onto parallel or 180-degree inverted symmetric walls:**
   - Check `inlier_ratio` in telemetry. Inverted or false locks typically show low inlier ratio (< 0.25).
   - Ensure `matching_params.min_inlier_ratio` is set to `0.35` and `matching_params.max_inlier_accept_cost` is `0.22`.
   - Ensure `global_search.top_k_candidates` is at least `4` so the recovery solver explores alternatives beyond the false coarse minimum.

3. **High-speed transit (3 to 4 m/s) tracking:**
   - Monitor `linear_speed` and `search_range_xy`. The adaptive search window automatically expands `range_xy` from 0.30m to 0.45m when speed exceeds 2.0 m/s.
   - If tracking lags behind sudden stops, increase `velocity_estimation.window_sec` slightly (e.g. 0.25s) or adjust EMA smoothing.

---

## 4. Dynamic Reconfiguration

Parameters marked as mutable can be updated dynamically at runtime via ROS 2 parameters:
```bash
ros2 param set /gn10_localization_node matching_params.weight_2d_lidar 0.60
ros2 param set /gn10_localization_node matching_params.min_inlier_ratio 0.40
ros2 param set /gn10_localization_node velocity_estimation.enable true
```

When a parameter is changed, the node outputs a confirmation log:
```
[INFO] [gn10_localization_node]: [DynamicReconfigure] 'matching_params.weight_2d_lidar' updated to: 0.600000
```
If `global_search.*` parameters are reconfigured, `GlobalSearcher` is automatically updated and synchronized:
```
[INFO] [gn10_localization_node]: [DynamicReconfigure] GlobalSearcher config synchronized.
```

Immutable parameters (such as topic names, ESDF map file path, and memory allocation sizes) are safely guarded against runtime modification and will return an error if attempted.
