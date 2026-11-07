# GN10 Point Cloud Localization Parameter Tuning Guide

[Japanese Version / 日本語版](./parameter_tuning_guide_jp.md)

This document explains the architecture, processing stages, parameter effects, anti-oscillation 3D/2D fallback strategy, safety TF gating, and practical tuning procedures for `gn10-pointcloud-localization`.

---

## 1. Architecture & Flow Overview

The localization pipeline consists of:
1. **Sensor Ingestion & Deskew Preprocessing**:
   - **3D LiDAR (Livox MID360)**: Rolling-buffer scan accumulation (`scan_accumulation.window_s = 0.10 s`) with per-point rotational (IMU yaw integration) and translational ($v_{\text{local}}$) deskew to the end-of-scan timestamp $t_{\text{3D}}$, followed by GPU `GroundFilter` obstacle extraction (`ground_z_thresh = 0.06 m`, `robot_radius = 0.80 m`).
   - **2D Planar LiDAR (Lakibeam 1)**: Per-beam rotational and translational deskew across the scan sweep (`time_increment`), plus cross-sensor time synchronization (`dt_sync = t_3d - t_2d`) to align 2D points to $t_{\text{3D}}$.
2. **Motion Prediction & Adaptive Search Window**:
   - IMU angular velocity integration ($\Delta\psi_{\text{imu}}$) + linear velocity extrapolation with Exponential Moving Average (EMA, $\alpha = 0.6$) smoothing.
   - Adaptive XY search window scaling at speeds $\ge 2.0\text{ m/s}$ and dynamic Yaw search window expansion during rapid rotation ($|\omega_z| > 0.50\text{ rad/s}$).
3. **GPU ESDF Matching with Anti-Oscillation 3D/2D Fallback (`PoseSolver::matchObstacleCloud`)**:
   - Evaluates candidate poses in $\mathcal{O}(1)$ per point using GPU 3D Texture Memory hardware trilinear interpolation, followed by 2-level hierarchical sub-voxel refinement (`0.05 m` $\to$ `0.01 m` $\to$ `0.002 m`).
   - Independently computes 3D and 2D average surface distance residuals ($C_{\text{3D}}, C_{\text{2D}}$) and employs a two-guard anti-oscillation fallback between Combined (3D+2D), 3D-only, and 2D-only modes.
4. **Safety Output Gating & Top-K Global Recovery**:
   - Strict match acceptance gating (`pass_cost || pass_inliers`).
   - **Immediate TF & Pose Stoppage on Failure**: When a match is rejected, `map -> base_link` TF and `/platform_constraint` are immediately halted (never extrapolated into an unverified state), and `/robot/operation/localization_success` publishes `false`.
   - After `lost_count_thresh = 10` consecutive rejected frames, `GlobalSearcher` performs full-field GPU grid evaluation + Non-Maximum Suppression (NMS) Top-K local refinement to recover.

### Architecture Diagrams (PlantUML)
- **Pipeline Flow & Parameter Mapping**:
  - [pipeline_flow_and_parameters.pu](pipeline_flow_and_parameters.pu) (English)
  - [pipeline_flow_and_parameters_jp.pu](pipeline_flow_and_parameters_jp.pu) (Japanese)
- **Matching Cost, Anti-Oscillation 3D/2D Fallback & Acceptance Logic**:
  - [matching_cost_and_weighting.pu](matching_cost_and_weighting.pu) (English)
  - [matching_cost_and_weighting_jp.pu](matching_cost_and_weighting_jp.pu) (Japanese)

---

## 2. Parameter Category Reference

All shared defaults are defined in [`config/common_params.yaml`](../config/common_params.yaml), with profile overrides in [`config/localization_params.yaml`](../config/localization_params.yaml), [`config/red_params.yaml`](../config/red_params.yaml), and [`config/blue_params.yaml`](../config/blue_params.yaml).

### 2.1 Map & 3D ESDF Generation (`esdf.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `map_source_type` | `"json"` | string | Map format (`"json"`, `"pcd"`, `"esdf"`, `"ros2_param"`). Profile YAML files (`red_params.yaml`, `blue_params.yaml`) typically override this to `"pcd"`. |
| `map_file_path` | `""` | string | Map file name or path. Relative paths automatically resolve inside the package's `map/` directory. |
| `esdf.resolution` | `0.05` | double | Voxel grid cell size [m] (`0.05 m` = 5 cm). Smaller values capture finer geometry at the cost of cubic GPU memory growth. |
| `esdf.max_dist` | `0.50` | double | Maximum Euclidean truncation distance [m] stored in the ESDF. |
| `esdf.crop_enabled` | `true` | bool | Crops PCD map points to the configured bounding box before ESDF generation. |
| `esdf.crop_min_x / max_x` | `-5.5` / `5.5` | double | X-axis bounding box limits [m] for ESDF generation. |
| `esdf.crop_min_y / max_y` | `-6.0` / `6.0` | double | Y-axis bounding box limits [m] for ESDF generation. |
| `esdf.min_z / max_z` | `-0.20` / `4.00` | double | Z-axis height limits [m] for ESDF generation. |
| `esdf.publish_map` | `true` | bool | Publishes the ESDF voxel cloud once at startup on `/esdf_map` (`transient_local` QoS). |
| `esdf.publish_max_distance` | `-1.0` | double | Maximum voxel distance [m] included in `/esdf_map` (`-1.0` exports all surface voxels $< \text{max\_dist}$). |
| `esdf.publish_stride` | `1` | int | Voxel decimation stride for `/esdf_map` (`1` = full resolution, `2` = 1/8 voxels). |

---

### 2.2 Scan Accumulation (`scan_accumulation.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `scan_accumulation.window_s` | `0.10` | double | Livox 3D LiDAR rolling accumulation window [s]. Set to `0.10` (100 ms) for Livox MID360 (10 Hz scan rate) to ensure dense obstacle coverage. |
| `scan_accumulation.timestamp_field` | `"timestamp"` | string | PointCloud2 field name containing per-point timestamps. |
| `scan_accumulation.timestamp_scale` | `1.0e-9` | double | Multiplier to convert raw point timestamps into seconds (`1e-9` for Livox nanosecond timestamps). |
| `scan_accumulation.timestamp_relative` | `false` | bool | `false` when point timestamps are absolute epoch nanoseconds; `true` when relative to `header.stamp`. |

---

### 2.3 Point Cloud Filtering (`filter_params.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `filter_params.ground_z_thresh` | `0.06` | double | Floor removal height threshold [m] in `base_link`. Points with $z \le 0.06\text{ m}$ are classified as ground. Increase slightly if floor unevenness or carpet seams appear in `/obstacle_cloud`. |
| `filter_params.robot_radius` | `0.80` | double | Self-chassis exclusion radius [m] in the XY plane ($x^2 + y^2 \le r^2$). Rejects reflections from bumpers, wheels, and mechanisms. |
| `filter_params.robot_height_min` | `0.0` | double | Minimum valid obstacle height $z$ [m] in `base_link`. |
| `filter_params.robot_height_max` | `1.2` | double | Maximum valid obstacle height $z$ [m] in `base_link`. Rejects ceiling structures and overhead lighting. |
| `filter_params.max_range` | `12.0` | double | Maximum radial sensor distance [m] from `base_link`. |
| `filter_params.max_points` | `200000` | int | Pre-allocated GPU buffer capacity (maximum points per frame). |

---

### 2.4 Motion Prediction & Velocity Estimation (`velocity_estimation.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `velocity_estimation.enable` | `true` | bool | Enables constant-velocity motion extrapolation between frames to keep the local search grid centered during acceleration and high-speed transit. |
| `velocity_estimation.window_sec` | `0.20` | double | Pose history sliding window [s] used to differentiate linear velocity. |
| `velocity_estimation.min_dt_sec` | `0.15` | double | Minimum elapsed time $\Delta t$ [s] between pose samples required to compute $\Delta \mathbf{p} / \Delta t$. Prevents noise amplification across short intervals. |
| `velocity_estimation.max_dt_sec` | `0.50` | double | Maximum elapsed time [s] over which velocity extrapolation is permitted. If tracking is lost longer than `0.50 s`, velocity dead-reckoning is zeroed for safety. |
| `velocity_estimation.max_linear_velocity` | `4.0` | double | Physical upper speed clamp [m/s] to reject outlier pose jumps. |

An Exponential Moving Average (EMA) filter with $\alpha = 0.6$ smooths the differentiated velocity:
$$\mathbf{v}_k = 0.6 \, \mathbf{v}_{\text{raw}} + 0.4 \, \mathbf{v}_{k-1}$$

When linear speed $\|\mathbf{v}\| \ge 2.0\text{ m/s}$, the translational search half-width `search_range_xy` automatically scales up to $1.5\times$ (`0.30 m` $\to$ `0.45 m`):
$$\text{scale} = \min\left(1.5,\; 1.0 + 0.5 \times \frac{\|\mathbf{v}\| - 2.0}{v_{\max} - 2.0}\right)$$

---

### 2.5 ESDF Local Matching, Independent 3D/2D Weighting & Anti-Oscillation Fallback (`matching_params.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `matching_params.search_range_xy` | `0.30` | double | Coarse local XY search half-width [m] ($\pm 0.30\text{ m}$ around predicted pose). |
| `matching_params.search_step_xy` | `0.05` | double | Coarse local XY grid step [m] (`0.05 m` = 13 $\times$ 13 XY grid). |
| `matching_params.search_range_yaw` | `0.15` | double | Baseline yaw search half-width [rad] ($\approx \pm 8.6^\circ$). |
| `matching_params.search_step_yaw` | `0.03` | double | Coarse yaw grid step [rad] ($\approx 1.7^\circ$, 11 yaw steps). |
| `matching_params.dynamic_yaw_expansion` | `true` | bool | Expands `search_range_yaw` proportionally to $|\omega_z|$ during fast turns. |
| `matching_params.dynamic_yaw_omega_thresh` | `0.50` | double | Angular speed threshold $|\omega_z|$ [rad/s] that triggers yaw window expansion. |
| `matching_params.dynamic_yaw_factor` | `0.15` | double | Expansion gain [s]: $\text{range\_yaw} = \max(\text{base}, \min(\text{max\_range}, |\omega_z| \times 0.15))$. |
| `matching_params.dynamic_yaw_max_range` | `0.40` | double | Maximum expanded yaw search half-width [rad] ($\approx \pm 22.9^\circ$). |
| `matching_params.fine_refine` | `true` | bool | Enables hierarchical $5 \times 5 \times 5$ sub-voxel refinement around the coarse optimum. |
| `matching_params.fine_refine_levels` | `2` | int | Number of refinement stages (`2` levels divide step by $5^2 = 25\times$: `0.05 m` $\to$ `0.002 m`, `0.03 rad` $\to$ `0.0012 rad`). |
| `matching_params.max_dist_thresh` | `0.25` | double | Per-point ESDF distance clamp [m] during matching evaluation. Prevents dynamic obstacles from skewing the pose gradient. |
| `matching_params.dynamic_dist_thresh` | `0.15` | double | Distance threshold [m] above which obstacle points are published to `/dynamic_cloud`. |
| `matching_params.weight_2d_lidar` | `0.40` | double | Baseline weight $w_{\text{2D,base}}$ for 2D planar LiDAR residual when both 3D and 2D are active. |
| `matching_params.min_2d_points` | `30` | int | Minimum 2D obstacle points ($N_{\text{2D}}$) required to include the 2D residual term. |
| `matching_params.target_2d_points` | `200` | int | Reference 2D point count for full FoV scaling: $\text{fov\_scale} = \min(1.0, N_{\text{2D}} / 200)$. |
| `matching_params.min_3d_points` | `50` | int | Minimum 3D obstacle points ($N_{\text{3D}}$). Below this threshold, matching switches directly to 2D-only ($w_{\text{2D}} = 1.0$). |
| `matching_params.cost_threshold` | `0.22` | double | Legacy alias for `cost_threshold_3d` [m]. |
| `matching_params.cost_threshold_3d` | `0.22` | double | Independent 3D average residual acceptance threshold [m] (`pass_3d`: $C_{\text{3D}} < 0.22\text{ m}$). |
| `matching_params.cost_threshold_2d` | `0.08` | double | Independent 2D average residual acceptance threshold [m] (`pass_2d`: $C_{\text{2D}} < 0.08\text{ m}$). |
| `matching_params.inlier_dist_thresh` | `0.08` | double | Surface distance threshold [m] (`8 cm`) to classify an obstacle point as an inlier. |
| `matching_params.min_inliers` | `400` | int | Minimum inlier point count ($N_{\text{inlier}} \ge 400$) required for secondary inlier acceptance. |
| `matching_params.inlier_cost_thresh` | `0.035` | double | Maximum average inlier distance [m] ($C_{\text{inlier}} < 0.035\text{ m}$) for inlier acceptance. |
| `matching_params.min_inlier_ratio` | `0.28` | double | Minimum inlier ratio ($N_{\text{inlier}} / N_{\text{total}} \ge 0.28$). Prevents false locking onto partial walls. |
| `matching_params.max_inlier_accept_cost` | `0.22` | double | Maximum total composite cost [m] ($C_{\text{total}} < 0.22\text{ m}$) allowed when accepting via inliers. |

#### Independent 3D/2D Cost Formulation
For each candidate pose, the GPU kernel separately averages the clamped ESDF distances of 3D points ($N_{\text{3D}}$) and 2D points ($N_{\text{2D}}$):
$$C_{\text{3D}} = \frac{1}{N_{\text{3D}}} \sum_{i=1}^{N_{\text{3D}}} \min(d_i, d_{\max}), \qquad C_{\text{2D}} = \frac{1}{N_{\text{2D}}} \sum_{j=1}^{N_{\text{2D}}} \min(d_j, d_{\max})$$
$$C_{\text{total}} = (1 - w_{\text{2D}}) \, C_{\text{3D}} + w_{\text{2D}} \, C_{\text{2D}}, \qquad w_{\text{2D}} = w_{\text{2D,base}} \cdot \min\!\left(1.0, \frac{N_{\text{2D}}}{\text{target\_2d\_points}}\right)$$

#### Anti-Oscillation Bidirectional 3D / 2D Fallback (`PoseSolver::matchObstacleCloud`)
When both 3D and 2D LiDARs have sufficient points ($N_{\text{3D}} \ge 50$ and $N_{\text{2D}} \ge 30$), a naive hard switch between Combined ($w_{\text{2D}} = 0.40$), 3D-only ($w_{\text{2D}} = 0.0$), and 2D-only ($w_{\text{2D}} = 1.0$) could cause frame-to-frame pose oscillation if the two sensors have slight extrinsic/map discrepancies or if 2D-only jumps to a distant wall. Two guards prevent this:

1. **Guard 1 — Hysteresis Deadband on 2D Detachment (`[0.08 m, 0.16 m)`)**:
   - After the initial Combined match ($w_{\text{2D}} = 0.40 \cdot \text{fov\_scale}$), if `pass_3d` ($C_{\text{3D}} < 0.22\text{ m}$) already holds, 2D is **not** detached when $C_{\text{2D}}$ merely crosses `cost_threshold_2d` (`0.08 m`).
   - Instead, 2D is detached (`w_2d = 0.0`, re-running 3D-only) **only if** `!pass_3d` **or** $C_{\text{2D}} \ge 2.0 \times \text{cost\_threshold\_2d}$ (`0.16 m`, severe 2D corruption/occlusion).
   - For mild 2D offsets in $[0.08, 0.16)\text{ m}$ where 3D is still healthy, the pose remains continuously weighted at $w_{\text{2D}} = 0.40 \cdot \text{fov\_scale}$ and `pass_cost` succeeds via `pass_3d`, eliminating threshold-boundary weight chattering.
2. **Guard 2 — Tight Local Continuity Window (`±0.10 m, ±0.06 rad`) on 2D-Only Fallback**:
   - If 3D-only also fails (`!pass_3d`), the solver attempts 2D-only fallback ($w_{\text{2D}} = 1.0$).
   - Because 3D points are still present ($N_{\text{3D}} \ge 50$), the 2D-only coarse search window around the IMU+velocity predicted `search_base_pose` is restricted to:
     $$\text{range\_xy}_{\text{2D-only}} = \min(\text{range\_xy},\; 2 \cdot \text{step\_xy}) = \pm 0.10\text{ m}, \qquad \text{range\_yaw}_{\text{2D-only}} = \min(\text{range\_yaw},\; 2 \cdot \text{step\_yaw}) = \pm 0.06\text{ rad}$$
   - This allows 2D LiDAR to maintain smooth local tracking when 3D is momentarily degraded, while preventing 2D-only from jumping 20–30 cm to a parallel wall and oscillating against 3D on the next frame. If 2D-only also fails ($C_{\text{2D}} \ge 0.08\text{ m}$), the 3D-only diagnostic state is restored.

#### Match Acceptance Logic
A local match is accepted (`match_accepted = true`) if and only if:
1. **Primary Cost Gate (`pass_cost`)**:
   - Combined mode (mild 2D offset $< 0.16\text{ m}$): $C_{\text{3D}} < 0.22\text{ m}$
   - 3D-only mode ($w_{\text{2D}} = 0.0$): $C_{\text{3D}} < 0.22\text{ m}$
   - 2D-only mode ($w_{\text{2D}} = 1.0$): $C_{\text{2D}} < 0.08\text{ m}$
2. **OR Secondary Inlier Gate (`pass_inliers`)** — all 4 conditions must hold simultaneously:
   - $N_{\text{inlier}} \ge 400$ (`min_inliers`)
   - $C_{\text{inlier}} < 0.035\text{ m}$ (`inlier_cost_thresh`)
   - $N_{\text{inlier}} / N_{\text{total}} \ge 0.28$ (`min_inlier_ratio`)
   - $C_{\text{total}} < 0.22\text{ m}$ (`max_inlier_accept_cost`)

---

### 2.6 2D Planar LiDAR (`lidar_2d.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `lidar_2d.enable` | `true` | bool | Enables Lakibeam 1 2D planar LiDAR integration. |
| `lidar_2d.deskew` | `true` | bool | Compensates each beam for rotational ($\Delta\psi_{\text{imu}}$) and translational ($v_{\text{local}}$) motion during the scan sweep, and synchronizes the 2D scan timestamp to the 3D scan reference time $t_{\text{3D}}$. |
| `lidar_2d.time_offset_sec` | `0.0` | double | Static timestamp offset [s] added to `LaserScan.header.stamp`. Since `Lakibeam_ROS2_Driver` stamps each scan with the first-beam packet time, keep this at `0.0`. |
| `lidar_2d.topic` | `"/lakibeam/scan"` | string | Input `sensor_msgs/msg/LaserScan` topic. |
| `lidar_2d.frame_id` | `"lakibeam_frame"` | string | Fallback TF frame ID if `LaserScan.header.frame_id` is empty. |
| `lidar_2d.tf_wait_sec` | `0.05` | double | Timeout [s] for `base_link` $\leftarrow$ `lakibeam_frame` TF lookup. |

---

### 2.7 Global Search Recovery (`global_search.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `global_search.range_min_x / max_x` | `-5.25` / `5.25` | double | Field X search bounds [m] during global recovery. |
| `global_search.range_min_y / max_y` | `-5.70` / `5.70` | double | Field Y search bounds [m] during global recovery. |
| `global_search.range_yaw_diff` | `3.1415` | double | Yaw search half-width [rad] around IMU yaw prior (`1.0 rad` in `red_params.yaml` / `blue_params.yaml`). |
| `global_search.step_xy` | `0.30` | double | Coarse XY grid resolution [m] for full-field GPU evaluation. |
| `global_search.step_yaw` | `0.30` | double | Coarse yaw grid resolution [rad]. |
| `global_search.downsample_stride` | `2` | int | Obstacle point decimation stride during coarse global search. |
| `global_search.top_k_candidates` | `4` | int | Number of distinct coarse minima extracted via Non-Maximum Suppression (NMS: $\Delta xy \ge 1.0\text{ m}$ or $|\Delta\psi| \ge 0.5\text{ rad}$) for fine local refinement. |
| `global_search.lost_count_thresh` | `10` | int | Consecutive failed frames before triggering asynchronous global recovery (`~0.2 s` at 50 Hz). |

---

### 2.8 Telemetry & Diagnostics (`telemetry.*`, `topics.*`)
| Parameter | Default | Type | Description & Tuning Advice |
|---|---|---|---|
| `telemetry.enable` | `true` | bool | Publishes `LocalizationTelemetry` at the cloud rate (~50 Hz). |
| `topics.output_telemetry` | `"/localization/telemetry"` | string | Topic name for `gn10_pointcloud_localization/msg/LocalizationTelemetry`. |
| `topics.output_localization_success` | `"/robot/operation/localization_success"` | string | Topic name for health status boolean (`std_msgs/msg/Bool`). |

---

## 3. Practical Tuning Workflow with Telemetry

### 3.1 Telemetry Data Fields
`gn10_pointcloud_localization/msg/LocalizationTelemetry` provides:
- `best_cost`: Total composite matching cost $C_{\text{total}}$ [m].
- `cost_3d`, `cost_2d`: Independent 3D and 2D normalized residual costs [m].
- `weight_2d`: Active 2D weight $w_{\text{2D}}$ (`0.0` in 3D-only, `0.40 * fov_scale` in Combined, `1.0` in 2D-only).
- `inlier_count`, `inlier_cost`, `inlier_ratio`: Inlier count $N_{\text{inlier}}$, average distance $C_{\text{inlier}}$ [m], and fraction $N_{\text{inlier}} / N_{\text{total}}$.
- `obstacle_count_3d`, `obstacle_count_2d`, `obstacle_count_total`: Obstacle point counts per sensor.
- `linear_speed`, `velocity_x`, `velocity_y`: EMA-smoothed linear velocity [m/s].
- `search_range_xy`, `search_range_yaw`: Active coarse search window half-widths [m, rad].
- `match_accepted`, `is_lost`, `lost_frame_count`, `global_recovered`: Tracking health and recovery flags.

### 3.2 Visualizing with PlotJuggler or `scripts/plot_telemetry.py`
```bash
# Real-time PlotJuggler:
ros2 run plotjuggler plotjuggler

# Built-in interactive plotter:
ros2 run gn10_pointcloud_localization plot_telemetry.py

# Offline CSV analysis:
ros2 topic echo /localization/telemetry --csv > telemetry.csv
python3 scripts/plot_telemetry.py --csv telemetry.csv --save telemetry_analysis.png
```
Recommended curves to inspect:
- `cost_3d` vs `cost_threshold_3d` (`0.22 m`) and `cost_2d` vs `cost_threshold_2d` (`0.08 m`) / detachment threshold (`0.16 m`)
- `weight_2d` (verify smooth `0.40` behavior without frame-to-frame chattering)
- `inlier_ratio` vs `min_inlier_ratio` (`0.28`) and `inlier_count` vs `min_inliers` (`400`)

---

## 4. Dynamic Reconfiguration

Parameters marked as mutable can be updated online via `ros2 param set`:
```bash
ros2 param set /gn10_pointcloud_localization_node matching_params.weight_2d_lidar 0.40
ros2 param set /gn10_pointcloud_localization_node matching_params.cost_threshold_3d 0.22
ros2 param set /gn10_pointcloud_localization_node matching_params.cost_threshold_2d 0.08
```
Immutable parameters (topic names, map file paths, and GPU buffer sizes) are protected and reject runtime changes.
