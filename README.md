# gn10-pointcloud-localization

[Japanese Version / 日本語版](./README_jp.md)

A real-time GPU-accelerated 3D + 2D LiDAR self-localization ROS 2 package built for the GN10 autonomous robot (Livox MID360 3D LiDAR + Lakibeam 1 2D Planar LiDAR + 6-axis IMU).

It achieves instant global pose recovery (kidnapped robot problem) and sub-centimeter local tracking using only LiDAR point clouds and IMU data—without requiring wheel odometry.

<img width="814" height="562" alt="image" src="https://github.com/user-attachments/assets/4254e8ba-a133-42b7-86b6-07034116f0f9" />

## Table of Contents

1. [Overview and Key Features](#1-overview-and-key-features)
2. [How 3D ESDF & GPU 3D Texture Memory Work](#2-how-3d-esdf--gpu-3d-texture-memory-work)
3. [System Architecture and Localization Algorithm](#3-system-architecture-and-localization-algorithm)
   - [Anti-Oscillation 3D / 2D Bidirectional Fallback](#anti-oscillation-3d--2d-bidirectional-fallback)
   - [Safety Output Gating (Immediate TF Stoppage on Failure)](#safety-output-gating-immediate-tf-stoppage-on-failure)
4. [Build Environment and Dependencies](#4-build-environment-and-dependencies)
5. [Usage](#5-usage)
   - [Placing Map Files](#placing-map-files)
   - [Pre-converting PCD to ESDF (CLI)](#pre-converting-pcd-to-esdf-cli)
   - [Launch](#launch)
   - [RViz2 ESDF Map Visualization](#rviz2-esdf-map-visualization)
6. [Parameter Reference](#6-parameter-reference)
7. [Documentation & Architecture Diagrams](#7-documentation--architecture-diagrams)
8. [License](#8-license)

---

## 1. Overview and Key Features

* **$\mathcal{O}(1)$ Ultra-Fast Scan Matching via GPU 3D Texture Memory**:
  Converts point cloud maps (`.pcd`) or CAD definitions (`.json`) into a **3D ESDF (Euclidean Signed Distance Field)** and binds it to NVIDIA GPU **3D Texture Memory**. Hardware trilinear interpolation evaluates surface distance residuals in constant $\mathcal{O}(1)$ time per point without k-d Tree traversal or branch divergence.
* **Hybrid 3D + 2D LiDAR Fusion with Anti-Oscillation Fallback**:
  Evaluates 3D LiDAR (`Livox MID360`) and 2D planar LiDAR (`Lakibeam 1`) residuals independently in the GPU kernel, combining them with Field-of-View (FoV) scaling and a two-guard anti-oscillation fallback so tracking survives sensor occlusion without pose jumping.
* **Full Rotational & Translational Deskew + Cross-Sensor Time Sync**:
  Compensates both 3D accumulated scans (`0.10 s` window) and 2D sweep scans (`time_increment`) for IMU yaw rotation and robot linear velocity ($v_{\text{local}}$), aligning 2D points to the exact 3D scan reference timestamp $t_{\text{3D}}$.
* **Global Pose Recovery (Top-K NMS Kidnapped Robot Solver)**:
  Evaluates thousands of candidate poses across the full field in a single GPU pass and refines the Top-K spatially distinct minima (via Non-Maximum Suppression) to recover rapidly from lost states.
* **Dynamic Obstacle Extraction**:
  Classifies points exceeding `dynamic_dist_thresh` (`0.15 m`) from static map surfaces in real time and publishes them to `/dynamic_cloud`.

---

## 2. How 3D ESDF & GPU 3D Texture Memory Work

<img width="1007" height="711" alt="image" src="https://github.com/user-attachments/assets/cd8ddd3f-9a64-45c4-b60e-91cef41f2de9" />

### Traditional Nearest-Neighbor Search vs. 3D ESDF Texture Fetch
Querying a k-d Tree directly on a PCD map requires $\mathcal{O}(\log N_{\text{pcd}})$ operations per point and causes severe GPU warp branch divergence. By pre-computing the Euclidean distance to the nearest surface on a `0.05 m` 3D voxel grid and uploading it to a CUDA 3D Texture (`cudaTextureObject_t`):
1. **Single-Cycle Hardware Trilinear Interpolation**: Dedicated GPU texture units interpolate continuous sub-voxel distances in hardware.
2. **Zero Warp Divergence**: Every GPU thread executes an identical `tex3D<float>()` instruction, enabling thousands of candidate poses $(x, y, \psi)$ to be evaluated in a few milliseconds on Jetson Orin Nano.

---

## 3. System Architecture and Localization Algorithm

```
[3D LiDAR (Livox MID360)] ──> [ScanAccumulator (0.10s Deskew)] ──> [GroundFilter (CUDA)] ──> [3D Obstacles]
                                                                                                   │
[2D LiDAR (Lakibeam 1)]   ──> [LaserScanDeskew (Rot+Trans+Sync)] ──────────────────────────> [2D Obstacles]
                                                                                                   │
[IMU + EMA Velocity Pred] ─────────────────────────────────────────────────────────────────>       ▼
                                                                                    [ESDF Matcher (CUDA 3D Texture)]
[3D ESDF Map (GPU Texture)] ───────────────────────────────────────────────────────────────>       │
      │                                                                                            ▼
      ▼ (Transient Local)                                                   [Anti-Oscillation 3D/2D Fallback & Gate]
[ESDF Cloud (/esdf_map)]                                                                           │
                                                                          ┌────────────────────────┴────────────────────────┐
                                                                          ▼ (Accepted)                                      ▼ (Rejected)
                                                         [Publish TF (map -> base_link)]                   [Halt TF & Pose Output Immediately]
                                                         [/platform_constraint & /dynamic_cloud]           [localization_success = false]
                                                         [localization_success = true]                     [Trigger GlobalSearch after 10 frames]
```

### Anti-Oscillation 3D / 2D Bidirectional Fallback
In [`PoseSolver::matchObstacleCloud`](src/pose_solver.cpp), 3D ($N_{\text{3D}}$) and 2D ($N_{\text{2D}}$) average surface distance costs ($C_{\text{3D}}, C_{\text{2D}}$) are computed independently. To prevent frame-to-frame pose oscillation when switching between Combined (`w_2d = 0.40 * fov_scale`), 3D-only (`w_2d = 0.0`), and 2D-only (`w_2d = 1.0`) modes, two guards are enforced when both sensors have points ($N_{\text{3D}} \ge 50, N_{\text{2D}} \ge 30$):

1. **Guard 1 — Hysteresis Deadband on 2D Detachment (`[0.08 m, 0.16 m)`)**:
   - When the Combined match satisfies `pass_3d` ($C_{\text{3D}} < 0.22\text{ m}$), 2D is **not** detached if $C_{\text{2D}}$ merely exceeds `cost_threshold_2d` (`0.08 m`).
   - 2D is detached (`w_2d = 0.0`, re-running 3D-only) **only if** `!pass_3d` **or** $C_{\text{2D}} \ge 2.0 \times \text{cost\_threshold\_2d}$ (`0.16 m`).
   - Within the `[0.08 m, 0.16 m)` deadband, `w_2d` stays smooth at `0.40 * fov_scale` and `pass_cost` passes via `pass_3d`, preventing weight chattering near `0.08 m`.
2. **Guard 2 — Tight Local Continuity Window (`±0.10 m, ±0.06 rad`) on 2D-Only Fallback**:
   - If 3D-only also fails (`!pass_3d`), the solver attempts 2D-only (`w_2d = 1.0`).
   - Because 3D points are still present ($N_{\text{3D}} \ge 50$), the 2D-only coarse search window around the IMU+velocity predicted `search_base_pose` is clamped to $\min(\text{range\_xy}, 2 \cdot \text{step\_xy}) = \pm 0.10\text{ m}$ and $\min(\text{range\_yaw}, 2 \cdot \text{step\_yaw}) = \pm 0.06\text{ rad}$.
   - This keeps local tracking alive during brief 3D degradation while making it impossible for 2D-only to jump 20–30 cm to a false parallel wall and oscillate against 3D. (Full `±0.30 m` 2D-only search is used only when $N_{\text{3D}} < 50$, i.e., complete 3D sensor dropout).

### Safety Output Gating (Immediate TF Stoppage on Failure)
* When a frame fails both `pass_cost` and `pass_inliers`, [`LocalizationNode`](src/localization_node.cpp) **immediately stops publishing** `map -> base_link` TF and `/platform_constraint`, and publishes `false` on `/robot/operation/localization_success`.
* Unverified IMU/velocity predictions are **never** broadcast as TF, ensuring downstream controllers trigger an immediate safety stop rather than driving blindly on dead-reckoning.

---

## 4. Build Environment and Dependencies

* **OS**: Ubuntu 22.04 LTS
* **ROS**: ROS 2 Humble
* **GPU**: NVIDIA GPU (Compute Capability 8.7 / 8.9: Jetson Orin Nano / AGX Orin / RTX 40 series)
* **CUDA Toolkit**: 12.0 or higher

### Installing Dependencies & Building

```bash
sudo apt update
sudo apt install -y \
  libceres-dev \
  libeigen3-dev \
  nlohmann-json3-dev \
  libpcl-dev \
  libopenmpi-dev \
  ros-humble-pcl-conversions

cd ~/ros2_ws
colcon build --symlink-install --packages-select gn10_pointcloud_localization
source install/setup.bash
```

---

## 5. Usage

### Placing Map Files
Place map files (`.pcd`, `.esdf`, `.json`) in the package's `map/` folder. Relative paths in YAML config files resolve inside `map/` automatically.

### Pre-converting PCD to ESDF (CLI)
```bash
# Usage: pcd_to_esdf_converter <input.pcd> <output.esdf> <resolution_m> [max_dist_m] [min_x max_x min_y max_y min_z max_z]
ros2 run gn10_pointcloud_localization pcd_to_esdf_converter \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.pcd \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.esdf \
  0.05 0.50 -5.5 5.5 -6.0 6.0 -0.2 4.0
```
When `map_source_type: "pcd"` is used at launch, a binary cache (`map/cache/<filename>.esdf`) is automatically generated on the first run and reused on subsequent startups.

### Launch
```bash
# Default launch (common_params.yaml + localization_params.yaml)
ros2 launch gn10_pointcloud_localization localization.launch.py

# Simulation / Rosbag playback
ros2 launch gn10_pointcloud_localization localization.launch.py use_sim_time:=true

# Competition Zone Presets (Red Zone / Blue Zone)
ros2 launch gn10_pointcloud_localization red.launch.py
ros2 launch gn10_pointcloud_localization blue.launch.py
```

#### Calibrated Static TF Defaults in Launch Files
All three launch files (`red.launch.py`, `blue.launch.py`, `localization.launch.py`) publish identical real-robot calibrated static transforms (`start_livox_tf:=true`, `start_lakibeam_tf:=true`):
- **`base_link -> livox_frame`**: `x = 0.25 m, y = -0.25 m, z = 1.09 m, yaw = 0.0 rad, pitch = -0.273 rad, roll = 3.14159 rad`
- **`base_link -> lakibeam_frame`**: `x = -0.33 m, y = 0.0 m, z = 0.08 m, yaw = 3.13 rad, pitch = 0.0 rad, roll = 0.0 rad`

### RViz2 ESDF Map Visualization
At startup, the node publishes the ESDF surface voxel cloud once to `/esdf_map` (`sensor_msgs/msg/PointCloud2`, `Transient Local` QoS).
- **Topic**: `/esdf_map` (`Durability Policy: Transient Local`, `Color Transformer: Intensity`, `Style: Flat Squares ~0.04 m`)

---

## 6. Parameter Reference

Shared parameters are defined in [`config/common_params.yaml`](config/common_params.yaml) and overridden by profile configs (`localization_params.yaml`, `red_params.yaml`, `blue_params.yaml`).

| Parameter Name | Default (`common_params.yaml`) | Role / Mechanism |
| --- | :---: | --- |
| `map_source_type` | `"json"` (`"pcd"` in red/blue) | Map format (`json`, `pcd`, `esdf`, `ros2_param`). |
| `map_file_path` | `""` | Map filename or path (searched in `map/` if relative). |
| `esdf.resolution` | `0.05` | 3D ESDF voxel resolution [m]. |
| `esdf.max_dist` | `0.50` | Maximum truncation distance [m] stored in the ESDF. |
| `esdf.crop_enabled` | `true` | Crops PCD map to `[crop_min_x..max_x, crop_min_y..max_y, min_z..max_z]`. |
| `esdf.crop_min_x` / `crop_max_x` | `-5.5` / `5.5` | ESDF X crop bounds [m]. |
| `esdf.crop_min_y` / `crop_max_y` | `-6.0` / `6.0` | ESDF Y crop bounds [m]. |
| `esdf.min_z` / `max_z` | `-0.20` / `4.00` | ESDF Z crop bounds [m]. |
| `scan_accumulation.window_s` | `0.10` | 3D LiDAR rolling accumulation window [s] with per-point motion deskew. |
| `filter_params.ground_z_thresh` | `0.06` | Floor removal threshold $z$ [m] in `base_link`. |
| `filter_params.robot_radius` | `0.80` | Robot chassis exclusion radius [m] in XY plane. |
| `filter_params.robot_height_min` / `max` | `0.0` / `1.2` | Valid obstacle Z height range [m] in `base_link`. |
| `filter_params.max_range` | `12.0` | Maximum radial sensor range [m]. |
| `matching_params.search_range_xy` | `0.30` | Local XY search half-width [m] (scales up to $1.5\times$ at speeds $\ge 2.0\text{ m/s}$). |
| `matching_params.search_step_xy` | `0.05` | Coarse XY grid step [m]. |
| `matching_params.search_range_yaw` | `0.15` | Baseline yaw search half-width [rad] ($\approx \pm 8.6^\circ$). |
| `matching_params.search_step_yaw` | `0.03` | Coarse yaw grid step [rad] ($\approx 1.7^\circ$). |
| `matching_params.fine_refine` / `levels` | `true` / `2` | 2-level $5 \times 5 \times 5$ sub-voxel refinement (`0.05 m` $\to$ `0.002 m`). |
| `matching_params.max_dist_thresh` | `0.25` | Per-point ESDF residual truncation distance [m]. |
| `matching_params.dynamic_dist_thresh` | `0.15` | Surface distance threshold [m] for `/dynamic_cloud` obstacle extraction. |
| `matching_params.cost_threshold` | `0.22` | Legacy alias for `cost_threshold_3d` [m]. |
| `matching_params.cost_threshold_3d` | `0.22` | Independent 3D residual acceptance threshold [m] ($C_{\text{3D}} < 0.22\text{ m}$). |
| `matching_params.cost_threshold_2d` | `0.08` | Independent 2D residual acceptance threshold [m] ($C_{\text{2D}} < 0.08\text{ m}$; hysteresis detachment at $2\times = 0.16\text{ m}$). |
| `matching_params.inlier_dist_thresh` | `0.08` | Surface distance threshold [m] (`8 cm`) for inlier classification. |
| `matching_params.min_inliers` | `400` | Minimum inlier count ($N_{\text{inlier}} \ge 400$) for secondary inlier acceptance. |
| `matching_params.inlier_cost_thresh` | `0.035` | Maximum average inlier distance [m] ($C_{\text{inlier}} < 0.035\text{ m}$). |
| `matching_params.min_inlier_ratio` | `0.28` | Minimum inlier ratio ($N_{\text{inlier}} / N_{\text{total}} \ge 0.28$). |
| `matching_params.max_inlier_accept_cost` | `0.22` | Upper total composite cost guard [m] ($C_{\text{total}} < 0.22\text{ m}$) for inlier acceptance. |
| `matching_params.weight_2d_lidar` | `0.40` | Baseline weight $w_{\text{2D,base}}$ for 2D LiDAR residual in Combined mode. |
| `matching_params.min_2d_points` / `target_2d_points` | `30` / `200` | Minimum 2D points to activate 2D cost and reference count for 100% FoV scaling. |
| `matching_params.min_3d_points` | `50` | Minimum 3D points; below this, switches to full-window 2D-only matching ($w_{\text{2D}} = 1.0$). |
| `matching_params.dynamic_yaw_expansion` | `true` | Expands yaw search window up to `0.40 rad` when $|\omega_z| > 0.50\text{ rad/s}$ (`factor: 0.15`). |
| `velocity_estimation.enable` | `true` | Enables EMA-smoothed ($\alpha = 0.6$) linear velocity extrapolation (`window_sec: 0.20`, `min_dt_sec: 0.15`, `max_dt_sec: 0.50`, `max_linear_velocity: 4.0`). |
| `lidar_2d.enable` / `deskew` | `true` / `true` | Enables Lakibeam 2D LiDAR with per-beam rotational + translational deskew and 3D time sync. |
| `lidar_2d.time_offset_sec` | `0.0` | Static timestamp offset [s] for 2D scans. |
| `global_search.top_k_candidates` | `4` | Number of NMS-separated coarse candidates refined during global recovery. |
| `global_search.lost_count_thresh` | `10` | Consecutive failed frames before triggering global recovery. |

---

## 7. Documentation & Architecture Diagrams

All important documents and PlantUML diagrams are maintained in both English and Japanese:

| Document / Diagram | English | Japanese (日本語) |
| --- | --- | --- |
| **README** | [README.md](./README.md) | [README_jp.md](./README_jp.md) |
| **Parameter Tuning & Algorithm Guide** | [docs/parameter_tuning_guide.md](./docs/parameter_tuning_guide.md) | [docs/parameter_tuning_guide_jp.md](./docs/parameter_tuning_guide_jp.md) |
| **Coding Rules & Safety Standards** | [docs/coding-rules.md](./docs/coding-rules.md) | [docs/coding-rules_jp.md](./docs/coding-rules_jp.md) |
| **Pipeline Flow & Parameter Diagram (PlantUML)** | [docs/pipeline_flow_and_parameters.pu](./docs/pipeline_flow_and_parameters.pu) | [docs/pipeline_flow_and_parameters_jp.pu](./docs/pipeline_flow_and_parameters_jp.pu) |
| **Matching Cost & Anti-Oscillation Fallback (PlantUML)** | [docs/matching_cost_and_weighting.pu](./docs/matching_cost_and_weighting.pu) | [docs/matching_cost_and_weighting_jp.pu](./docs/matching_cost_and_weighting_jp.pu) |

---

## 8. License

This repository is released under the [Apache 2.0 License](LICENSE).
