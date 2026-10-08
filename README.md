# gn10-pointcloud-localization

[Japanese version](./README_jp.md)

A ROS 2 package for real-time 3D LiDAR self-localization using 3D LiDARs such as the Livox MID360.

It achieves global initial pose estimation and stable tracking using only point clouds and IMU data, without requiring prior odometry or initial pose estimates.

## Table of Contents

1. [Overview and Features](https://www.google.com/search?q=%231-overview-and-features)
2. [How 3D ESDF & GPU 3D Texture Memory Work](https://www.google.com/search?q=%232-how-3d-esdf--gpu-3d-texture-memory-work)
3. [System Architecture and Processing Flow](https://www.google.com/search?q=%233-system-architecture-and-processing-flow)
4. [Build Environment and Dependencies](https://www.google.com/search?q=%234-build-environment-and-dependencies)
5. [Usage](https://www.google.com/search?q=%235-usage)
* [Placing Map Files](https://www.google.com/search?q=%23placing-map-files)
* [Pre-converting PCD to ESDF (CLI)](https://www.google.com/search?q=%23pre-converting-pcd-to-esdf-cli)
* [Launch](https://www.google.com/search?q=%23launch)


6. [Parameter Explanation](https://www.google.com/search?q=%236-parameter-explanation)
7. [License](https://www.google.com/search?q=%237-license)

---

## 1. Overview and Features

This package is designed for real-time robot localization in field environments (e.g., competition arenas, indoor transit paths).

* **$\mathcal{O}(1)$ Ultra-fast Scan Matching**:
Converts point cloud maps (PCD) or CAD object definitions into a **3D ESDF (Euclidean Signed Distance Field)** and loads it into NVIDIA GPU **3D Texture Memory**. By eliminating geometric calculations and k-d Tree branch searches, it achieves exact distance field lookups in constant $\mathcal{O}(1)$ time per point using hardware-accelerated trilinear interpolation.
* **Global Pose Identification (Kidnapped Robot Solver)**:
Leverages the absolute Yaw orientation from the IMU to evaluate thousands to tens of thousands of candidate poses across the entire XY/Yaw search space within a single frame using GPU parallel reduction (CUB DeviceReduce). Instantly identifies the robot's pose even from an unknown initial state.
* **Dynamic Obstacle and Pedestrian Filtering**:
Separates dynamic obstacles, pedestrians, and nearby structures around the robot in real time based on their distance from the static map surface, publishing them to `/dynamic_cloud`.

---

## 2. How 3D ESDF & GPU 3D Texture Memory Work

### Traditional Nearest-Neighbor Search

When performing nearest-neighbor searches directly on a PCD point cloud map, querying a k-d Tree for each point results in a computational complexity of $\mathcal{O}(\log N_{\text{pcd}})$.

On GPUs, thread execution path mismatches (branch divergence) and random memory access cause significant frame rate drops.

### 3D ESDF + 3D Texture Memory

Beforehand or at startup, the map space is discretized into a voxel grid to construct a 3D distance field (ESDF), where each voxel cell stores the Euclidean distance to the nearest wall or obstacle surface.

1. **Zero-Overhead Interpolation**:
Dedicated GPU texture units execute trilinear interpolation in nearly 1 clock cycle in hardware. This enables smooth and continuous sampling between grid cells, even with a grid resolution of 5 cm.
2. **Zero Divergence**:
All GPU threads execute a single texture fetch instruction without branching, maximizing parallel computing throughput.

---

## 3. System Architecture and Processing Flow

```
[3D LiDAR (Livox MID360)]  ──> [GroundFilter (CUDA)] ──> [Obstacle Cloud]
                                    │                           │
                               [Ground Cloud]                   ▼
                                                  [ESDF Matcher (CUDA 3D Texture)]
[IMU (Yaw Integral Pred)] ───────────────────────────>          │
                                                               ▼
[3D ESDF Map (GPU Texture)] ──────────────────────> [ArgMin Pose Estimation]
                                                               │
                                  ┌────────────────────────────┴────────────────────────┐
                                  ▼                                                     ▼
                     [Estimated Pose (/platform_constraint)]            [Dynamic Point Cloud (/dynamic_cloud)]
                     [TF (map -> base_link)]

```

1. **Point Cloud Preprocessing (`ground_filter.cu`)**:
Excludes points within the robot's radius, clips valid height bounds, and separates flat floor (ground) point clouds.
2. **Pose Prediction**:
Integrates IMU angular velocity to predict rotational changes from the last confirmed pose and updates the search origin.
3. **Local Tracking / Global Search (`esdf_matcher.cu`)**:
Expands all candidate poses across the search range (XY / Yaw) onto a GPU grid, querying the ESDF texture to compute residual costs in parallel. Performs fine local refinement on the best solution.
4. **Inlier & Dynamic Obstacle Classification**:
Points within a proximity threshold to the map surface are classified as inliers to evaluate fitness. Isolated points exceeding the threshold are extracted as dynamic obstacles.

---

## 4. Build Environment and Dependencies

* **OS**: Ubuntu 22.04 LTS
* **ROS**: ROS 2 Humble
* **GPU**: NVIDIA GPU (Compute Capability 8.7 / 8.9: Jetson Orin / RTX 40 series, etc.)
* **CUDA Toolkit**: 12.0 or higher

### Installing Dependencies

```bash
sudo apt update
sudo apt install -y \
  libceres-dev \
  libeigen3-dev \
  nlohmann-json3-dev \
  libpcl-dev \
  libopenmpi-dev \
  ros-humble-pcl-conversions

```

### Building

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select gn10_pointcloud_localization
source install/setup.bash

```

---

## 5. Usage

### Placing Map Files

Place map files (`.pcd`, `.esdf`, `.json`) into the `map/` folder inside the package. They will be automatically referenced by filename in the parameter configuration.

```bash
# Example: Copying your custom point cloud map
cp my_field.pcd ~/ros2_ws/src/gn10-pointcloud-localization/map/

```

### Pre-converting PCD to ESDF (CLI)

Pre-building a 3D ESDF binary (`.esdf`) from a PCD file reduces node startup time to zero.

Relative paths are supported and will automatically search inside `map/`.

```bash
# Usage: pcd_to_esdf_converter <input.pcd> <output.esdf> <resolution_m> [max_dist_m]
ros2 run gn10_pointcloud_localization pcd_to_esdf_converter \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.pcd \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.esdf \
  0.05 0.50

```

If `map_source_type: "pcd"` is specified when launching the node, a `<filename>.esdf` cache will be generated automatically on the first run and loaded on subsequent launches.

### Launch

```bash
# Normal launch (default parameters: config/localization_params.yaml)
ros2 launch gn10_pointcloud_localization localization.launch.py

# When using simulation time (e.g., Rosbag playback)
ros2 launch gn10_pointcloud_localization localization.launch.py use_sim_time:=true

# Team presets (Red Zone / Blue Zone)
ros2 launch gn10_pointcloud_localization red.launch.py
ros2 launch gn10_pointcloud_localization blue.launch.py

```

---

## 6. Parameter Explanation

Primary settings are configured in [`config/localization_params.yaml`](https://www.google.com/search?q=./config/localization_params.yaml).

| Parameter Name | Default | Role / Mechanism |
| --- | --- | --- |
| `map_source_type` | `"json"` | Map type (`json`, `pcd`, `esdf`, `ros2_param`). |
| `map_file_path` | `""` | File name or path. Relative paths search inside `map/`. |
| `esdf.resolution` | `0.05` | 3D ESDF grid cell spacing [m]. Higher resolution captures finer protrusions. |
| `esdf.max_dist` | `0.50` | Truncation distance for the distance field [m]. Effective range for texture memory. |
| `scan_accumulation.window_s` | `0.10` | Point cloud accumulation window [s]. Motion-corrects points to the end-of-scan time before matching. 0 for single scan, max 0.5. |
| `scan_accumulation.timestamp_field` | `timestamp` | Acquisition timestamp field name for each point in PointCloud2. |
| `scan_accumulation.timestamp_scale` | `1.0e-9` | Scale factor to convert point timestamps to seconds. Matches Livox absolute nanosecond timestamps. |
| `scan_accumulation.timestamp_relative` | `false` | When true, treats point timestamps as relative time offsets from the header stamp. |
| `matching_params.search_range_xy` | `0.30` | Search range during local tracking [m] ($\pm 0.30\text{ m}$). |
| `matching_params.search_step_xy` | `0.05` | Grid step size during local tracking [m]. |
| `matching_params.search_range_yaw` | `0.60` | Rotation search range during local tracking [rad] ($\approx \pm 34^\circ$). |
| `matching_params.search_step_yaw` | `0.05` | Rotation step size during local tracking [rad] ($\approx 2.8^\circ$). |
| `matching_params.use_map_bounds` | `true` | Evaluates XY bounds from map headers during PCD/ESDF loading. Set to false to manually restrict bounds. |
| `matching_params.fine_refine` | `true` | Performs an additional 125 candidate micro-searches around the best solution to achieve sub-voxel precision. |
| `matching_params.fine_refine_levels` | `2` | Number of micro-search iterations. 2 steps refine an XY step of 0.05m down to 0.002m. |
| `matching_params.cost_threshold` | `0.165` | Average residual cost threshold for matching failure (increments lost count). |
| `matching_params.inlier_dist_thresh` | `0.08` | Classifies points within 8 cm of the map surface as inliers. |
| `matching_params.min_inliers` | `60` | Minimum required inlier count for a successful match. |
| `global_search.lost_count_thresh` | `10` | Number of consecutive failed frames before triggering a global full-area search. |

---

## 7. License

This repository is released under the [MIT License](https://www.google.com/search?q=./LICENSE).