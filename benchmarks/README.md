# Benchmark & Reproducibility Suite (`gn10-pointcloud-localization`)

All scripts and tools in this directory are licensed under **Apache-2.0** to keep the repository clean and permissive-license compliant.

## 1. Building Optional Microbenchmarks

By default, `colcon build` does **not** build benchmark binaries (`BUILD_BENCHMARKS=OFF`), keeping robot/Jetson runtime builds unchanged.

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --packages-select gn10_pointcloud_localization --symlink-install \
  --cmake-args -DCMAKE_BUILD_TYPE=Release -DBUILD_BENCHMARKS=ON
```

Run the 5-way ESDF lookup microbenchmark (`CUDA tex3D`, `Global 8-fetch`, `Global Nearest`, `Analytic FieldSDF`, and optional `CPU OpenMP Trilinear`):

```bash
~/ros2_ws/install/gn10_pointcloud_localization/lib/gn10_pointcloud_localization/microbench_esdf_lookup \
  --map ~/ros2_ws/src/gn10-pointcloud-localization/map/nhk2026_map.json \
  --include-cpu \
  --json-out benchmarks/results/microbench.json
```

## 2. Running End-to-End Pipeline & Resource Harness

Optional Python dependencies (all permissive licenses; no GPL libraries imported):

```bash
pip3 install -r benchmarks/requirements.txt
```

Run a single trial (isolated localhost DDS so Wi-Fi never interferes):

```bash
python3 scripts/benchmark_harness.py run \
  --bag ~/rosbag_national_test_run_clean \
  --duration 50 \
  --label tracking_eval \
  --outdir benchmarks/results
```

## 3. Running Reproducible Ablation & Multi-Dataset Suite

Run all configured suites (`human_approach_3d`, `deskew_2d_ablation`, `full_national_tracking`, `opponent_match_edited`) and automatically generate Markdown, LaTeX, and TUM trajectory files:

```bash
python3 benchmarks/runners/run_ablation_suite.py \
  --outdir benchmarks/results/ablation_desktop
```

Evaluate any exported TUM trajectory against a reference TUM trajectory using the pure Apache-2.0 evaluator (and optionally pass the exported `.txt` files to external CLI tools):

```bash
python3 benchmarks/analysis/evaluate_trajectories.py evaluate \
  --est benchmarks/results/ablation_desktop/full_national_tracking__production_v122_3d2d_est_tum.txt \
  --ref benchmarks/results/ablation_desktop/full_national_tracking__production_v122_3d2d_ref_tum.txt
```

## 4. Running Global Search & Safety Suite

```bash
python3 benchmarks/runners/run_global_search_eval.py \
  --bag ~/rosbag_national_test_run_clean \
  --out-json benchmarks/results/global_search_clean.json
```

To include Group C out-of-arena scenarios (`t >= 285 s`), pass the full bag containing out-of-arena segments (`~/rosbag_national_test_run`).

## 5. Running on Jetson Orin Nano via SSH (Zero Wi-Fi DDS Jitter)

Never play `ros2 bag play` over Wi-Fi DDS from a remote PC. Copy the trimmed `zstd` bags once to Jetson local storage and execute locally over SSH:

```bash
# 1. From host PC (once):
rsync -av ~/rosbag_national_test_run_clean ~/rosbagjetson6_edited_clean ararobo@<JETSON_IP>:~/

# 2. Inside Jetson SSH session (optional clock lock on 25W MAXN_SUPER for deterministic microbenchmarks):
sudo nvpmodel -m 2     # Verify active mode ID with `sudo nvpmodel -q`
sudo jetson_clocks     # Lock GPU/CPU/EMC clocks for repeatable p95/p99

export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=77
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

python3 benchmarks/runners/run_ablation_suite.py \
  --suites full_national_tracking opponent_match_edited \
  --outdir benchmarks/results/jetson_25w
```

Also record a pass under default DVFS (`jetson_clocks` OFF) to measure real-world `VDD_IN` board power (`[W]`) and energy per frame (`[mJ/frame]`) via `tegrastats`.

## 6. External Ground-Truth & `.pcd -> .esdf` Benchmarks (Step 3)

Run exact physics ground-truth suites (e.g. `robot-sim-unity` publishing `/robot/ground_truth/pose` as `geometry_msgs/msg/PoseStamped`) and `.pcd -> .esdf` converted indoor datasets (`external_gt_matrix.yaml`):

```bash
# Unity physics GT (JSON vs PCD->ESDF)
python3 benchmarks/runners/run_ablation_suite.py \
  --matrix benchmarks/manifests/external_gt_matrix.yaml \
  --suites unity_physics_gt \
  --outdir benchmarks/results/unity_gt

# Unity cold-start & kidnap global search against exact physics GT
python3 benchmarks/runners/run_global_search_eval.py \
  --bag ~/rosbag_unity_nhk_dynamic_clean \
  --ref-topic /robot/ground_truth/pose \
  --manifest benchmarks/manifests/unity_gt_scenarios.yaml \
  --map-file ~/external_datasets/unity_nhk_scanned_map.esdf \
  --map-type esdf \
  --rate 1.0 \
  --out-json benchmarks/results/unity_gt/global_search_unity_esdf.json

# External indoor Velodyne .pcd -> .esdf dataset (HDL-501 leveled floor sequence)
python3 benchmarks/runners/run_ablation_suite.py \
  --matrix benchmarks/manifests/external_gt_matrix.yaml \
  --suites hdl_501_pcd_esdf \
  --outdir benchmarks/results/hdl501
```

## 7. External SOTA Baseline Comparisons & Fair 10 Hz / 0.10 s Protocol

To preserve strict `Apache-2.0` licensing inside `gn10-pointcloud-localization`, all external baseline repositories (`robotics-upo/dll` [BSD-3-Clause], `rsasaki0109/lidar_localization_ros2` + `ndt_omp_ros2` [BSD-2-Clause], `KOKIAOKI/3d_bbs` [MIT], and `octomap` [BSD]) are built outside this repository in `~/external_baselines_ws`.

Notice that Livox Mid-360 bags record 50 Hz packets (20 ms, ~4k pts/packet) whereas `gn10` internally accumulates a `0.10 s` window (~20k pts). To evaluate both raw packet input and genuinely fair `0.10 s` (`10 Hz`) scan input (with IMU preintegration + continuous-time deskewing enabled on `lidar_localization_ros2` and both IMU/constant-velocity deskewing tested on `DLL` / `KISS-ICP` / `3D-BBS`), generate the 10 Hz bags once with `benchmarks/tools/accumulate_livox_bag.py`:

```bash
# Always run unit tests with ROS 2 + workspace setup sourced so CUDA/test shared libs resolve cleanly
source /opt/ros/humble/setup.bash
source install/setup.bash
ctest --test-dir build/gn10_pointcloud_localization --output-on-failure -V

# Optional: sample CAD JSON geometry into PCD for external PCD/OctoMap baselines
python3 benchmarks/tools/json_map_to_pcd.py \
  map/nhk2026_map.json ~/external_datasets/nhk2026_cad_floor.pcd --include-floor

# Generate 10 Hz (0.10 s) accumulated + IMU-deskewed bags (strictly excludes GT/reference topics)
python3 benchmarks/tools/accumulate_livox_bag.py \
  --input ~/rosbag_unity_nhk_dynamic_clean \
  --output ~/external_datasets/rosbag_unity_nhk_dynamic_accum10hz --overwrite
python3 benchmarks/tools/accumulate_livox_bag.py \
  --input ~/rosbag_national_test_run_clean \
  --output ~/external_datasets/rosbag_national_test_run_clean_accum10hz --overwrite

# Trajectory tracking & CPU/RAM/power comparison (raw 20ms & fair 0.10s suites + HDL-501)
python3 benchmarks/runners/run_baseline_compare.py tracking \
  --external-ws ~/external_baselines_ws \
  --suites unity_physics_gt,unity_physics_gt_fair,hdl_501_eval,national_human_3d,national_human_3d_fair \
  --output-dir benchmarks/results/baseline_compare

# Global localization & out-of-arena safety rejection (single-packet & 0.10s accumulated 3D-BBS vs gn10)
python3 benchmarks/runners/run_baseline_compare.py global-compare \
  --external-ws ~/external_baselines_ws \
  --output-dir benchmarks/results/baseline_compare
```
