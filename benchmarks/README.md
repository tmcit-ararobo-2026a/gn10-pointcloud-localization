# Benchmark & Reproducibility Suite (`gn10-pointcloud-localization`)

All scripts and tools in this directory are licensed under **Apache-2.0** to keep the repository clean and permissive-license compliant.

## 1. Building Optional Microbenchmarks

By default, `colcon build` does **not** build benchmark binaries (`BUILD_BENCHMARKS=OFF`), keeping robot/Jetson runtime builds unchanged.

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
colcon build --packages-select gn10_pointcloud_localization --symlink-install \
  --cmake-args -DBUILD_BENCHMARKS=ON
```

Run the 5-way ESDF lookup microbenchmark (`CUDA tex3D`, `Global 8-fetch`, `Global Nearest`, `Analytic FieldSDF`, and optional `CPU OpenMP Trilinear`):

```bash
~/ros2_ws/install/gn10_pointcloud_localization/lib/gn10_pointcloud_localization/microbench_esdf_lookup \
  --map ~/ros2_ws/src/gn10-pointcloud-localization/map/nhk2026_map.json \
  --include-cpu \
  --json-out benchmarks/results/microbench.json
```

## 2. Running End-to-End Pipeline & Resource Harness

Optional Python dependencies (for plotting / pandas / YAML):

```bash
pip3 install -r benchmarks/requirements.txt
```

Run a trial (isolated localhost DDS so Wi-Fi never interferes):

```bash
python3 scripts/benchmark_harness.py run \
  --bag ~/rosbag_national_test_run_clean \
  --duration 50 \
  --label tracking_eval \
  --outdir benchmarks/results
```

## 3. Running Global Search & Safety Suite

```bash
python3 benchmarks/runners/run_global_search_eval.py \
  --bag ~/rosbag_national_test_run_clean \
  --out-json benchmarks/results/global_search_clean.json
```

To include Group C out-of-arena scenarios (`t >= 285 s`), pass the full bag containing out-of-arena segments (`~/rosbag_national_test_run`).

## 4. Running on Jetson Orin Nano via SSH (Zero Wi-Fi DDS Jitter)

Never play `ros2 bag play` over Wi-Fi DDS from a remote PC. Copy the trimmed `zstd` bags once to Jetson local storage and execute locally over SSH:

```bash
# 1. From host PC (once):
scp -r ~/rosbag_national_test_run_clean ~/rosbagjetson6_edited_clean jetson@<JETSON_IP>:~/

# 2. Inside Jetson SSH session (locked clocks for deterministic benchmark comparison):
sudo nvpmodel -m 2     # Select 25W / MAXN_SUPER mode (verify ID with `sudo nvpmodel -q`)
sudo jetson_clocks     # Lock GPU/CPU/EMC clocks for repeatable microbenchmark & latency p95/p99

export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=77
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash

python3 scripts/benchmark_harness.py run \
  --bag ~/rosbag_national_test_run_clean \
  --duration 120 \
  --label jetson_25w_locked \
  --outdir benchmarks/results
```

Also run a second pass with `jetson_clocks` disabled (reboot or default boot state) to record real-world `VDD_IN` wattage and thermal behavior via `tegrastats`.
