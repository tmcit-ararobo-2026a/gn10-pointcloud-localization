# gn10-pointcloud-localization

[English](./README.md)

CUDA 3D Texture Memory（`tex3D`）を用いた、ROS 2 Humble 向けの静的事前マップ平面 3-DoF $(x, y, \psi)$ LiDAR--IMU 自己位置推定パッケージです。
車輪オドメトリや初期位置入力なしに、事前生成した 3D 距離場へ点群を直接照合して自己位置を推定します。
NHK 高専ロボコン 2026 の自律移動・位置合わせに使用しました。

<img width="814" height="562" alt="RViz2 EDF 可視化" src="https://github.com/user-attachments/assets/4254e8ba-a133-42b7-86b6-07034116f0f9" />

## 1. 主な特徴

- **ハードウェア補間 3D EDF（`tex3D`）**：`.json` CAD 定義または `.pcd` 点群をオフラインで打ち切り非負表面ユークリッド距離場（$d \in [0, d_{\max}]$、`.esdf` float32 バイナリ保存）へ変換し、読み取り専用 CUDA 3D Texture Memory 上で $O(1)$ のハードウェア三線形補間を実行します。
- **GPU 前処理・ハッシュボクセル均一化**：`0.10 s` 蓄積した 3D LiDAR 点群を IMU ヨー角＋EMA 並進速度で歪み補正し、床面・自機を除去した上で、固定メモリ GPU 空間ハッシュ表（`262,144` スロット、`6.00 MiB` VRAM、`0.08 m` ボクセル）により近距離の $1/r^2$ 点密度偏りを抑えます。
- **Clipped $L_1$ 直接照合**：純粋な Clipped $L_1$ 残差（上限 `0.25 m`）を用いた粗密グリッド探索（最終 `2 mm` / `0.0012 rad`）と独立コスト閾値判定（`cost_3d < 0.23 m`）で姿勢を推定します。補助 2D LiDAR 融合・フォールバックにも対応します。
- **GPU グローバル再位置推定**：起動時や連続 `25` フレーム棄却時に同一の `tex3D` EDF を再利用して全域探索します（Top-4 3D NMS・Top-1/Top-2 比率ゲート `0.950`・`cost_3d < 0.202 m` による 3 フレーム連続検証）。
- **動的障害物分離**：事前マップに存在しない点群（$d \ge 0.15\,\text{m}$）を `/dynamic_cloud` として配信します。

https://github.com/user-attachments/assets/c095bf62-cfac-403c-aa1b-679e0ed4c053

## 2. 処理パイプライン概要

```text
[3D LiDAR (50 Hz)] ──> [0.10 s IMU歪み補正・フィルタ] ──> [GPUハッシュボクセル均一化 (0.08 m)] ──┐
[2D LiDAR (任意)  ] ──> [ビーム単位歪み補正・時刻同期] ───────────────────────────────────────────┤
[IMU + EMA 速度  ] ──> [姿勢予測] ───────────────────────────────────────────────────────────────▼
[静的 3D EDF     ] ──> [CUDA 3D Texture (tex3D)] ──> [粗密 Clipped L1 マッチャー (2 mm)]
                                                                   │
                                        ┌──────────────────────────┴──────────────────────────┐
                                        ▼ (cost_3d < 0.23 m)                                  ▼ (棄却)
                       [TF配信 (map -> base_link)]                           [TF・推定姿勢の配信停止]
                       [/platform_constraint & /dynamic_cloud]               [25フレーム連続失敗 -> GPU大域探索]
                                                                             [(Top-4 NMS + 比率ゲート + 3フレーム検証)]
```

## 3. 動作環境・ビルド

- **OS / ROS**：Ubuntu 22.04 LTS、ROS 2 Humble
- **GPU / CUDA**：NVIDIA GPU（Compute Capability 8.7 / 8.9：Jetson Orin Nano, RTX 4060 等）、CUDA Toolkit 12.0 以上

```bash
sudo apt update && sudo apt install -y \
  libeigen3-dev nlohmann-json3-dev libpcl-dev libopenmpi-dev ros-humble-pcl-conversions

cd ~/ros2_ws
colcon build --symlink-install --packages-select gn10_pointcloud_localization
source install/setup.bash

# 単体テスト
ctest --test-dir build/gn10_pointcloud_localization --output-on-failure -V
```

## 4. 使い方

`.json` / `.pcd` / `.esdf` マップを `map/` ディレクトリに配置します。`map_source_type: "pcd"` 指定時は初回起動時に `map/cache/<ファイル名>.esdf` へ自動キャッシュされます。

```bash
# 任意: .pcd から .esdf への事前変換
ros2 run gn10_pointcloud_localization pcd_to_esdf_converter \
  map/my_field.pcd map/my_field.esdf 0.05 0.50 -5.5 5.5 -6.0 6.0 -0.2 4.0

# 起動（実機 / rosbag再生）
ros2 launch gn10_pointcloud_localization localization.launch.py
ros2 launch gn10_pointcloud_localization localization.launch.py use_sim_time:=true
```

## 5. 主要パラメータ（`config/common_params.yaml`）

| パラメータ | 既定値 | 説明 |
| :--- | :---: | :--- |
| `map_source_type` / `map_file_path` | `"json"` / `""` | マップ形式（`json`, `pcd`, `esdf`, `ros2_param`）とパス（相対パスは `map/` 内を探索）。 |
| `esdf.resolution` / `max_dist` | `0.05` / `0.50` | 3D EDF グリッド解像度 [m] と打ち切り距離上限 [m]。 |
| `esdf.crop_enabled` | `true` | 指定境界ボックス（`crop_min/max_x,y`, `min/max_z`）でクロップ（CAD は $\pm d_{\max}$ パディング付与、クロップ済 PCD はなし）。 |
| `scan_accumulation.window_s` | `0.10` | 3D LiDAR スライディング蓄積窓 [s]。 |
| `filter_params.voxel_size` | `0.08` | GPU ハッシュボクセル重心均一化の辺長 [m]（`<= 0` で無効）。 |
| `filter_params.ground_z_thresh` | `0.06` | `base_link` 基準の床面除去高さ閾値 [m]。 |
| `filter_params.robot_radius` / `max_range` | `0.80` / `12.0` | 自機円柱除外半径 [m]（`z` は `[0.0, 1.2]`）と最大有効距離 [m]。 |
| `matching_params.search_range_xy` / `step_xy` | `0.30` / `0.05` | ローカル粗探索の XY 半幅と刻み幅 [m]。 |
| `matching_params.search_range_yaw` / `step_yaw` | `0.15` / `0.03` | ローカル粗探索の Yaw 半幅と刻み幅 [rad]。 |
| `matching_params.fine_refine` / `levels` | `true` / `2` | 2 段の $5 \times 5 \times 5$ 精密化（`0.05 m` $\to$ `0.002 m`, `0.0012 rad`）。 |
| `matching_params.max_dist_thresh` | `0.25` | 点ごとの Clipped $L_1$ 残差上限 [m]。 |
| `matching_params.cost_threshold_3d` / `2d` | `0.23` / `0.04` | 3D / 2D の独立平均残差合格閾値 [m]。 |
| `matching_params.dynamic_dist_thresh` | `0.15` | `/dynamic_cloud` 抽出の距離閾値 [m]。 |
| `matching_params.inlier_dist_thresh` | `0.08` | 壁面インライア距離閾値 [m]（品質診断テレメトリ専用）。 |
| `lidar_2d.enable` / `deskew` | `true` / `true` | 補助 2D LiDAR 融合（`weight_2d_lidar: 0.40`, `min_2d_points: 30`）と歪み補正。 |
| `velocity_estimation.enable` | `true` | EMA（$\alpha = 0.6$）並進速度予測（`window_sec: 0.20`, `max_linear_velocity: 4.0`）。 |
| `global_search.range_yaw_diff` / `recovery_yaw_diff` | `3.1415` / `0.40` | 全周探索時 / IMU 方位事前値有効時の Yaw 探索半幅 [rad]。 |
| `global_search.top_k_candidates` / `uniqueness_ratio` | `4` / `0.950` | NMS 上位候補数と異なる姿勢モード間の Top-1/Top-2 比率ゲート。 |
| `global_search.verify_frames` / `cost_threshold_3d` | `3` / `0.202` | 大域復帰確定に必要な連続合格フレーム数と 3D コスト閾値 [m]。 |
| `global_search.lost_count_thresh` | `25` | グローバル探索発動までの連続棄却フレーム数。 |

## 6. 設計図

![処理の流れとパラメータの作用](docs/pipeline_flow_and_parameters_jp.png)

![マッチング時のコスト評価](docs/matching_cost_and_weighting_jp.png)

## 7. ベンチマーク要約（`v1.4.0`、3D LiDAR + IMU、Clipped $L_1$）

Desktop（Ryzen 7 5700X + RTX 4060）および Jetson Orin Nano（`25 W` モード、`tegrastats` モジュール総入力 `VDD_IN`、アイドル `4.5 W`）での実測結果です。公開データセット（R1 アリーナ内・S1 Unity 6 物理真値）：<https://doi.org/10.5281/zenodo.24369541>、再現手順：[`benchmarks/README.md`](./benchmarks/README.md)。

| 評価項目 | Desktop (RTX 4060) | Jetson Orin Nano (`25 W`) |
| :--- | :--- | :--- |
| **EDF 参照マイクロベンチ（`240×260×104`）** | `tex3D` Glob8 比 `1.80×–3.38×`（解析的 SDF 比 `4.62×–8.65×`） | `tex3D` Glob8 比 `1.76×–3.44×`（解析的 SDF 比 `5.16×–11.5×`） |
| **R1 近接人物静止評価（`40–65 s`, 参照不要）** | $\sigma_{xy}$ `242.3 → 5.3 mm`（`-97.8%`）, 最大 `16.1 mm`, ジャンプ `0`（`0.55 ms`） | $\sigma_{xy}$ `238.5 → 5.2 mm`（`-97.8%`）, 最大 `16.9 mm`, ジャンプ `0`（`5.15 ms`, `5.09 W`） |
| **R1 フル走行（`20–280 s`, `50 Hz`）** | 平均/p95 `0.56 / 0.65 ms`（ロスト `0`, インライア残差 `25.0 mm`, ジッタ `5.9 mm`） | 平均/p95 `5.37 / 6.63 ms`（`5.09 W`, ロスト `0`, インライア残差 `25.0 mm`） |
| **S1 Unity 6 物理真値（`50 Hz` PCD / CAD）** | 平均 `12.7` / `18.8 mm`（RMSE `15.4` / `20.8 mm`, `0.58–0.59 ms`） | 平均 `13.4` / `18.6 mm`（RMSE `15.9` / `20.6 mm`, `5.43–5.90 ms`, `5.1 W`） |
| **大域再位置推定（S1 `N=8` / R1 `8+2`）** | S1 `8/8`（`5.15 ms`, `16.9 mm`）；R1 `7/8`（`6.99 ms`）, 負例 `2/2` 棄却（誤ジャンプ `0`） | S1 `8/8`（`34.8 ms`, `14.5 mm`）；R1 `7/8`（`60.8 ms`）, 負例 `2/2` 棄却（誤ジャンプ `0`） |
| **P1 公開 `hdl_501`（手持ち 6-DoF・IMU なし）** | `0.79 ms`（3-DoF RMSE `0.348 m` vs. 6-DoF NDT-OMP `0.045 m`） | `6.42 ms`（`4.79 W`, NDT/DLL/GICP 比 `3.9×–23.0×` 高速） |

**適用範囲・制約**：
- 平面 3-DoF $(x, y, \psi)$ 前提のため、IMU なしの手持ち上下動（`hdl_501`）では 6-DoF 手法に精度で劣ります。
- 密な 3D CUDA Array はマップ体積 $V$ に対し $O(V / r_{\text{grid}}^3)$ の VRAM を消費するため、数十メートル規模の静的屋内空間を対象とします。
- NVIDIA CUDA GPU が必須で、ホスト RSS は `168–223 MiB` 消費します。

## 8. ライセンス

[Apache 2.0 License](./LICENSE) のもとで公開されています。
