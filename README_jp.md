# gn10-pointcloud-localization

[English](./README.md)

GN10 自律移動ロボット向けの GPU 高速化 3D + 2D ハイブリッド LiDAR 自己位置推定 ROS 2 パッケージです（Livox MID360 3D LiDAR + Lakibeam 1 2D 平面 LiDAR + 6 軸 IMU）。  
点群と IMU のみからグローバル初期位置同定（誘拐ロボット問題の解決）およびサブセンチメートル精度の安定したローカルトラッキングを実現します。

<img width="814" height="562" alt="image" src="https://github.com/user-attachments/assets/4254e8ba-a133-42b7-86b6-07034116f0f9" />

## 目次

1. [概要と特徴](#1-概要と特徴)
2. [3D ESDF & GPU 3D Texture Memory の仕組み](#2-3d-esdf--gpu-3d-texture-memory-の仕組み)
3. [システム構成と自己位置推定アルゴリズム](#3-システム構成と自己位置推定アルゴリズム)
   - [3D / 2D 双方向フォールバックと位置振動防止機構](#3d--2d-双方向フォールバックと位置振動防止機構)
   - [異常時の安全設計（TF 配信の即時停止）](#異常時の安全設計tf-配信の即時停止)
4. [ビルド環境・依存関係](#4-ビルド環境依存関係)
5. [使い方](#5-使い方)
   - [マップファイルの配置](#マップファイルの配置)
   - [PCD から ESDF への事前変換 (CLI)](#pcd-から-esdf-への事前変換-cli)
   - [起動 (Launch)](#起動-launch)
   - [RViz2 での ESDF マップ可視化](#rviz2-での-esdf-マップ可視化)
6. [パラメータ解説](#6-パラメータ解説)
7. [ドキュメント・設計図一覧](#7-ドキュメント設計図一覧)
8. [ライセンス](#8-ライセンス)

---

## 1. 概要と特徴

- **GPU 3D Texture Memory による $\mathcal{O}(1)$ スキャンマッチング**:  
  点群マップ（`.pcd`）や CAD 定義（`.json`）を **3D ESDF（Euclidean Signed Distance Field）** に変換し、NVIDIA GPU の **3D Texture Memory** にロード。k-d Tree 探索を行わず、ESDF テクスチャ参照とハードウェア・トライリニア補間により点あたり $\mathcal{O}(1)$ の定数時間で表面距離残差を評価します。
- **3D + 2D LiDAR 独立残差融合と振動防止フォールバック**:  
  3D LiDAR (`Livox MID360`) と 2D 平面 LiDAR (`Lakibeam 1`) の残差を GPU カーネル内で独立に評価し、視野角（FoV）スケーリングおよび **2 段の振動防止ガード機構** を用いて統合します。片方のセンサが遮蔽・劣化した際にも、位置飛びやチャタリングを起こさずに追従を継続します。
- **回転・並進歪み補正 (Deskew) と 3D/2D センサ間時刻同期**:  
  3D 蓄積点群（`0.10 s` 窓）および 2D スキャン（`time_increment`）の各ビームに対し、IMU 角速度積分（回転）とロボット局所速度 $v_{\text{local}}$（並進）による歪み補正を実施し、さらに 2D 点群を 3D 点群基準時刻 $t_{\text{3D}}$ へ時刻同期します。
- **グローバル自己位置同定 (Top-K NMS Kidnapped Robot Solver)**:  
  IMU の絶対 Yaw 姿勢を活用し、フィールド全域の候補を GPU で一括評価。非極大抑制（NMS）で抽出した上位 K 個の極小候補を精密ローカル探索にかけて自己位置を復帰します。
- **動的障害物・歩行者フィルタリング**:  
  静的マップ表面からの距離（`dynamic_dist_thresh = 0.15 m`）に基づいてロボット周囲の動的障害物や他機体をリアルタイムに分離し、`/dynamic_cloud` として配信します。

---

## 2. 3D ESDF & GPU 3D Texture Memory の仕組み

<img width="1007" height="711" alt="image" src="https://github.com/user-attachments/assets/cd8ddd3f-9a64-45c4-b60e-91cef41f2de9" />

### 従来の最近傍探索と 3D ESDF テクスチャ参照の比較
PCD 点群マップに対して直接 k-d Tree 最近傍探索を行うと、点あたり $\mathcal{O}(\log N_{\text{pcd}})$ の計算量がかかり、GPU スレッド間の分岐不一致（Warp Divergence）とランダムメモリアクセスによりフレームレートが低下します。  
本パッケージでは、`0.05 m` 解像度の 3D ボクセルグリッドに最近傍壁面距離を事前格納した ESDF を CUDA 3D Texture (`cudaTextureObject_t`) にバインドすることで:
1. **ハードウェア・トライリニア補間**: GPU 専用テクスチャユニットが 3 次線形補間をハードウェアで計算し、5 cm グリッド間を滑らかな連続距離としてサンプリングします。
2. **均一なテクスチャサンプリング**: 全 GPU スレッドが同一の `tex3D<float>()` 命令を実行するため、Jetson Orin Nano 上で数千通りの候補姿勢 $(x, y, \psi)$ を数ミリ秒で並列評価できます。

---

## 3. システム構成と自己位置推定アルゴリズム

```
[3D LiDAR (Livox MID360)] ──> [ScanAccumulator (0.10s Deskew)] ──> [GroundFilter (CUDA)] ──> [3D 障害物点群]
                                                                                                   │
[2D LiDAR (Lakibeam 1)]   ──> [LaserScanDeskew (回転+並進+時刻同期)] ──────────────────────> [2D 障害物点群]
                                                                                                   │
[IMU + EMA 並進速度外挿]  ─────────────────────────────────────────────────────────────────>       ▼
                                                                                    [ESDF Matcher (CUDA 3D Texture)]
[3D ESDF Map (GPU Texture)] ───────────────────────────────────────────────────────────────>       │
      │                                                                                            ▼
      ▼ (Transient Local)                                                   [3D/2D 振動防止フォールバック & 合格判定]
[ESDF点群 (/esdf_map)]                                                                             │
                                                                          ┌────────────────────────┴────────────────────────┐
                                                                          ▼ (合格: Accepted)                                ▼ (不合格: Rejected)
                                                         [TF 配信 (map -> base_link)]                      [TF・推定姿勢の配信を即時停止]
                                                         [/platform_constraint & /dynamic_cloud]           [localization_success = false]
                                                         [localization_success = true]                     [25フレーム連続失敗でGlobalSearch発動]
```

### 3D / 2D 双方向フォールバックと位置振動防止機構
[`PoseSolver::matchObstacleCloud`](src/pose_solver.cpp) では、3D 点群 ($N_{\text{3D}}$) と 2D 点群 ($N_{\text{2D}}$) の平均表面距離コスト ($C_{\text{3D}}, C_{\text{2D}}$) を独立に計算します。両方のセンサに十分な点数がある場合（$N_{\text{3D}} \ge 50, N_{\text{2D}} \ge 30$）、Combined モード（`w_2d = 0.40 * fov_scale`）、3D 単独モード（`w_2d = 0.0`）、2D 単独モード（`w_2d = 1.0`）の切り替えで位置飛びや振動が発生しないよう、以下の **2 段ガード機構** を備えています:

1. **ガード 1 — 2D 切り離し判定のヒステリシス不感帯 (`[0.04 m, 0.08 m)`)**:
   - 初回の Combined マッチングで 3D が合格（`pass_3d`: $C_{\text{3D}} < 0.21\text{ m}$）している場合、2D コスト $C_{\text{2D}}$ が通常の合格閾値 `cost_threshold_2d` (`0.04 m`) をわずかに超えただけでは 2D を切り離しません。
   - 2D を切り離して 3D 単独（`w_2d = 0.0`）で再計算するのは、`!pass_3d` **または** $C_{\text{2D}} \ge 2.0 \times$ `cost_threshold_2d` (`0.08 m`、2D の明確な異常・遮蔽) のときのみです。
   - これにより、$C_{\text{2D}} \in [0.04, 0.08)\text{ m}$ の不感帯では重み `w_2d = 0.40 * fov_scale` が連続的に保たれたまま `pass_3d` で合格となり、`0.04 m` 付近での重みチャタリング（微小振動）を防止します。
2. **ガード 2 — 2D 単独フォールバック時の局所連続性ウィンドウ制限 (`±0.10 m, ±0.06 rad`)**:
   - 3D 単独でも不合格（`!pass_3d`）となった場合は 2D 単独（`w_2d = 1.0`）を試行しますが、3D 点群が存在する状態（$N_{\text{3D}} \ge 50$）では、IMU＋速度予測姿勢 `search_base_pose` の周囲の探索範囲を $\min(r_{xy}, 2 s_{xy}) = \pm 0.10\text{ m}$、$\min(r_{\psi}, 2 s_{\psi}) = \pm 0.06\text{ rad}$（$r, s$ は探索半幅と刻み幅）に制限します。
   - これにより、3D が一時的に劣化したフレームでも 2D で滑らかに局所追従を継続しつつ、2D 単独が 20〜30 cm 離れた別の平行壁へジャンプして 3D 復帰時に揺り戻し振動を起こす現象を防止します（$N_{\text{3D}} < 50$ の完全な 3D センサ欠損時のみフル探索窓 $\pm 0.30\text{ m}$ を使用します）。

### 異常時の安全設計（TF 配信の即時停止）
- マッチングが `pass_cost` と `pass_inliers` の両方で不合格となったフレームでは、[`LocalizationNode`](src/localization_node.cpp) は **即座に `map -> base_link` TF および `/platform_constraint` の配信を停止** し、`/robot/operation/localization_success` に `false` を配信します。
- 未検証の予測姿勢（デッドレコニング）で TF を延長配信しないため、異常時には下位制御が確実に安全停止します。

---

## 4. ビルド環境・依存関係

- **OS**: Ubuntu 22.04 LTS
- **ROS**: ROS 2 Humble
- **GPU**: NVIDIA GPU (Compute Capability 8.7 / 8.9: Jetson Orin Nano / AGX Orin / RTX 40 シリーズ等)
- **CUDA Toolkit**: 12.0 以上

### 依存パッケージのインストールとビルド

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

## 5. 使い方

### マップファイルの配置
マップファイル（`.pcd`, `.esdf`, `.json`）はパッケージ内の `map/` フォルダに配置すると、YAML 設定からファイル名のみで自動参照できます。

### PCD から ESDF への事前変換 (CLI)
```bash
# 使用法: pcd_to_esdf_converter <input.pcd> <output.esdf> <resolution_m> [max_dist_m] [min_x max_x min_y max_y min_z max_z]
ros2 run gn10_pointcloud_localization pcd_to_esdf_converter \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.pcd \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.esdf \
  0.05 0.50 -5.5 5.5 -6.0 6.0 -0.2 4.0
```
`map_source_type: "pcd"` を指定して起動した場合も、初回起動時に `map/cache/<ファイル名>.esdf` キャッシュが自動生成され、2 回目以降は高速に読み込まれます。

### 起動 (Launch)
```bash
# 通常起動 (common_params.yaml + localization_params.yaml)
ros2 launch gn10_pointcloud_localization localization.launch.py

# Rosbag 再生などのシミュレーション時刻を使用する場合
ros2 launch gn10_pointcloud_localization localization.launch.py use_sim_time:=true

# 競技ゾーン別プリセット (赤ゾーン / 青ゾーン)
ros2 launch gn10_pointcloud_localization red.launch.py
ros2 launch gn10_pointcloud_localization blue.launch.py
```

#### 実機キャリブレーション済み静的 TF 既定値
3 つの launch ファイル（`red.launch.py`, `blue.launch.py`, `localization.launch.py`）はすべて実機合わせ込み済みの同一の静的 TF（`start_livox_tf:=true`, `start_lakibeam_tf:=true`）を配信します:
- **`base_link -> livox_frame`**: `x = 0.25 m, y = -0.25 m, z = 1.09 m, yaw = 0.0 rad, pitch = -0.273 rad, roll = 3.14159 rad`
- **`base_link -> lakibeam_frame`**: `x = -0.33 m, y = 0.0 m, z = 0.08 m, yaw = 3.13 rad, pitch = 0.0 rad, roll = 0.0 rad`

### RViz2 での ESDF マップ可視化
起動時に `/esdf_map` (`sensor_msgs/msg/PointCloud2`, `Transient Local` QoS) へ可視化用ボクセル点群が 1 回配信されます。
- **RViz2 設定**: `Topic: /esdf_map`, `Durability Policy: Transient Local`, `Color Transformer: Intensity`, `Style: Flat Squares (~0.04 m)`

---

## 6. パラメータ解説

共通設定は [`config/common_params.yaml`](./config/common_params.yaml) に定義されており、各プロファイル（`localization_params.yaml`, `red_params.yaml`, `blue_params.yaml`）で差分を上書きします。

| パラメータ名 | 既定値 (`common_params.yaml`) | 役割・メカニズム |
| :--- | :---: | :--- |
| `map_source_type` | `"json"` (red/blue は `"pcd"`) | マップ種別 (`json`, `pcd`, `esdf`, `ros2_param`)。 |
| `map_file_path` | `""` | マップファイル名またはパス（相対パスは `map/` 内を探索）。 |
| `esdf.resolution` | `0.05` | 3D ESDF グリッドのセル解像度 [m]。 |
| `esdf.max_dist` | `0.50` | ESDF 距離場の最大打ち切り距離 [m]。 |
| `esdf.crop_enabled` | `true` | PCD マップから ESDF を生成する際に指定境界ボックスでクロップ。 |
| `esdf.crop_min_x` / `crop_max_x` | `-5.5` / `5.5` | ESDF クロップ範囲の X 下限/上限 [m]。 |
| `esdf.crop_min_y` / `crop_max_y` | `-6.0` / `6.0` | ESDF クロップ範囲の Y 下限/上限 [m]。 |
| `esdf.min_z` / `max_z` | `-0.20` / `4.00` | ESDF クロップ範囲の Z 下限/上限 [m]。 |
| `scan_accumulation.window_s` | `0.10` | 3D LiDAR 点群の蓄積時間窓 [s]（スキャン末尾時刻へ運動補正）。 |
| `filter_params.ground_z_thresh` | `0.06` | `base_link` 基準の床面除去高さ閾値 [m]。 |
| `filter_params.robot_radius` | `0.80` | 自己機体除外半径 [m]（XY 平面）。 |
| `filter_params.robot_height_min` / `max` | `0.0` / `1.2` | 障害物として有効な高さ範囲 $z$ [m]。 |
| `filter_params.max_range` | `12.0` | センサの最大有効距離 [m]。 |
| `matching_params.search_range_xy` | `0.30` | ローカル追従時の XY 探索半径 [m]（2.0 m/s 以上の高速移動時は最大 1.5 倍へ適応拡大）。 |
| `matching_params.search_step_xy` | `0.05` | ローカル粗探索の XY グリッド刻み幅 [m]。 |
| `matching_params.search_range_yaw` | `0.15` | ローカル粗探索の基本 Yaw 探索半径 [rad]（約 $\pm 8.6^\circ$）。 |
| `matching_params.search_step_yaw` | `0.03` | ローカル粗探索の Yaw 刻み幅 [rad]（約 $1.7^\circ$）。 |
| `matching_params.fine_refine` / `levels` | `true` / `2` | 2 段の $5 \times 5 \times 5$ サブボクセル精密探索（`0.05 m` $\to$ `0.002 m`）。 |
| `matching_params.max_dist_thresh` | `0.25` | マッチング評価時の点ごとの ESDF 残差打ち切り距離 [m]。 |
| `matching_params.dynamic_dist_thresh` | `0.15` | `/dynamic_cloud` として抽出するマップ表面からの距離閾値 [m]。 |
| `matching_params.cost_threshold_3d` | `0.21` | 3D LiDAR 独立残差合格しきい値 [m] ($C_{\text{3D}} < 0.21\text{ m}$)。 |
| `matching_params.cost_threshold_2d` | `0.04` | 2D LiDAR 独立残差合格しきい値 [m] ($C_{\text{2D}} < 0.04\text{ m}$、ヒステリシス切り離しは $2\times = 0.08\text{ m}$)。 |
| `matching_params.inlier_dist_thresh` | `0.08` | マップ壁面から `0.08 m` (8 cm) 以内の点をインライアと判定。 |
| `matching_params.min_inliers` | `400` | インライア合格判定に必要な最小インライア点数 ($N_{\text{inlier}} \ge 400$)。 |
| `matching_params.inlier_cost_thresh` | `0.035` | インライア合格判定に必要な平均インライア残差上限 [m] ($C_{\text{inlier}} < 0.035\text{ m}$)。 |
| `matching_params.min_inlier_ratio` | `0.28` | 全障害物点に対する最小インライア比率 ($N_{\text{inlier}} / N_{\text{total}} \ge 0.28$)。 |
| `matching_params.max_inlier_accept_cost` | `0.25` | インライア合格判定時の全体合成コスト上限ガード [m] ($C_{\text{total}} < 0.25\text{ m}$)。 |
| `matching_params.weight_2d_lidar` | `0.40` | Combined モード時の 2D LiDAR 独立残差項の基本重み係数 $w_{\text{2D,base}}$。 |
| `matching_params.min_2d_points` / `target_2d_points` | `30` / `200` | 2D 残差有効化の最小点数と、100% 視野角重み付けとなる基準 2D 点数。 |
| `matching_params.min_3d_points` | `50` | 最小 3D 障害物点数（これ未満の場合はフル探索窓の 2D 単独モード $w_{\text{2D}} = 1.0$ へ移行）。 |
| `matching_params.dynamic_yaw_expansion` | `true` | $|\omega_z| > 0.50\text{ rad/s}$ の急旋回時に Yaw 探索窓を最大 `0.40 rad` まで動的拡大（係数 `0.15`）。 |
| `velocity_estimation.enable` | `true` | EMA 平滑化（$\alpha = 0.6$）つき並進速度推定と等速外挿の有効化（`window_sec: 0.20`, `min_dt_sec: 0.15`, `max_dt_sec: 0.50`, `max_linear_velocity: 4.0`）。 |
| `lidar_2d.enable` / `deskew` | `true` / `true` | Lakibeam 2D LiDAR の有効化と、回転・並進歪み補正および 3D 基準時刻への同期補正。 |
| `lidar_2d.time_offset_sec` | `0.0` | 2D LiDAR タイムスタンプの静的オフセット [s]。 |
| `global_search.top_k_candidates` | `4` | グローバル復帰探索時に NMS で抽出して精密化する上位候補数。 |
| `global_search.lost_count_thresh` | `25` | グローバル全域探索へ移行する連続失敗フレーム数。 |

---

## 7. ドキュメント・設計図一覧

主要なドキュメントおよび PlantUML 設計図の一覧です:

| ドキュメント / 設計図 | 日本語 | 英語 |
| :--- | :--- | :--- |
| **README** | [README_jp.md](./README_jp.md) | [README.md](./README.md) |
| **アルゴリズム・不変的メカニズム解説** | [docs/algorithm_and_mechanisms_jp.md](./docs/algorithm_and_mechanisms_jp.md) | [docs/algorithm_and_mechanisms.md](./docs/algorithm_and_mechanisms.md) |
| **コーディング規約・安全設計基準** | [docs/coding-rules_jp.md](./docs/coding-rules_jp.md) | [docs/coding-rules.md](./docs/coding-rules.md) |
| **全体パイプライン・パラメータ対応図 (PlantUML)** | [docs/pipeline_flow_and_parameters_jp.pu](./docs/pipeline_flow_and_parameters_jp.pu) | [docs/pipeline_flow_and_parameters.pu](./docs/pipeline_flow_and_parameters.pu) |
| **独立残差コスト・振動防止フォールバック図 (PlantUML)** | [docs/matching_cost_and_weighting_jp.pu](./docs/matching_cost_and_weighting_jp.pu) | [docs/matching_cost_and_weighting.pu](./docs/matching_cost_and_weighting.pu) |

---

## 8. ライセンス

本リポジトリは [Apache 2.0 ライセンス](./LICENSE) のもとで公開されています。
