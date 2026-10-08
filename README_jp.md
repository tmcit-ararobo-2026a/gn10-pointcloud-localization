# gn10-pointcloud-localization

3D LiDAR（Livox MID360 等）を用いた自己位置推定 ROS 2パッケージです。  
オドメトリや初期位置の事前情報なしでも、点群とIMUのみからグローバル初期位置推定および安定したトラッキングを実現します。

## 目次

1. [概要と特徴](#1-概要と特徴)
2. [3D ESDF & GPU 3D Texture Memory の仕組み](#2-3d-esdf--gpu-3d-texture-memory-の仕組み)
3. [システム構成と処理フロー](#3-システム構成と処理フロー)
4. [ビルド環境・依存関係](#4-ビルド環境依存関係)
5. [使い方](#5-使い方)
   - [マップファイルの配置](#マップファイルの配置)
   - [PCD から ESDF への事前変換 (CLI)](#pcd-から-esdf-への事前変換-cli)
   - [起動 (Launch)](#起動-launch)
6. [パラメータ解説](#6-パラメータ解説)
7. [ライセンス](#7-ライセンス)

## 1. 概要と特徴

本パッケージは、フィールド環境（競技用フィールド、屋内搬送路など）におけるロボットのリアルタイム自己位置推定を目的として設計されています。

- **$\mathcal{O}(1)$ 高速スキャンマッチング**:  
  点群マップ（PCD）やCADオブジェクト定義を **3D ESDF（Euclidean Signed Distance Field）** に変換し、NVIDIA GPU の **3D Texture Memory** にロード。幾何計算やk-d Tree分岐探索を完全撤廃し、ハードウェア・トライリニア補間により点あたり $\mathcal{O}(1)$ の定数時間で正確な距離場を参照します。
- **グローバル自己位置同定**:  
  IMU の絶対 Yaw 姿勢を活用し、全域の XY/Yaw 探索（数千〜数万通りの候補姿勢）を GPU 並列リダクション（CUB DeviceReduce）で 1 フレーム内に一括評価。初期位置不明の状態からでも瞬時に自己位置を同定します。
- **動的障害物・歩行者フィルタリング**:  
  静的マップ表面からの距離に基づいて、ロボット周囲の動的障害物・人物・機体などをリアルタイムに分離し、`/dynamic_cloud` としてパブリッシュします。

## 2. 3D ESDF & GPU 3D Texture Memory の仕組み

### 従来の最近傍探索
PCD点群マップをそのまま用いて最近傍探索を行う場合、各点についてk-d Treeを探索するため計算量は $\mathcal{O}(\log N_{\text{pcd}})$ となります。  
GPU上ではスレッドごとの木探索パスの不一致分岐拡散やランダムメモリアクセスが発生し、フレームレートが著しく低下します。

### 3D ESDF + 3D Texture Memory
事前または起動時にマップ空間をボクセルグリッド化し、各ボクセルセルに「直近の壁・障害物表面までのユークリッド距離」を格納した 3D 距離場（ESDF）を構築します。

1. **ゼロ・オーバーヘッド補間**:  
   GPU の専用テクスチャユニットが3次線形補間をほぼ1クロックでハードウェア計算します。これにより、グリッド解像度が 5cm 刻みであっても、セル間を滑らかな連続値として正確にサンプリング可能です。
2. **Divergence ゼロ**:  
   すべての GPU スレッドが分岐なしにテクスチャフェッチ命令を 1 回実行するだけになり、GPU の並列演算性能を限界まで引き出せます。

## 3. システム構成と処理フロー

```
[3D LiDAR (Livox MID360)]  ──> [GroundFilter (CUDA)] ──> [Obstacle Cloud]
                                    │                           │
                               [Ground Cloud]                   ▼
                                                  [ESDF Matcher (CUDA 3D Texture)]
[IMU (Yaw 積分予測)] ───────────────────────────>          │
                                                               ▼
[3D ESDF Map (GPU Texture)] ──────────────────────> [ArgMin 姿勢判定]
      │                                                        │
      ▼ (起動時 1回配信: Transient Local)   ┌──────────────────┴──────────────────┐
[ESDF点群 (/esdf_map)]                      ▼                                     ▼
                               [推定自己位置 (/platform_constraint)]   [動的障害物点群 (/dynamic_cloud)]
                               [TF (map -> base_link)]
```

1. **点群前処理 (`ground_filter.cu`)**:  
   自己機体半径の除外、有効高さのクリッピング、および平坦な床面（グラウンド）点群の分離。
2. **姿勢予測**:  
   IMU 角速度積分により、前回確定姿勢からの回転変化を先読みして探索原点を更新。
3. **ローカル追従 / グローバル探索 (`esdf_matcher.cu`)**:  
   探索範囲（XY / Yaw）の全候補を GPU グリッド上に展開し、ESDF テクスチャを参照して残差コストを一括計算。最良解で精密ローカルリファインを実施。
4. **インライア & 動的障害物判定**:  
   マップ表面までの距離が近傍なしきい値以内の点を Inlier として適合度を判定。しきい値を超える孤立点を動的障害物として抽出。

---

## 4. ビルド環境・依存関係

- **OS**: Ubuntu 22.04 LTS
- **ROS**: ROS 2 Humble
- **GPU**: NVIDIA GPU (Compute Capability 8.7 / 8.9: Jetson Orin / RTX 40 シリーズ等)
- **CUDA Toolkit**: 12.0 以上

### 依存パッケージのインストール

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

### ビルド

```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select gn10_pointcloud_localization
source install/setup.bash
```

---

## 5. 使い方

### マップファイルの配置
マップファイル（`.pcd`, `.esdf`, `.json`）は、パッケージ内の `map/` フォルダに配置することで、パラメータ設定からファイル名だけで自動参照できます。

```bash
# 例: 自前の点群マップを配置
cp my_field.pcd ~/ros2_ws/src/gn10-pointcloud-localization/map/
```

### PCD から ESDF への事前変換 (CLI)
PCD ファイルから事前に 3D ESDF バイナリ（`.esdf`）を作成しておくことで、ノード起動時間をゼロに短縮できます。
パスは相対パスでも指定可能です。相対パスの場合は `map/` 内を探索します。

```bash
# 使用法: pcd_to_esdf_converter <input.pcd> <output.esdf> <resolution_m> [max_dist_m]
ros2 run gn10_pointcloud_localization pcd_to_esdf_converter \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.pcd \
  ~/ros2_ws/src/gn10-pointcloud-localization/map/my_field.esdf \
  0.05 0.50
```

ノード起動時に `map_source_type: "pcd"` を指定した場合、初回起動時に自動で `<ファイル名>.esdf` キャッシュが生成され、次回以降は自動でキャッシュが読み込まれます。

### 起動 (Launch)

```bash
# 通常起動 (標準パラメータ config/localization_params.yaml)
ros2 launch gn10_pointcloud_localization localization.launch.py

# Rosbag 再生などのシミュレーション時刻を使用する場合
ros2 launch gn10_pointcloud_localization localization.launch.py use_sim_time:=true

# チーム別プリセット (赤ゾーン / 青ゾーン)
ros2 launch gn10_pointcloud_localization red.launch.py
ros2 launch gn10_pointcloud_localization blue.launch.py
```

### RViz2 での ESDF マップ可視化

ノード起動後に ESDF マップが読み込まれると、`/esdf_map` (`sensor_msgs/msg/PointCloud2`) トピックへ**1回だけ**可視化用点群が配信されます。  
QoS に `transient_local` を採用しているため、ノード起動後に RViz2 を立ち上げても即座にマップ点群を受信できます。

- **用途**:
  - 読み込んだマップファイル（PCD/ESDF/JSON）の形状があっているかの確認
  - 3D LiDAR 障害物点群 (`/obstacle_cloud`) とマップ壁面が正しく一致できているかの視覚的検証
  - 探索範囲や解像度のチューニング時の材料
- **RViz2 表示設定**:
  - `Topic`: `/esdf_map`
  - `Durability Policy`: `Transient Local` (Latched)
  - `Color Transformer`: `Intensity` (ボクセルごとの壁面距離 [m] が格納されており、壁面 0m からグラデーション表示されます)
  - `Style`: `Flat Squares` (Size 0.04m 前後を推奨)

---

## 6. パラメータ解説

主要な設定は [`config/localization_params.yaml`](./config/localization_params.yaml) で行います。

| パラメータ名 | デフォルト | 役割・メカニズム |
| :--- | :---: | :--- |
| `map_source_type` | `"json"` | マップ種別 (`json`, `pcd`, `esdf`, `ros2_param`)。 |
| `map_file_path` | `""` | ファイル名またはパス。相対パスの場合は `map/` 内を探索。 |
| `esdf.resolution` | `0.05` | 3D ESDF グリッドのセル間隔 [m]。解像度を高めると微細な突起が再現可能。 |
| `esdf.max_dist` | `0.50` | 距離場の打ち切り距離 [m]。テクスチャメモリの有効レンジ。 |
| `esdf.min_z` | `-0.20` | フィールドオブジェクトから ESDF を生成する際の Z 下限 [m]。 |
| `esdf.max_z` | `2.00` | フィールドオブジェクトから ESDF を生成する際の Z 上限 [m]。 |
| `esdf.publish_map` | `true` | マップ読み込み完了後に ESDF を可視化用点群 (`/esdf_map`) として 1 回パブリッシュ。 |
| `esdf.publish_max_distance` | `-1.0` | 可視化する最大距離 [m] (-1.0 の場合は `max_dist` 未満の全ボクセル)。 |
| `esdf.publish_stride` | `1` | 可視化時のボクセル間引きステップ (1: 全ボクセル, 2: 1/8 に間引き)。 |
| `topics.output_esdf_map` | `"/esdf_map"` | 3D ESDF マップ可視化用点群トピック (QoS: Transient Local)。 |
| `scan_accumulation.window_s` | `0.10` | 点群の蓄積時間 [s]。各点をスキャン末尾時刻へ運動補正して照合。0で単一スキャン、最大0.5。 |
| `scan_accumulation.timestamp_field` | `timestamp` | PointCloud2の各点の取得時刻フィールド。 |
| `scan_accumulation.timestamp_scale` | `1.0e-9` | 各点の取得時刻を秒へ換算する係数。Livoxの絶対ナノ秒時刻に対応。 |
| `scan_accumulation.timestamp_relative` | `false` | trueの場合、各点の取得時刻を点群ヘッダーからの相対時刻として扱う。 |
| `matching_params.search_range_xy` | `0.30` | ローカル追従時の探索範囲 [m] (±0.30m)。 |
| `matching_params.search_step_xy` | `0.05` | ローカル追従時のグリッド刻み幅 [m]。 |
| `matching_params.search_range_yaw` | `0.60` | ローカル追従時の回転探索幅 [rad] (約 ±34°)。 |
| `matching_params.search_step_yaw` | `0.05` | ローカル追従時の回転刻み幅 [rad] (約 2.8°)。 |
| `matching_params.use_map_bounds` | `true` | PCD/ESDF読込時に地図ヘッダーのXY範囲を評価範囲に使用。意図的に範囲を制限する場合はfalse。 |
| `matching_params.fine_refine` | `true` | 最良解の周りでさらに 125 候補の微小探索を行い sub-voxel 精度を向上。 |
| `matching_params.fine_refine_levels` | `2` | 微小探索の反復回数。XY刻み0.05mなら2段で最終刻み0.002m。 |
| `matching_params.cost_threshold` | `0.165` | 平均残差がこの値を超えるとマッチング失敗判定 (ロストカウント加算)。 |
| `matching_params.inlier_dist_thresh`| `0.08` | マップ壁面から 8cm 以内の点を Inlier（適合点）と判定。 |
| `matching_params.min_inliers` | `60` | マッチング成立に必要な最小 Inlier 点数。 |
| `global_search.lost_count_thresh` | `10` | 連続で失敗判定となった際にグローバル全域探索へ移行するフレーム数。 |

---

## 7. ライセンス

本リポジトリは [MIT ライセンス](./LICENSE) のもとで公開されています。
