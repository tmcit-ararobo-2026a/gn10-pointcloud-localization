# GN10 点群自己位置推定 パラメータチューニングガイド

[English Version](./parameter_tuning_guide.md)

本ドキュメントでは、`gn10-pointcloud-localization` の全体アーキテクチャ、各処理ステージのアルゴリズム、パラメータの物理的意味と作用、3D/2D 振動防止フォールバック機構、異常時の TF 安全停止設計、およびテレメトリを用いた実践的なチューニング手順を解説します。

---

## 1. アーキテクチャと処理フロー概要

自己位置推定パイプラインは以下の 4 つの主要ステージで構成されます:

1. **センサ入力と歪み補正（Deskew）前処理**:
   - **3D LiDAR (Livox MID360)**: ローリングバッファによる点群蓄積（`scan_accumulation.window_s = 0.10 s`）を行い、各点をスキャン末尾時刻 $t_{\text{3D}}$ へ IMU 角速度積分（回転）＋ロボット局所速度 $v_{\text{local}}$（並進）で運動補正した後、GPU `GroundFilter` で床面（`ground_z_thresh = 0.06 m`）と自己機体半径（`robot_radius = 0.80 m`）を除外します。
   - **2D LiDAR (Lakibeam 1)**: スキャン掃引中（`time_increment`）の各ビームの回転・並進歪みを補正するとともに、3D LiDAR 基準時刻 $t_{\text{3D}}$ との時刻差（`dt_sync = t_3d - t_2d`）を並進・回転補償して同一時刻座標系へ統合します。
2. **運動予測と探索窓の動的スケーリング**:
   - IMU 角速度積分（$\Delta\psi_{\text{imu}}$）と指数移動平均（EMA, $\alpha = 0.6$）平滑化つき並進速度外挿により、探索中心姿勢 `search_base_pose` を予測します。
   - 高速移動時（$\|\mathbf{v}\| \ge 2.0\text{ m/s}$）には XY 探索幅を最大 $1.5$ 倍まで拡大し、急旋回時（$|\omega_z| > 0.50\text{ rad/s}$）には Yaw 探索幅を最大 `0.40 rad` まで自動拡大します。
3. **GPU ESDF マッチングと 3D/2D 振動防止フォールバック (`PoseSolver::matchObstacleCloud`)**:
   - NVIDIA GPU の **3D Texture Memory** によるハードウェア・トライリニア補間を用いて点あたり $\mathcal{O}(1)$ で ESDF 距離場を参照し、粗探索後に 2 段の階層的サブボクセル精密化（`0.05 m` $\to$ `0.01 m` $\to$ `0.002 m`）を実行します。
   - 3D 点群と 2D 点群の平均表面距離残差（$C_{\text{3D}}, C_{\text{2D}}$）を独立に算出し、2 つのガード機構（ヒステリシス不感帯＋局所連続性ウィンドウ）によって 3D/2D 間の位置飛びや振動を起こさずに Combined / 3D 単独 / 2D 単独を切り替えます。
4. **安全出力ゲートと Top-K グローバル復帰探索**:
   - 厳格な合格判定（`pass_cost || pass_inliers`）を通過したフレームのみ姿勢を更新します。
   - **異常時の即時 TF 配信停止（暴走防止）**: マッチング不合格時は予測値による TF 外挿を一切行わず、`map -> base_link` TF および `/platform_constraint` の配信を即座に停止し、`/robot/operation/localization_success` に `false` を出力して下位制御を安全に停止させます。
   - 連続 `lost_count_thresh = 10` フレーム不合格が続いた場合、`GlobalSearcher` が非同期で全域 GPU 探索＋NMS Top-K 精密化を実行して復帰します。

### アーキテクチャ図 (PlantUML)
- **全体パイプラインとパラメータ対応図**:
  - [pipeline_flow_and_parameters_jp.pu](pipeline_flow_and_parameters_jp.pu) (日本語)
  - [pipeline_flow_and_parameters.pu](pipeline_flow_and_parameters.pu) (English)
- **独立残差コスト計算・3D/2D 振動防止フォールバック・合格判定フロー図**:
  - [matching_cost_and_weighting_jp.pu](matching_cost_and_weighting_jp.pu) (日本語)
  - [matching_cost_and_weighting.pu](matching_cost_and_weighting.pu) (English)

---

## 2. パラメータカテゴリ別リファレンス

共通の既定値は [`config/common_params.yaml`](../config/common_params.yaml) に定義されており、起動プロファイルに応じて [`config/localization_params.yaml`](../config/localization_params.yaml)、[`config/red_params.yaml`](../config/red_params.yaml)、[`config/blue_params.yaml`](../config/blue_params.yaml) で上書きされます。

### 2.1 マップ & 3D ESDF 生成 (`esdf.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `map_source_type` | `"json"` | string | マップ形式 (`"json"`, `"pcd"`, `"esdf"`, `"ros2_param"`)。`red_params.yaml` / `blue_params.yaml` では `"pcd"` を使用します。 |
| `map_file_path` | `""` | string | マップファイル名またはパス。相対パスはパッケージ内 `map/` フォルダから自動解決されます。 |
| `esdf.resolution` | `0.05` | double | 3D ESDF ボクセルグリッドの解像度 [m] (`0.05 m` = 5 cm)。小さくすると微細形状を捉えますが GPU メモリ消費が 3 乗に比例して増えます。 |
| `esdf.max_dist` | `0.50` | double | ESDF 距離場の最大打ち切り距離 [m]。 |
| `esdf.crop_enabled` | `true` | bool | PCD マップから ESDF を構築する際、指定した境界ボックスでクロップするかどうか。 |
| `esdf.crop_min_x / max_x` | `-5.5` / `5.5` | double | ESDF 生成領域の X 軸下限・上限 [m]。 |
| `esdf.crop_min_y / max_y` | `-6.0` / `6.0` | double | ESDF 生成領域の Y 軸下限・上限 [m]。 |
| `esdf.min_z / max_z` | `-0.20` / `4.00` | double | ESDF 生成領域の Z 軸（高さ）下限・上限 [m]。 |
| `esdf.publish_map` | `true` | bool | 起動時に `/esdf_map` (`transient_local` QoS) へ可視化用 ESDF ボクセル点群を 1 回配信します。 |
| `esdf.publish_max_distance` | `-1.0` | double | `/esdf_map` に含める最大表面距離 [m] (`-1.0` で `max_dist` 未満の全ボクセルを出力)。 |
| `esdf.publish_stride` | `1` | int | 可視化点群のボクセル間引きステップ (`1` = 全ボクセル、`2` = 1/8 間引き)。 |

---

### 2.2 3D 点群蓄積 (`scan_accumulation.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `scan_accumulation.window_s` | `0.10` | double | Livox 3D LiDAR の点群蓄積時間窓 [s]。Livox MID360 (10 Hz) では `0.10` (100 ms) に設定することで十分な空間密度を確保します。 |
| `scan_accumulation.timestamp_field` | `"timestamp"` | string | PointCloud2 内の各点の取得時刻フィールド名。 |
| `scan_accumulation.timestamp_scale` | `1.0e-9` | double | 点タイムスタンプを秒 [s] に変換する係数（ナノ秒表現の Livox では `1.0e-9`）。 |
| `scan_accumulation.timestamp_relative` | `false` | bool | 各点のタイムスタンプが絶対時刻の場合は `false`、`header.stamp` からの相対時刻の場合は `true`。 |

---

### 2.3 点群フィルタリング (`filter_params.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `filter_params.ground_z_thresh` | `0.06` | double | `base_link` 基準の床面除去高さ閾値 [m]。$z \le 0.06\text{ m}$ の点を床面として除外します。フィールドの継ぎ目や傾きが障害物として残る場合は微増させます。 |
| `filter_params.robot_radius` | `0.80` | double | 自己機体除外半径 [m] ($x^2 + y^2 \le r^2$)。バンパーや車輪・機構部からの反射点を除外します。 |
| `filter_params.robot_height_min` | `0.0` | double | 障害物として扱う最小高さ $z$ [m] (`base_link` 基準)。 |
| `filter_params.robot_height_max` | `1.2` | double | 障害物として扱う最大高さ $z$ [m] (`base_link` 基準)。天井や照明構造物を除外します。 |
| `filter_params.max_range` | `12.0` | double | センサの最大有効距離 [m]。遠距離のノイジーな反射点をカットします。 |
| `filter_params.max_points` | `200000` | int | GPU メモリに事前確保する最大点数バッファサイズ。 |

---

### 2.4 運動予測と並進速度推定 (`velocity_estimation.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `velocity_estimation.enable` | `true` | bool | フレーム間の等速直線運動外挿を有効化。急加速・高速移動時の探索窓遅れを防止します。 |
| `velocity_estimation.window_sec` | `0.20` | double | 並進速度の差分計算に使用する過去姿勢履歴の保持時間窓 [s]。 |
| `velocity_estimation.min_dt_sec` | `0.15` | double | 速度 $\Delta \mathbf{p} / \Delta t$ を算出する最小時間間隔 [s]。微小時間差分によるノイズ増幅を防ぎます。 |
| `velocity_estimation.max_dt_sec` | `0.50` | double | 速度外挿を許容する最大経過時間 [s]。これを超えてロストした場合は外挿をゼロにリセットします。 |
| `velocity_estimation.max_linear_velocity` | `4.0` | double | 推定並進速度の物理上限クランプ値 [m/s]。 |

算出された並進速度には指数移動平均（EMA, $\alpha = 0.6$）が適用され、単一フレームのマッチングジッタを平滑化します:
$$\mathbf{v}_k = 0.6 \, \mathbf{v}_{\text{raw}} + 0.4 \, \mathbf{v}_{k-1}$$

さらに、高速移動時（$\|\mathbf{v}\| \ge 2.0\text{ m/s}$）には XY 探索範囲 `search_range_xy` が速度に応じて最大 $1.5$ 倍（`0.30 m` $\to$ `0.45 m`）まで自動拡大されます:
$$\text{scale} = \min\left(1.5,\; 1.0 + 0.5 \times \frac{\|\mathbf{v}\| - 2.0}{v_{\max} - 2.0}\right)$$

---

### 2.5 ESDF ローカルマッチング・独立残差重み付け・3D/2D 振動防止フォールバック (`matching_params.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `matching_params.search_range_xy` | `0.30` | double | ローカル粗探索の XY 探索半径 [m]（予測姿勢を中心に $\pm 0.30\text{ m}$）。 |
| `matching_params.search_step_xy` | `0.05` | double | ローカル粗探索の XY グリッド刻み幅 [m]（`0.05 m` = $13 \times 13$ グリッド）。 |
| `matching_params.search_range_yaw` | `0.15` | double | ローカル粗探索の基本 Yaw 探索半径 [rad]（約 $\pm 8.6^\circ$）。 |
| `matching_params.search_step_yaw` | `0.03` | double | ローカル粗探索の Yaw 刻み幅 [rad]（約 $1.7^\circ$、11 ステップ）。 |
| `matching_params.dynamic_yaw_expansion` | `true` | bool | 急旋回時に角速度 $|\omega_z|$ に比例して Yaw 探索幅を自動拡大します。 |
| `matching_params.dynamic_yaw_omega_thresh` | `0.50` | double | 動的 Yaw 探索拡大を発動する角速度閾値 $|\omega_z|$ [rad/s]。 |
| `matching_params.dynamic_yaw_factor` | `0.15` | double | 角速度に乗じる拡大係数 [s]: $\text{range\_yaw} = \min(\text{max\_range},\; \text{base} + (|\omega_z| - \omega_{\text{thresh}}) \times 0.15)$。 |
| `matching_params.dynamic_yaw_max_range` | `0.40` | double | 動的 Yaw 探索幅の最大上限値 [rad]（約 $\pm 22.9^\circ$）。 |
| `matching_params.fine_refine` | `true` | bool | 粗探索の最良姿勢周辺で $5 \times 5 \times 5$ の階層的サブボクセル精密探索を実行します。 |
| `matching_params.fine_refine_levels` | `2` | int | 精密探索の反復段数（`2` 段でステップ幅を $1/25$ に細分化: `0.05 m` $\to$ `0.002 m`、`0.03 rad` $\to$ `0.0012 rad`）。 |
| `matching_params.max_dist_thresh` | `0.25` | double | マッチング評価時の点ごとの ESDF 距離打ち切り上限 [m]。動的障害物による外れ値勾配を抑制します。 |
| `matching_params.dynamic_dist_thresh` | `0.15` | double | 確定姿勢でのマップ表面距離がこの値 [m] を超える障害物点を `/dynamic_cloud` として抽出します。 |
| `matching_params.weight_2d_lidar` | `0.40` | double | 3D と 2D の両方が有効な場合の基本 2D 残差重み $w_{\text{2D,base}}$。 |
| `matching_params.min_2d_points` | `30` | int | 2D 残差項を有効化するために必要な最小 2D 障害物点数 ($N_{\text{2D}}$)。 |
| `matching_params.target_2d_points` | `200` | int | 視野角スケーリングが 100% になる基準 2D 点数: $\text{fov\_scale} = \min(1.0, N_{\text{2D}} / 200)$。 |
| `matching_params.min_3d_points` | `50` | int | 最小 3D 障害物点数 ($N_{\text{3D}}$)。これ未満の場合は即座に 2D 単独マッチング ($w_{\text{2D}} = 1.0$) に切り替わります。 |
| `matching_params.cost_threshold_3d` | `0.22` | double | 3D LiDAR 独立残差コストの合格閾値 [m] (`pass_3d`: $C_{\text{3D}} < 0.22\text{ m}$)。 |
| `matching_params.cost_threshold_2d` | `0.08` | double | 2D LiDAR 独立残差コストの合格閾値 [m] (`pass_2d`: $C_{\text{2D}} < 0.08\text{ m}$)。 |
| `matching_params.inlier_dist_thresh` | `0.08` | double | マップ表面から `0.08 m` (8 cm) 以内の障害物点をインライア（適合点）と判定します。 |
| `matching_params.min_inliers` | `400` | int | インライア合格条件に必要な最小インライア点数 ($N_{\text{inlier}} \ge 400$)。 |
| `matching_params.inlier_cost_thresh` | `0.035` | double | インライア合格条件に必要なインライア平均距離上限 [m] ($C_{\text{inlier}} < 0.035\text{ m}$)。 |
| `matching_params.min_inlier_ratio` | `0.28` | double | 全障害物点に対する最小インライア比率 ($N_{\text{inlier}} / N_{\text{total}} \ge 0.28$)。一部の壁のみの一致による誤ロックを防止します。 |
| `matching_params.max_inlier_accept_cost` | `0.22` | double | インライア合格判定時の合成コスト上限ガード [m] ($C_{\text{total}} < 0.22\text{ m}$)。 |

#### 独立 3D / 2D 残差コストの定義
各候補姿勢について、GPU カーネルは 3D 点群 ($N_{\text{3D}}$) と 2D 点群 ($N_{\text{2D}}$) の打ち切り付き ESDF 距離平均を独立に計算します:
$$C_{\text{3D}} = \frac{1}{N_{\text{3D}}} \sum_{i=1}^{N_{\text{3D}}} \min(d_i, d_{\max}), \qquad C_{\text{2D}} = \frac{1}{N_{\text{2D}}} \sum_{j=1}^{N_{\text{2D}}} \min(d_j, d_{\max})$$
$$C_{\text{total}} = (1 - w_{\text{2D}}) \, C_{\text{3D}} + w_{\text{2D}} \, C_{\text{2D}}, \qquad w_{\text{2D}} = w_{\text{2D,base}} \cdot \min\!\left(1.0, \frac{N_{\text{2D}}}{\text{target\_2d\_points}}\right)$$

#### 3D / 2D 間の位置ずれ・振動を防ぐ 2 段ガード機構 (`PoseSolver::matchObstacleCloud`)
3D LiDAR と 2D LiDAR の両方に十分な点数がある場合（$N_{\text{3D}} \ge 50$ かつ $N_{\text{2D}} \ge 30$）、わずかな外部パラメータ誤差や壁面の見え方の違いがある状態で Combined ($w_{\text{2D}} = 0.40$)、3D 単独 ($w_{\text{2D}} = 0.0$)、2D 単独 ($w_{\text{2D}} = 1.0$) を単純な閾値でハードスイッチすると、フレーム間で推定位置がチャタリング（振動）する恐れがあります。これを防ぐため、以下の **2 つの振動防止ガード** を実装しています:

1. **ガード 1 — 2D 切り離し判定のヒステリシス不感帯 (`[0.08 m, 0.16 m)`)**:
   - 初回の Combined マッチング ($w_{\text{2D}} = 0.40 \cdot \text{fov\_scale}$) において 3D コストが合格 (`pass_3d`: $C_{\text{3D}} < 0.22\text{ m}$) している場合、2D コスト $C_{\text{2D}}$ が通常の合格閾値 `cost_threshold_2d` (`0.08 m`) をわずかに超えただけでは 2D を切り離しません。
   - 2D を切り離して 3D 単独 ($w_{\text{2D}} = 0.0$) で再評価するのは、`!pass_3d` **または** $C_{\text{2D}} \ge 2.0 \times \text{cost\_threshold\_2d}$ (`0.16 m`、2D の明確な異常・遮蔽) の場合のみです。
   - これにより、$C_{\text{2D}} \in [0.08, 0.16)\text{ m}$ の軽微な 2D ずれ領域では重み $w_{\text{2D}} = 0.40 \cdot \text{fov\_scale}$ が連続的に維持されたまま `pass_3d` により合格となり、閾値境界での $w_{\text{2D}} = 0.40 \leftrightarrow 0.0$ チャタリングを完全に排除します。
2. **ガード 2 — 2D 単独フォールバック時の局所連続性ウィンドウ制限 (`±0.10 m, ±0.06 rad`)**:
   - 万一 3D 単独でも不合格 (`!pass_3d`) となった場合、2D 単独フォールバック ($w_{\text{2D}} = 1.0$) を試行します。
   - ただし 3D 点群が存在する状態 ($N_{\text{3D}} \ge 50$) では、2D LiDAR は高さ $z = 0.08\text{ m}$ の単一平面しか観測できないため、広い探索範囲（$\pm 0.30\text{ m}$）を与えると別の平行壁へ 20〜30 cm ジャンプし、次フレームで 3D が復帰した際に揺り戻し振動を起こす危険があります。
   - そのため、$N_{\text{3D}} \ge 50$ での 2D 単独フォールバック時は、IMU＋速度予測姿勢 `search_base_pose` の周囲の探索窓を **2 グリッドステップ以内** に厳格制限します:
     $$\text{range\_xy}_{\text{2D-only}} = \min(\text{range\_xy},\; 2 \cdot \text{step\_xy}) = \pm 0.10\text{ m}, \qquad \text{range\_yaw}_{\text{2D-only}} = \min(\text{range\_yaw},\; 2 \cdot \text{step\_yaw}) = \pm 0.06\text{ rad}$$
   - これにより、3D が一時的に劣化したフレームでも 2D で滑らかに局所追従を継続しつつ、遠方の誤った極小解へのジャンプと 3D 復帰時の位置振動を防止します（2D 単独でも $C_{\text{2D}} \ge 0.08\text{ m}$ で不合格の場合は 3D 単独の診断状態へ復元して棄却します）。

#### マッチング合格判定ロジック
各フレームのマッチングは以下のいずれかを満たした場合にのみ合格 (`match_accepted = true`) となります:
1. **主コスト判定 (`pass_cost`)**:
   - Combined モード（$C_{\text{2D}} < 0.16\text{ m}$）: $C_{\text{3D}} < 0.22\text{ m}$
   - 3D 単独モード（$w_{\text{2D}} = 0.0$）: $C_{\text{3D}} < 0.22\text{ m}$
   - 2D 単独モード（$w_{\text{2D}} = 1.0$）: $C_{\text{2D}} < 0.08\text{ m}$
2. **または 副インライア判定 (`pass_inliers`)** — 以下の 4 条件すべてを同時に満たす場合:
   - $N_{\text{inlier}} \ge 400$ (`min_inliers`)
   - $C_{\text{inlier}} < 0.035\text{ m}$ (`inlier_cost_thresh`)
   - $N_{\text{inlier}} / N_{\text{total}} \ge 0.28$ (`min_inlier_ratio`)
   - $C_{\text{total}} < 0.22\text{ m}$ (`max_inlier_accept_cost`)

---

### 2.6 2D 平面 LiDAR (`lidar_2d.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `lidar_2d.enable` | `true` | bool | Lakibeam 1 2D 平面 LiDAR 統合の有効化。 |
| `lidar_2d.deskew` | `true` | bool | スキャン掃引中の各ビームの回転歪み（IMU 積分）と並進歪み（$v_{\text{local}}$）を補正し、さらに 3D 点群基準時刻 $t_{\text{3D}}$ との時刻差を同期補正します。 |
| `lidar_2d.time_offset_sec` | `0.0` | double | `LaserScan.header.stamp` に加算する静的オフセット [s]。`Lakibeam_ROS2_Driver` 側で先頭ビーム時刻が正確に打刻されるため `0.0` のままにします。 |
| `lidar_2d.topic` | `"/lakibeam/scan"` | string | 入力 `LaserScan` トピック名。 |
| `lidar_2d.frame_id` | `"lakibeam_frame"` | string | メッセージの `frame_id` が空の場合のフォールバック TF フレーム名。 |
| `lidar_2d.tf_wait_sec` | `0.05` | double | `base_link` $\leftarrow$ `lakibeam_frame` TF 参照の待機タイムアウト [s]。 |

---

### 2.7 グローバル復帰探索 (`global_search.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `global_search.range_min_x / max_x` | `-5.25` / `5.25` | double | グローバル探索のフィールド X 範囲 [m]。 |
| `global_search.range_min_y / max_y` | `-5.70` / `5.70` | double | グローバル探索のフィールド Y 範囲 [m]。 |
| `global_search.range_yaw_diff` | `3.1415` | double | IMU 絶対 Yaw 姿勢を中心とした角度探索半径 [rad]（`red_params.yaml` / `blue_params.yaml` では `1.0 rad`）。 |
| `global_search.step_xy` | `0.30` | double | グローバル粗探索の XY グリッド解像度 [m]。 |
| `global_search.step_yaw` | `0.30` | double | グローバル粗探索の Yaw 解像度 [rad]。 |
| `global_search.downsample_stride` | `2` | int | グローバル粗探索時の障害物点群の間引きストライド。 |
| `global_search.top_k_candidates` | `4` | int | 非極大抑制（NMS: 距離 $1.0\text{ m}$ 以上または角度 $0.5\text{ rad}$ 以上）で抽出して精密ローカル探索にかける上位候補数。 |
| `global_search.lost_count_thresh` | `10` | int | グローバル復帰探索を発動するまでの連続失敗フレーム数（50 Hz 時に約 `0.2 s`）。 |

---

### 2.8 テレメトリ・診断出力 (`telemetry.*`, `topics.*`)
| パラメータ | 既定値 | 型 | 役割とチューニング指針 |
|---|---|---|---|
| `telemetry.enable` | `true` | bool | 点群コールバック周期（約 50 Hz）でのテレメトリ配信を有効化。 |
| `topics.output_telemetry` | `"/localization/telemetry"` | string | `LocalizationTelemetry` メッセージの出力トピック名。 |
| `topics.output_localization_success` | `"/robot/operation/localization_success"` | string | 自己位置推定の成否 (`std_msgs/msg/Bool`) を配信するトピック名。 |

---

## 3. テレメトリを用いた実践的チューニング手順

### 3.1 テレメトリの主要フィールド
`/localization/telemetry` (`gn10_pointcloud_localization/msg/LocalizationTelemetry`) には以下の情報が毎フレーム出力されます:
- `best_cost`: 合成マッチングコスト $C_{\text{total}}$ [m]
- `cost_3d`, `cost_2d`: 3D / 2D それぞれの独立平均残差コスト [m]
- `weight_2d`: 適用された 2D 重み $w_{\text{2D}}$（3D 単独時は `0.0`、Combined 時は `0.40 * fov_scale`、2D 単独時は `1.0`）
- `inlier_count`, `inlier_cost`, `inlier_ratio`: インライア点数 $N_{\text{inlier}}$、平均距離 $C_{\text{inlier}}$ [m]、インライア比率 $N_{\text{inlier}} / N_{\text{total}}$
- `obstacle_count_3d`, `obstacle_count_2d`, `obstacle_count_total`: 3D / 2D / 合計の障害物点数
- `linear_speed`, `velocity_x`, `velocity_y`: EMA 平滑化後の推定並進速度 [m/s]
- `search_range_xy`, `search_range_yaw`: 現在有効なローカル探索窓サイズ [m, rad]
- `match_accepted`, `is_lost`, `lost_frame_count`, `global_recovered`: 合格フラグ・ロスト状態・復帰フラグ

### 3.2 PlotJuggler または `scripts/plot_telemetry.py` による可視化
```bash
# PlotJuggler でリアルタイム監視:
ros2 run plotjuggler plotjuggler

# パッケージ同梱のリアルタイムプロッタ:
ros2 run gn10_pointcloud_localization plot_telemetry.py

# CSV 記録とオフライン画像出力:
ros2 topic echo /localization/telemetry --csv > telemetry.csv
python3 scripts/plot_telemetry.py --csv telemetry.csv --save telemetry_analysis.png
```

---

## 4. 動的パラメータ変更 (Dynamic Reconfigure)

実行中に変更可能なパラメータは `ros2 param set` で即座に反映できます:
```bash
ros2 param set /gn10_pointcloud_localization_node matching_params.weight_2d_lidar 0.40
ros2 param set /gn10_pointcloud_localization_node matching_params.cost_threshold_3d 0.22
ros2 param set /gn10_pointcloud_localization_node matching_params.cost_threshold_2d 0.08
```
トピック名、マップファイルパス、GPU バッファサイズなどの不変（Immutable）パラメータは実行時の変更から安全に保護されており、変更要求時にはエラーを返して拒否します。
