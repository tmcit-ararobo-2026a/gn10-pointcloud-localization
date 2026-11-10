# gn10-pointcloud-localization: 処理アルゴリズムと不変的メカニズム詳解

[English](./algorithm_and_mechanisms.md)

本ドキュメントは、`gn10-pointcloud-localization` の処理アルゴリズム、内部アーキテクチャ、数理モデル、および各パラメータがシステムにもたらす不変的な作用（数理的性質・制御安定性・制約境界）を詳解する仕様解説書です。本システムは、3D LiDAR (Livox MID360) と 2D 平面 LiDAR (Lakibeam 1) を GPU ESDF（ユークリッド符号付き距離場）上で統合し、高速旋回・高速移動下でも振動や発散を起こさずに安定した自己位置推定（`map -> base_link` TF）を維持するよう設計されています。

---

## 1. 全体アーキテクチャと設計思想

従来の点群自己位置推定（ICP や NDT など）では、動的障害物による外れ値、センサ間の微小な時間ずれや外部パラメータ誤差による相互干渉、急旋回時の探索窓遅れ、およびマッチング不良時の過度な予測外挿による暴走が課題となっていました。

本パッケージではこれらを解決するため、以下の不変原則に基づいてパイプラインを構築しています:
1. **空間標本化とハードウェア $\mathcal{O}(1)$ 距離場参照**: 事前構築した 3D ESDF を NVIDIA GPU の 3D Texture Memory に常駐させ、三線形補間によって任意座標の最近傍表面距離を $\mathcal{O}(1)$ で取得する。
2. **完全な時空間整合性（Spatiotemporal Rigidity）**: 3D 生点群および 2D 平面スキャンの全計測点を、IMU 角速度積分および並進速度外挿により単一の基準時刻 $t_{\text{ref}}$ におけるロボット座標系へ厳密に復元（Deskew）する。
3. **独立残差評価と 2 段ガードフォールバック**: 3D 点群と 2D 点群の平均距離残差を独立して評価し、ヒステリシス不感帯と狭小連続性ウィンドウにより、センサ間切り替えに伴う位置飛びやリミットサイクル振動を数理的に遮断する。
4. **安全停止（Fail-Safe）最優先の TF 配信制御**: マッチングが不合格となった場合、未検証な予測姿勢による TF 外挿を行わず即座に配信を完全停止し、下流の追従制御ノード側で安全な緊急停止を作動させる。

### アーキテクチャ図 (PlantUML)
- **全体パイプラインとパラメータ対応図**: [pipeline_flow_and_parameters_jp.pu](pipeline_flow_and_parameters_jp.pu)
- **コスト計算・3D/2D 振動防止フォールバック状態遷移図**: [matching_cost_and_weighting_jp.pu](matching_cost_and_weighting_jp.pu)

---

## 2. 前処理と時空間補正アルゴリズム

```
[Livox 3D LiDAR] ──> ScanAccumulator (window_s) ──> filterGroundWithDeskewKernel (CUDA)
[Livox 6-DoF IMU] ──> YawHistory (trapezoidal)  ───┘ (Rotational + Translational Deskew)
[Lakibeam 2D LiDAR] ─> deskewLaserScan ─────────> alignPointsToTimestamp (dt_sync)
                                                           │
                                                           ▼
                                                d_obstacle_ (CUDA unified memory)
```

### 2.1 3D 点群蓄積と二重運動歪み補正 (GPU Deskew)
3D LiDAR は 1 フレーム（100 ms）の掃引中にビーム照射角度を連続走査するため、ロボットの運動によって点群に幾何学的歪みが生じます。

- **タイムスタンプ抽出と基準時刻決定**:
  各点 $(x_i, y_i, z_i)$ のナノ秒タイムスタンプ $t_i$ を抽出し、スライディングウィンドウ `scan_accumulation.window_s` 内の最新点時刻を基準時刻 $t_{\text{ref}}$ と定めます。
- **IMU 角速度区分 2 次補間による回転補正**:
  GPU 上の `filterGroundWithDeskewKernel` において、各点時刻 $t_i$ から $t_{\text{ref}}$ までの微小回転 $\Delta \mathbf{R}(t_i \to t_{\text{ref}})$ を、高周期（200 Hz）IMU バッファの二分探索および台形則積分により算出します。
- **並進運動補正**:
  並進推定速度 $\mathbf{v}_{\text{local}}$ を用い、時間差 $\Delta t = t_{\text{ref}} - t_i$（$\pm 100\text{ ms}$ でクランプ）による並進変位 $\Delta \mathbf{p} = \mathbf{v}_{\text{local}} \cdot \Delta t$ を差し引きます。
- **不変的作用**:
  高速並進（$2\text{ m/s}$ 以上）や急旋回（$0.5\text{ rad/s}$ 以上）下でも、壁面や柱が歪んで厚みを持つ現象が完全に排除され、ESDF マップ表面との距離残差が真のセンサノイズ水準まで収束します。

### 2.2 幾何学的境界分離と床面除去 (`GroundFilter`)
歪み補正された点は、直ちに以下の幾何学的条件で分類されます:
- **ロボット自己除外境界**: $x^2 + y^2 \le r_{\text{robot}}^2$ かつ $z \in [z_{\min}, z_{\max}]$ を満たす点（バンパー、車輪、上部機構）を破棄。
- **最大有効距離境界**: $\|\mathbf{p}\| > d_{\max,\text{range}}$ のノイズ点を破棄。
- **床面判定**: $z \le z_{\text{ground}}$ を満たす点を床面（Ground）として分離し、障害物点群 $N_{\text{3D}}$ から除外。
- **不変的作用**:
  床面上の継ぎ目や微小な段差、自己機体反射点が障害物バッファに混入して鉛直・水平方向のマッチング勾配を歪めるのを幾何学的に遮断します。

### 2.3 2D LiDAR ビーム単位歪み補正と異種センサ時刻同期
2D 平面 LiDAR (Lakibeam 1, 30 Hz) の各ビームは照射角 $\theta_j$ に応じて掃引時刻が異なります。
- **ビーム単位の Deskew (`deskewLaserScan`)**:
  各ビーム時刻 $t_j = t_{\text{start}} + j \cdot \Delta t_{\text{scan}}$ における機体姿勢・並進変位を補正し、スキャン末尾時刻へ揃えます。
- **センサ間時刻同期 (`alignPointsToTimestamp`)**:
  3D 点群の基準時刻 $t_{\text{ref}}$ と 2D スキャン時刻 $t_{\text{2D}}$ の差 $\Delta t_{\text{sync}} = t_{\text{ref}} - t_{\text{2D}}$（$| \Delta t_{\text{sync}} | < 0.20\text{ s}$）に対して、IMU ヨー差分および並進速度射影を施し、同一の `d_obstacle_` バッファ（3D 障害物の直後に 2D 障害物を配置）へ格納します。
- **不変的作用**:
  異なる周期・異なるフレームで取得された 3D 点群と 2D 点群が、幾何学的にも時間的にも完全に一致した単一の剛体点群として GPU 上で同時評価可能になります。

---

## 3. 運動予測と探索窓の動的幾何モデル

### 3.1 探索中心姿勢の予測 (`search_base_pose`)
前回の確定姿勢 $P_{k-1} = (x_{k-1}, y_{k-1}, \psi_{k-1})$ から基準時刻 $t_{\text{ref}}$ への探索中心を予測します:
$$\psi_{\text{pred}} = \psi_{k-1} + \Delta \psi_{\text{imu}}(t_{k-1} \to t_{\text{ref}})$$
$$\mathbf{p}_{\text{pred}} = \mathbf{p}_{k-1} + \mathbf{v}_{\text{ema}} \cdot (t_{\text{ref}} - t_{k-1})$$
ここで $\mathbf{v}_{\text{ema}}$ は指数移動平均（EMA, $\alpha = 0.6$）により平滑化された機体並進速度です:
$$\mathbf{v}_k = \alpha \, \mathbf{v}_{\text{meas}} + (1 - \alpha) \, \mathbf{v}_{k-1}$$
- **不変的作用**:
  高周波な観測ジッタを EMA で抑制しつつ、急加速時にもマッチング探索中心がロボットの実位置に追従し、探索窓外への飛び出しを未然に防ぎます。

### 3.2 速度・角速度に応じた探索窓の動的拡大
ロボットの運動状態に応じて、候補姿勢を生成するグリッド探索領域を適応的に拡大します。

1. **並進速度連動スケーリング**:
   並進速度 $\|\mathbf{v}\| \ge 2.0\text{ m/s}$ のとき、XY 探索半径 $r_{xy}$ を最大 $1.5$ 倍まで線形拡大します:
   $$\text{scale}_{xy} = \min\left(1.5,\; 1.0 + 0.5 \times \frac{\|\mathbf{v}\| - 2.0}{v_{\max} - 2.0}\right), \qquad r_{xy} = r_{xy,\text{base}} \times \text{scale}_{xy}$$
2. **急旋回角速度連動スケーリング (`dynamic_yaw_expansion`)**:
   IMU ヨー角速度 $|\omega_z| \ge \omega_{\text{thresh}}$（$0.5\text{ rad/s}$）のとき、Yaw 探索半径 $r_{\psi}$ を自動拡大します:
   $$r_{\psi} = \min\left(r_{\psi,\max},\; r_{\psi,\text{base}} + (|\omega_z| - \omega_{\text{thresh}}) \cdot k_{\text{factor}}\right)$$
- **不変的作用**:
  低速・停止時には無駄な探索グリッドを増やさず計算負荷を最小化する一方、急制動や高速スピン時には運動の不確かさ楕円をグリッドで完全に包含し、局所解へのスタックやトラッキング喪失を回避します。

---

## 4. GPU 3D Texture ESDF マッチングと多段階探索

### 4.1 3D テクスチャによる $\mathcal{O}(1)$ ESDF 参照と外れ値クリッピング
探索グリッド内の候補姿勢 $P_k = (r_x, r_y, r_{\psi})$ に対し、CUDA カーネル `evaluateESDFKernel` が各点 $\mathbf{l}_i$ をワールド座標系へ投影します:
$$w_{x,i} = \cos(r_{\psi}) l_{x,i} - \sin(r_{\psi}) l_{y,i} + r_x$$
$$w_{y,i} = \sin(r_{\psi}) l_{x,i} + \cos(r_{\psi}) l_{y,i} + r_y$$
$$w_{z,i} = l_{z,i}$$

投影点 $(w_x, w_y, w_z)$ における最近傍壁面距離 $d_i$ は、NVIDIA GPU の 3D テクスチャハードウェアを用いて 8 ボクセル三線形補間により $\mathcal{O}(1)$ 時間でフェッチされます。
点単位の残差コスト $c_i$ は以下のように飽和クリッピングされます:
$$c_i = \begin{cases} \min(d_i, d_{\max}) & \text{if } (w_{x,i}, w_{y,i}) \in \text{FieldBounds} \\ d_{\max} & \text{otherwise} \end{cases}$$
ここで $d_{\max} = \text{max\_dist\_thresh}$ です。

- **不変的作用**:
  $d_i > d_{\max}$ となる遠方の外れ値点（歩行者や対戦ロボットなどの動的物体）の勾配寄与 $\frac{\partial c_i}{\partial P}$ は厳密にゼロとなり、一定値 $d_{\max}$ のペナルティのみを与えます（Huber / Welsch ロバスト推定と同等の効果）。これにより、動的物体が周囲に存在しても壁面引き込みの幾何学的引力圏が破壊されません。

### 4.2 3 段階 Coarse-to-Fine 階層探索の幾何学的収束性
探索は GPU 上の全候補並列評価と `cub::DeviceReduce::ArgMin` による最小値抽出を 3 段連続で行います:
1. **粗探索 (Coarse Level)**:
   - 探索幅: $\pm r_{xy}$, $\pm r_{\psi}$
   - グリッド刻み: $s_{xy}$ (`0.05 m`), $s_{\psi}$ (`0.03 rad`)
2. **精密探索 段 1 (Fine Level 1, 1/5x)**:
   - 粗探索最良姿勢を中心に $5 \times 5 \times 5$ グリッド（125 候補）を展開
   - 刻み幅: $s_{xy} / 5$ (`0.01 m` = 1 cm), $s_{\psi} / 5$ (`0.006 rad` $\approx 0.34^\circ$)
3. **精密探索 段 2 (Fine Level 2, 1/25x)**:
   - 段 1 最良姿勢を中心に再度 $5 \times 5 \times 5$ グリッドを展開
   - 刻み幅: $s_{xy} / 25$ (`0.002 m` = 2 mm), $s_{\psi} / 25$ (`0.0012 rad` $\approx 0.07^\circ$)

- **不変的作用 (空間標本化定理の充足)**:
  粗探索刻み幅 $s_{xy} = 0.05\text{ m}$ は、ESDF の引力圏半径 $d_{\max} = 0.25\text{ m}$ に対して $s_{xy} \ll d_{\max}$ を満たしており、局所盆地の最小点を取りこぼす（エイリアシングする）リスクが幾何学的に排除されています。また、全探索候補数を爆発させることなく、ミリメートル・サブ度単位の極限精度に確定時間で収束します。

---

## 5. 3D/2D 独立残差統合と振動防止フォールバック機構

### 5.1 独立残差コストの定式化と FoV スケーリング
3D 点群（点数 $N_{\text{3D}}$）と 2D 点群（点数 $N_{\text{2D}}$）について、平均距離残差を独立して集計します:
$$C_{\text{3D}}(P_k) = \frac{1}{N_{\text{3D}}} \sum_{i=0}^{N_{\text{3D}}-1} c_i, \qquad C_{\text{2D}}(P_k) = \frac{1}{N_{\text{2D}}} \sum_{j=N_{\text{3D}}}^{N_{\text{3D}}+N_{\text{2D}}-1} c_j$$

複合目的関数は動的重み $w_{\text{2D}}$ で結合されます:
$$C_{\text{total}}(P_k) = (1 - w_{\text{2D}}) \, C_{\text{3D}}(P_k) + w_{\text{2D}} \, C_{\text{2D}}(P_k)$$
$$w_{\text{2D}} = \text{weight\_2d\_lidar} \times s_{\text{fov}}, \qquad s_{\text{fov}} = \min\left(1.0,\; \frac{N_{\text{2D}}}{\text{target\_2d\_points}}\right)$$

- **不変的作用**:
  ロボットが障害物陰に入り 2D LiDAR の有効点数 $N_{\text{2D}}$ が減少した場合、$s_{\text{fov}} \to 0$ となり 2D 残差項の影響が自動的かつ連続的に減衰します。急激な離散重み切り替えによる勾配不連続が生じません。

```
                    ┌─────────────────────────┐
                    │ Step A: Combined (3D+2D)│
                    │ w_2d = weight * s_fov   │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┴─────────────────────┐
          │ pass_3d == true AND                       │ !pass_3d OR
          │ cost_2d < 2.0 * cost_threshold_2d         │ cost_2d >= 2.0 * cost_threshold_2d
          ▼                                           ▼
┌─────────────────────────┐               ┌─────────────────────────┐
│ NORMAL ACCEPTANCE       │               │ Guard 1: Trigger 3D-Only│
│ (No mode switch)        │               │ w_2d = 0.0 (Full Window)│
└─────────────────────────┘               └───────────┬─────────────┘
                                                      │
                                ┌─────────────────────┴─────────────────────┐
                                │ pass_3d == true                           │ !pass_3d AND N_3d >= min_3d
                                ▼                                           ▼
                      ┌───────────────────┐                       ┌─────────────────────────┐
                      │ 3D-ONLY ACCEPTED  │                       │ Guard 2: 2D-Only        │
                      │ (2D isolated)     │                       │ Tight Continuity Window │
                      └───────────────────┘                       │ (+/-2*step_xy/yaw)      │
                                                                  └─────────────────────────┘
```

### 5.2 ガード 1: 2D 切り離し判定のヒステリシス不感帯
3D と 2D にわずかな較正誤差がある場合、$C_{\text{2D}}$ がしきい値付近で微小振動すると、Combined モードと 3D 単独モードが毎フレーム交互に切り替わり、位置振動（チャタリング）が発生します。

本システムでは、3D が合格 (`pass_3d`) している限り、以下の**ヒステリシス不感帯**を適用します:
- **3D 単独再探索の発動条件**:
  `!pass_3d`（2D の悪影響で 3D が不合格に引きずり降ろされた場合）
  **または** $C_{\text{2D}} \ge 2.0 \times \text{cost\_threshold\_2d}$（2D の明確な異常・激しい遮蔽スパイク）
- **不変的作用**:
  $C_{\text{2D}} \in [\text{cost\_threshold\_2d},\; 2.0 \times \text{cost\_threshold\_2d})$ の領域では Combined 重みが維持されたまま `pass_3d` により合格判定され、モード遷移が抑止されます。シュミットトリガと同様のヒステリシス特性により、しきい値境界でのリミットサイクル振動が数理的に完全に排除されます。

### 5.3 ガード 2: 2D 単独フォールバック時の狭小連続性ウィンドウ
3D が完全に遮蔽され 3D 単独でも `!pass_3d` となった場合、2D 単独フォールバック（$w_{\text{2D}} = 1.0$）を実行します。

しかし、2D LiDAR は単一平面（高さ $z \approx 0.08\text{ m}$）の輪郭しか観測できないため、広い探索窓を与えると、縦横に並ぶ類似の平行壁や遠方の柱に誤ってマッチ（空間エイリアシング）し、位置が $20 \sim 30\text{ cm}$ テレポートするリスクがあります。次フレームで 3D が復帰すると元の位置へ引き戻され、巨大な位置跳躍と制御破綻を招きます。

これを防ぐため、$N_{\text{3D}} \ge \text{min\_3d\_points}$ が存在する状態での 2D 単独フォールバックでは、探索窓を**直前予測姿勢から 2 ステップ以内**に物理制約します:
$$r_{xy,\text{2D-only}} \le 2 \, s_{xy}, \qquad r_{\psi,\text{2D-only}} \le 2 \, s_{\psi}$$
- **不変的作用**:
  1 フレームあたりの最大可能変位が $2 \, s_{xy}$（`0.10 m`）および $2 \, s_{\psi}$（`0.06 rad`）に厳密にバウンドされます。2D 単独フォールバックが遠方の偽の壁へジャンプすることを物理的に遮断し、3D 復帰時にも位置の不連続跳躍・振動が原理的に発生しません。

---

## 6. 合否判定・安全 TF ゲーティングとグローバル復帰

### 6.1 二重合格判定ロジック (`isMatchAccepted`)
各フレームのマッチング結果は、以下のいずれかを満たした場合にのみ合格 (`match_accepted = true`) となります:

1. **主コスト合格判定 (`pass_cost`)**:
   - 3D 観測が有効（$N_{\text{3D}} \ge \text{min\_3d\_points}$）かつ $C_{\text{3D}} < \text{cost\_threshold}$
   - または、2D 観測が有効（$N_{\text{2D}} \ge \text{min\_2d\_points}$）かつ $C_{\text{2D}} < \text{cost\_threshold\_2d}$
2. **副インライア・コンセンサス判定 (`pass_inliers`)**:
   動的障害物や未登録物体が大量に存在する場合、壁面と完璧に一致していても平均距離 $C_{\text{3D}}$ が押し上げられて `pass_cost` を満たせないことがあります。この場合、以下の 4 条件を同時に満たせば真の壁面アライメントが成立していると判定します:
   - $N_{\text{inlier}} \ge \text{min\_inlier\_points}$ (十分な数の壁面近傍点が存在)
   - $C_{\text{inlier}} < \text{max\_inlier\_accept\_cost}$ (壁面近傍点自体の平均距離が極めて低残差)
   - $N_{\text{inlier}} / N_{\text{3D}} \ge \text{inlier\_ratio\_threshold}$ (全体の一定比率以上が壁面を構成)
   - $C_{\text{3D}} < \text{cost\_threshold}$ (全体のコストも一定の上限内)

- **不変的作用**:
  混雑環境（動的物体多数）での不要なマッチング棄却を防ぎつつ、壁面が一部しか見えていない状態での誤ロック（部分的アライメントによる偽陽性）をコンセンサス比率により完全に防ぎます。

### 6.2 安全最優先の TF 配信停止設計 (Zero Extrapolation on Failure)
マッチング判定が不合格となった場合の TF 配信制御は、ロボットの安全停止を最上位目的として厳密に設計されています:

- **単発の不合格（1〜2 フレーム）**:
  短時間（$\le 0.10\text{ s}$）に限り、直前の IMU 積分および速度推定による予測姿勢で TF を補間配信し、過度な制御ジッタを緩和します。
- **連続不合格（3 フレーム以上 または ロスト状態）**:
  **`map -> base_link` TF および `/platform_constraint` の配信を直ちに完全停止**し、`/robot/operation/localization_success` に `false` を出力します。
- **不変的作用**:
  自己位置推定が不確実な状態で予測外挿を継続すると、ロボットは推測航法（Dead Reckoning）の誤差を蓄積したまま全速前進し、暴走・激突に至ります。TF を遮断することにより、下流の軌道追従ノード（`path_follower_node`）側の `pose_age` タイムアウト監視機構を確実にトリガーし、ハードウェアとしての安全な緊急停止を作動させます。

### 6.3 3D NMS Top-K グローバル復帰探索 (`GlobalSearcher`)
連続 `lost_count_thresh` フレーム不合格が続くと、ノードは `is_lost_ = true` へ遷移し、非同期のグローバル探索を起動します:
1. **全域粗探索**:
   障害物点群を間引き（`downsample_stride`）、フィールド全域（`range_min/max_x`, `range_min/max_y`, `range_yaw_diff`）を粗い解像度（`step_xy`, `step_yaw`）で一括 GPU 評価。
2. **3D 非極大抑制 (3D NMS)**:
   空間距離（$\ge 0.60\text{ m}$）または角度差（$\ge 0.50\text{ rad}$）で分離された上位 `top_k_candidates`（既定 4 候補）の極小解ピークを抽出。
3. **フル解像度多段階精密化**:
   抽出された Top-K 候補周辺でフル解像度の `matchObstacleCloud` を実行し、最もコストの低い候補を採用。
- **不変的作用**:
  初期位置が未知の状態や長時間の遮蔽から復帰する際、単一の局所最小値に囚われることなく、対称的なフィールド形状から真の大域最小解を安定して特定します。

---

## 7. パラメータの不変的作用・物理的制約境界リファレンス

以下の表は、各パラメータがアルゴリズム内部で担う数理的役割と、過大／過小設定時に現れる不変的な振る舞いの境界を示します。

| パラメータ名 | 役割と数理的メカニズム | 過大設定時の不変的影響 | 過小設定時の不変的影響 |
|---|---|---|---|
| `esdf.resolution` | 3D ESDF ボクセルの空間刻み幅 [m] | 壁面・柱の微細形状が鈍化し、角度・位置の推定解像度が低下する | GPU テクスチャメモリ消費が 3 乗で増大し、VRAM 枯渇リスクが生じる |
| `esdf.max_dist` | ESDF 距離場の打ち切り飽和距離 [m] | 動的障害物やフィールド外の点が引力圏に入り、勾配が歪む | 引力圏（Basin of Attraction）が狭まり、粗探索で解を捉えられなくなる |
| `scan_accumulation.window_s` | 3D 点群蓄積ウィンドウ時間 [s] | 速度変化・加速度による非線形な運動歪みが増大し、Deskew 誤差が残る | 点群密度が低下し、ESDF 平均コストの分散が大きくなる |
| `filter_params.ground_z_thresh` | 床面点判定の高さ閾値 [m] | 低い障害物やフェンスの土台部分が欠落し、マッチング制約が弱まる | フィールドの継ぎ目や床面の微小な段差が障害物として残り、コストが上昇する |
| `filter_params.robot_radius` | 自己機体反射の幾何学的除外半径 [m] | ロボット近傍の有効な壁面・柱の点群まで除外されてしまう | バンパーや配線・機構部の反射点が混入し、自己位置が機体中心へ引きずられる |
| `velocity_estimation.enable` | 並進速度の等速直線運動外挿の有効化 | （無効時）高速走行時に探索中心が後方に置き去りにされ、探索窓外へ脱落する | （有効時）EMA 平滑化によりジッタを防ぎつつ探索中心の追従性を保証 |
| `matching_params.dynamic_yaw_expansion` | 急旋回時の角速度連動 Yaw 探索幅拡大 | （無効時）高速スピン時に前フレーム予測からの回転遅れが探索窓を超過しロストする | （有効時）急旋回時のみ必要十分な探索幅を動的確保し、通常時の GPU 負荷を抑制 |
| `matching_params.dynamic_yaw_omega_thresh` | 動的 Yaw 探索拡大を開始する角速度閾値 [rad/s] | 旋回が始まっても探索幅が拡大されず、旋回加速時の追従遅れを招く | 通常のわずかな旋回ノイズでも探索幅が拡大し、不要な GPU 負荷が生じる |
| `matching_params.max_dist_thresh` | 点単位の残差打ち切り上限 $d_{\max}$ [m] | 動的障害物や未登録物体の距離ペナルティが過大となり、合格判定が不当に棄却される | 適合壁面であってもわずかなノイズで上限に達し、コスト関数の識別能が低下する |
| `matching_params.weight_2d_lidar` | 2D LiDAR 残差項の基本結合重み $w_{\text{2D,base}}$ | 2D LiDAR の外部パラメータ誤差やピッチ揺れが 3D 全体の姿勢推定を歪める | 2D LiDAR の高周波・高精度な水平壁面拘束が活かされず、水平位置ドリフトが増加する |
| `matching_params.cost_threshold` | 3D LiDAR 独立残差の合格判定閾値 [m] | 誤った壁面や傾いた姿勢でも合格と判定され、位置飛びのリスクが高まる | センサノイズや軽微な遮蔽で即座に不合格となり、TF 停止が頻発する |
| `matching_params.cost_threshold_2d` | 2D LiDAR 独立残差の合格判定閾値 [m] | 2D が誤った壁にロックしている場合でも合格と誤認される | ガード 1 の発動頻度が高まり、不要な 3D 単独再探索が発生する |
| `matching_params.min_inlier_ratio` | 合格判定に必要な最小インライア比率 | 障害物が多い環境で条件が厳しすぎ、正当なアライメントが棄却される | 一部の壁のみに一致した不完全な拘束（自由度縮退）を誤って合格とみなす |
| `global_search.lost_count_thresh` | グローバル探索へ移行する連続不合格フレーム数 | ロスト状態が長時間放置され、復帰が大幅に遅れる | 一時的な遮蔽や急激な加減速による単発の不合格で誤ってグローバル探索が走る |
