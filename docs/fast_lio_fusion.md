# MID360SのFAST-LIOオドメトリとフィールドマッチングの統合

## 構成

```
/livox/lidar (PointCloud2) ─┬─> GN10フィールドマッチャー ─> /platform_constraint_raw
                            └─> livox_pointcloud_bridge ─> FAST-LIO ─> /Odometry
/livox/imu ───────────────────────────────────────────────> FAST-LIO
/Odometry + /platform_constraint_raw ─> GN10姿勢フィルタ ─> /platform_constraint
                                                        └─> map → base_link TF
```

FAST-LIOの `camera_init → body` オドメトリを、IMU・LiDAR・`base_link` の外部パラメータを使って `base_link` の相対移動へ変換する。GN10の地図マッチング結果で平面位置とyawを補正する。フィルタが出す姿勢はマッチャーの次フレームの探索中心にも使う。

GN10の元ノードはこのlaunchでは `/platform_constraint_raw` を配信し、TFは配信しない。融合ノードだけが `map → base_link` を配信する。初回の有効な地図マッチングまで、融合ノードはmap姿勢を配信しない。マッチングが途切れた後もFAST-LIOのオドメトリが続く間は相対移動で姿勢を伝播するが、これは新しい地図観測ではない。オドメトリが1秒以上途切れるか、1フレームで大きく跳んだ場合は再初期化し、次の地図マッチングを待つ。

平面EKFは `x, y, yaw` を状態とし、FAST-LIOの連続する姿勢差で予測する。地図マッチングにはイノベーションゲートを設け、明らかな誤対応を採用しない。測定時刻が少し遅れて届いても、保持したオドメトリ履歴の近い時刻で補正し、後続の姿勢を再計算する。共分散とゲート値は初期値であり、実機・bagで調整が必要。

## FAST-LIOの導入

対象は [Ericsii/FAST_LIO_ROS2 の `ros2` ブランチ](https://github.com/Ericsii/FAST_LIO_ROS2/tree/ros2)。元の [hku-mars/FAST_LIO](https://github.com/hku-mars/FAST_LIO) はROS 1用。Livoxの `livox_ros_driver2` も同じROS 2ワークスペースでビルド・sourceする。

```bash
cd ~/ros2_ws/src
git clone --branch ros2 --recursive https://github.com/Ericsii/FAST_LIO_ROS2.git
cd ~/ros2_ws
colcon build --symlink-install --packages-select fast_lio gn10_pointcloud_localization
source install/setup.bash
```

`ros2 launch gn10_pointcloud_localization fast_lio_fusion.launch.py` で起動する。実機ではMID360Sのドライバから `/livox/lidar` と `/livox/imu` を配信する。FAST-LIOのROS 2版は `/Odometry` を配信し、既定では `camera_init` を親、`body` を子フレームとする。

決勝bagの点群は各点に `FLOAT64 timestamp`（Unix時刻のナノ秒値）を含む `PointCloud2`。同梱のbridgeはその値から `CustomMsg.offset_time` を生成し、FAST-LIOがdeskewに使う点時刻を保持する。単純にFAST-LIOのMID360用設定だけでbagの `/livox/lidar` を購読すると、メッセージ型が合わない。

bag再生では、bagに含まれる旧 `map → base_link` TFと `/platform_constraint` を再生しない。たとえば別ターミナルで次を実行する。

```bash
ros2 launch gn10_pointcloud_localization fast_lio_fusion.launch.py use_sim_time:=true
ros2 bag play '/home/lambda/bag_files/地区大会/rosbagjetson6/rosbag2_2026_09_20-16_25_27_0-004.db3' \
  --clock --start-offset 936.3 --topics /livox/lidar /livox/imu
```

既にFAST-LIOの `/Odometry` を別ノードが出している場合は `start_fast_lio:=false` にする。GN10のGPUマッチャーはCUDA対応GPU上で実行する。

## 調整と確認

- `config/fast_lio_mid360.yaml` の `mapping.extrinsic_T/R` と `config/fusion_params.yaml` の `imu_to_lidar.xyz/rpy` は同じIMU→LiDAR変換にする。掲載値はROS 2版FAST-LIOのMID360設定の初期値で、実機校正値ではない。
- `base_link → livox_frame` はlaunch内の静的TFと一致させる。FAST-LIOの `body` はIMUフレームであり、`base_link` と同一とは仮定しない。
- `/Odometry` の時刻、フレーム名、`/platform_constraint_raw`、`/platform_constraint`、`map → base_link` を確認する。FAST-LIOの `camera_init → body` TFを `map → base_link` と同じ親子名に変更しない。
- `fusion.match_xy_stddev`、`fusion.match_yaw_stddev`、プロセスノイズ、`fusion.innovation_gate` はbagの軌跡と誤復帰を確認して調整する。共通のLiDAR入力から得たFAST-LIOと地図マッチングには相関があるため、掲載共分散は統計的に校正された値ではない。
- この構成で連続してTFが出ても、地図マッチングのない区間の絶対位置が検証されたことにはならない。決勝bagの旧出力も真値ではない。

### 開発時の確認結果

決勝bagの `936.3–1116.3秒` を2倍速でFAST-LIOとbridgeに再生し、`/Odometry` を1,788件受信した。最初の出力は約936.88秒、最後は約1116.28秒で、最大出力間隔は約0.201秒だった。元のGN10マッチングTFの最長4秒の空白 `966.18–970.18秒` にも、FAST-LIOオドメトリは40件あった。これは相対移動の継続性を確認した結果であり、map座標の誤差を測定した結果ではない。GPUが使えない開発環境のため、GN10マッチャーを含む全系の再生と絶対精度の検証は未実施。
