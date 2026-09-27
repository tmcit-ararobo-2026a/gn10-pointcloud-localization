# 自己位置推定とRVizの起動

## 自己位置推定のみ（FAST-LIO + GN10 + 融合）

```bash
ros2 launch gn10_pointcloud_localization fast_lio_fusion.launch.py
```

## 設定済みRVizも起動

```bash
ros2 launch gn10_pointcloud_localization fast_lio_fusion_rviz.launch.py
```

両方とも既定入力はPointCloud2、/livox/lidar。CustomMsgの場合は次のようにする。

```bash
ros2 launch gn10_pointcloud_localization fast_lio_fusion_rviz.launch.py \
  input_cloud_type:=custom_msg \
  input_cloud_topic:=/livox/lidar
```

自己位置推定のみのlaunchにも同じ引数を渡せる。
bag再生ではuse_sim_time:=trueを追加し、bagは別ターミナルから--clock付きで再生する。
既知初期姿勢、各設定ファイル、start_fast_lio等の引数は両launchで共通。
2つのlaunchを同時には起動しない（RViz付きlaunchが自己位置推定を含む）。
既存localization.launch.py/red.launch.py/blue.launch.pyはGN10単独起動として維持する。

## RVizの設定

- Fixed Frame: map（既定）。rviz_fixed_frameで別の基準フレームも指定できる。
- フィールド地図: /field_map_markers（Transient Local）
- 入力点群: PointCloud2。CustomMsg時は/gn10/rviz_cloudへ変換して表示。
- PointCloud2入力時は元入力トピックをそのまま表示し、表示用変換ノードは起動しない。
- 歪み補正済み障害物点群: /obstacle_cloud（高さによる色分け）
- 床面点群は設定に含むが既定では非表示。TFも表示する。

入力点群は生観測のプレビューで、表示用変換自体は歪み補正しない。
地図との位置合わせの確認にはDeskewed obstaclesを使い、必要なら入力点群表示を切る。
表示用変換はheader/frame_id、XYZ、反射強度、tag、line、offset_timeを保持する。
GN10/FAST-LIOはCustomMsgを直接使い、表示用PointCloud2を推定処理へ戻さない。
表示用変換は購読者がいるときだけ実行する。

RViz設定はrviz/fusion.rviz。rviz_config:=/絶対パス/設定.rvizで変更できる。

## 動作確認（2026-09-27）

CustomMsgの12秒分の再生で、RVizのOpenGL描画初期化、設定された地図・点群の購読、
表示用PointCloud2出力120件を確認した。XYZ・header・点時刻を保持するテストを含む
CTest5件も通過。PointCloud2既定入力ではRVizが/livox/lidarを直接購読することを確認。
Qt offscreenではOgreの描画窓を作れなかったため、通常のxcb表示モードで検証した。

## 09-03-walkで入力点群が出なかった原因と対応

このbagのCustomMsgはpoint_num/timebase/offset_timeが正しく、変換条件を満たす。
修正前のmap表示では、有効なmap→base_linkがない間、入力点群のTF待ちが続いて
RVizのメッセージキューがあふれ、点群を表示できなかった。
RVizの基準はmapを維持する。rviz_fixed_frameで基準フレームを明示指定できるが、
有効なmap→base_linkがない状態で正しいmap点群を表示することはできない。

```bash
ros2 launch gn10_pointcloud_localization fast_lio_fusion_rviz.launch.py \
  input_cloud_type:=custom_msg input_cloud_topic:=/livox/lidar use_sim_time:=true
ros2 bag play /home/lambda/bag_files/09-03-walk --clock --topics /livox/lidar /livox/imu
```

bagは約105.6秒でLiDAR500件。メッセージ時刻は記録時刻より13–14秒古く、
LiDARのヘッダー時刻間隔は中央値約100ms、最大約2.1秒の欠落がある。
GN10が移動履歴不足でスキャンを破棄することは、表示用変換の失敗とは別。
このbagで正しいフィールド自己位置が推定できることは未確認。

### 修正後の再生結果

同じ取り付け条件で09-03-walk全体を2倍再生した。RVizはmap表示を維持し、
use_sim_time=trueと--clockを使用した。表示用PointCloud2は365件、
地図マッチング処理228回（採用169回）、融合姿勢/map→base_linkは328件を記録。
修正前の最初20秒の再現では、表示用PointCloud2は59件出ていたがmap TFは0件だった。
比較区間が異なるため件数の比率を改善率とは扱わない。

この時点で確認した原因は、オドメトリ待ちが短すぎてマッチングへ点群が届かなかったこと。
その後のカクつき・位置飛びの検証は下記の取得時刻再生と検証記録を参照。
待機を既定2秒・30スキャン、履歴を5秒にした。揃えば待機上限を待たずに即処理する。
後から補えない既知の履歴欠落は即座に破棄し、後続の有効スキャンを処理する。
FAST-LIOの末尾点間引きによる100µs以内の終端差は端の姿勢で扱う。
それ以上の時刻欠落を外挿してmap TFを作ることはしない。
推定の欠落は残っており、連続したmap表示や正しい絶対位置を保証する結果ではない。

## 取得時刻でのbag再生

SQLite形式でHeaderが先頭のPointCloud2/CustomMsg/Imuを対象に、
メッセージの取得時刻順に配信する補助ツールを用意した。
09-03-walkでは通常のbag再生が記録時刻のバーストをそのまま再現するため、
比較には以下を使う。launchは上記と同じuse_sim_time=true・Fixed Frame=map。
通常のros2 bag playと同時には実行しない。

```bash
ros2 run gn10_pointcloud_localization sensor_stamp_play.py \
  /home/lambda/bag_files/09-03-walk
```

既定トピックは/livox/lidarと/livox/imu。--topics、--rateで変更できる。
各メッセージの内容・ヘッダー時刻は変えず、欠落も補間しない。
独立した/clockを取得時刻に合わせて配信する。sqliteは読み取り専用。
初期の全域探索が採用条件を満たさない間は、map基準で点群を表示できない。
取得時刻順再生だけではFAST-LIOの終盤ドリフトは解消しなかった。

初期位置が分かっている場合、FAST-LIOのオドメトリ開始後にRVizの
「2D Pose Estimate」で地図上の位置と向きを指定できる。/initialposeは融合ノードが受信し、
最新センサー時刻で明示的に地図位置を初期化する。以後は通常の局所マッチングに戻る。
この操作は位置の正しさを自動判定しない。オドメトリ異常中は操作を受け付けない。

通常速度での比較結果と残る制限は[09-03-walk検証記録](walk_bag_validation.md)を参照。

## 停止後にマッチングが再開しない問題の修正

表示TF停止時に事前姿勢も止めていた循環を修正した。
FAST-LIOの標準Odometryから作る探索用姿勢は/gn10/matching_priorへ継続配信する。
表示用の/platform_constraintとmap TFは、地図観測の年齢・オドメトリ健全性で制御する。
launchのコマンドとRVizのFixed Frame=mapは従来と同じ。
[復帰検証記録](map_recovery_validation.md)に09-03-gameとwalkの結果を記載。
