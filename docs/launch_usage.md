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

- Fixed Frame: map
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
