# LiDAR入力型の切り替え

起動時の`input_cloud_type`で購読型を選ぶ。実行中の変更は不可（read_only）。

| 設定 | 購読型 |
|---|---|
| `pointcloud2`（既定） | sensor_msgs/msg/PointCloud2 |
| `custom_msg` | livox_ros_driver2/msg/CustomMsg |

どちらもトピック名は`topics.input_cloud`で指定する。
通常のlocalization/red/blue.launch.pyではparams_file内の設定を使う。

```yaml
/**:
  ros__parameters:
    input_cloud_type: custom_msg
    topics:
      input_cloud: /livox/lidar
```

FAST-LIO統合launchでは次の引数で双方の入力を設定する。

```bash
ros2 launch gn10_pointcloud_localization fast_lio_fusion.launch.py \
  input_cloud_type:=custom_msg \
  input_cloud_topic:=/livox/lidar
```

この場合、GN10とFAST-LIOは同じCustomMsgトピックを直接購読し、
`livox_pointcloud_bridge`は起動しない。既定のpointcloud2では従来どおり
bridgeが入力をCustomMsgへ変換し、FAST-LIOは/livox/lidar_customを購読する。
start_fast_lio=falseではいずれの入力型でもbridge/FAST-LIOを起動しない。
統合launchの入力型/トピックは引数を使う（matcher_config内の同名設定より優先）。

CustomMsgはPointCloud2へ変換せず、XYZと点時刻を共通の内部データへ直接読み込む。
ヘッダー時刻をスキャン開始、offset_timeを相対nsとして、FAST-LIOと同じ基準で使う。
非ゼロのtimebaseはheader.stampとの誤差1ms以内であることを確認する。
point_numとpointsの不一致、200msを超える点時刻、空/不正点群は拒否する。
点時刻を保持するので、オドメトリによる歪み補正・局所マッチングは両入力で共通。
CustomMsg側もTFの準備とスキャン中のオドメトリ履歴を待つ。

RViz向けのフィルタ済み出力は引き続きPointCloud2。
CustomMsg型を使うGN10ノードもlivox_ros_driver2のメッセージ定義にビルド依存する。
FAST-LIO内部への補正書き戻しは行わない。

## 検証（2026-09-27）

- 同じXYZ・点時刻を持つPointCloud2/CustomMsgでデコード結果が一致するテストを追加。
- point_num不一致、timebase不一致、異常なoffset_timeを拒否するテストを追加。
- 決勝bagの12秒分を一時CustomMsg bagへ変換して再生。bridgeなしでFAST-LIO出力116件、
  GN10マッチ113件、融合112件。起動・履歴不足で5スキャンを破棄した。
- 元のPointCloud2 bagも既定launchで再生し、bridge、FAST-LIO、GN10、融合の出力を確認。
- CTest5件通過。ネイティブ実機ドライバのストリームは未検証。
