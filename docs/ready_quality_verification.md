# 契約v1のReady品質評価（T05追加）

2026-10-14 Asia/Tokyo、feat/12-localization-contract-v1、7798df1からの型・monitor・GPU raw・品質profile・試験差分。
作業ディレクトリ `/home/lambda/ros2_ws/src`、Humble setup。
`colcon build --packages-select gn10_pointcloud_localization --build-base /tmp/manager-v1/build --install-base /tmp/manager-v1/install --cmake-args -DBUILD_GPU_NODES=OFF`。
CPUビルド成功／終了コード0。install setup後、`ctest --test-dir /tmp/manager-v1/build/gn10_pointcloud_localization --output-on-failure`。
成功／終了コード0、品質と既存鮮度の2試験。profile／計測Pose-TF一致・fallback・閾値・版不一致・NaN・処理session変更・1秒・重複結果を実DDSで検証。
固定共分散はfalse。既定品質profileは未承認。承認の受理は専用localhost simulationだけ、実機E2は未承認。

初回・再ビルド失敗／終了コード2: Humble rclcpp::Timeにto_msg()がない。ヘッダの変換operatorへ合わせて修正。
ビルド失敗状態で実行した最初の品質試験は失敗／CTest8。古い生成Python型にposeがなくAttributeError。
修正ビルド後に新installをsourceし再試験成功。失敗ログquality-final-build.log／quality-tests.log、成功quality-repair-build.log／quality-repair-tests.logを/tmp/manager-v1に保持。
GPUビルド・実行・bag精度・真値・HIL・実機: この記録時点では未実施。以前のGPU成功を変更後の成功として扱わない。

時計逆行時は、受理済み結果番号を保持して計測時刻の比較基準とTF履歴をリセットする。低い新epochの新規結果を受け取り、品質安定区間を作り直す。古い結果番号は再採用しない。
managerのuse_sim_time試験で、時計停止・逆行の失効、1秒の品質再成立、明示reset後の復帰を確認した（manager-acceptance-tests.log）。実機時刻同期や物理性能の受入ではない。
