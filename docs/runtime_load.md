# 自己位置推定の処理負荷（2026-09-27）

## 測定条件

コード: 4e449eb。Core i7-13620H / RTX 4060 Laptop 8GB / RAM 32GB。
Jetson実機ではない。fast_lio_fusion.launch.pyで全ノードを起動し、決勝bagの
936.3秒からLiDAR/IMUを等倍再生した。既知初期姿勢(-4.74,-1.68,1.54)から
局所探索。RViz、点群可視化購読、他のロボット処理は含まない。
約190秒の再生中、1秒間隔でpsutilとnvidia-smiを測定し、開始5秒以降の
175サンプルを集計した。CPU 100%は論理CPU1個相当。RSSは共有ページを含む。

| ノード | CPU平均 | CPU p95 | RAM中央値 | RAM最大 |
|---|---:|---:|---:|---:|
| FAST-LIO | 36.46% | 39.5% | 160.11 MiB | 163.21 MiB |
| GN10マッチング | 8.26% | 9.6% | 130.57 MiB | 130.58 MiB |
| 融合 | 1.04% | 1.9% | 27.34 MiB | 27.36 MiB |
| Livox bridge | 2.90% | 3.9% | 25.73 MiB | 26.49 MiB |
| static TF | 0.03% | 0% | 24.00 MiB | 24.00 MiB |

5ノードのCPU平均合計は約48.7%（0.487論理CPU相当）、RSS中央値の合計は
約368 MiB。ROS launch、bag再生、OS、DDS外部プロセスはこの合計に含まない。
GN10のnvidia-smi GPUメモリは110 MiB、FAST-LIOはCUDAを使わない。
GPU全体の使用率は平均4.81%、p95 6%、1秒サンプル最大7%。
GPU使用率は他プロセスも含み、短時間のカーネル負荷ピークや処理時間ではない。

再生中の90秒購読では/Odometry、/platform_constraint_raw、
/platform_constraintは各899件、約9.99 Hz。受信間隔最大は約0.20秒だった。
一定速度での局所探索を確認したもので、全域探索の最悪負荷や遅延は未測定。
メッセージ時刻はスキャン開始/終了の違いがあるので、時刻差をカーネル実行時間と
扱わない。

## Orin Nano Superの判断

メモリ量は自己位置推定部分として小さく、動作の見込みはある。
ただし上のCPU/GPU使用率をOrinへ直接換算できず、実機の余裕は未確認。
現在のFAST_LIO_ROS2/CMakeLists.txtはx86でMP_ENと3スレッドを有効にする一方、
ARMではMP_ENを有効にせずMP_PROC_NUM=1になる。CPU負荷の移植時に注意する。
GN10はCUDA浮動小数点処理であり、67 INT8 TOPSを性能見積もりに使えない。

実機では25Wモード・冷却条件を記録し、決勝bag180秒と初期全域探索/ロスト復帰を
測る。tegrastatsのCPU、RAM、GR3D、EMC、温度と出力Hz・処理待ちの増加を確認する。
通常時と復帰時の両方を通し、実際に同時稼働する検出・制御も含めて余裕を判断する。
RVizは可能なら別PCで動かす。

公式仕様: [NVIDIA Orin Nano Super](https://www.nvidia.com/ja-jp/autonomous-machines/embedded-systems/jetson-orin/nano-super-developer-kit/)
