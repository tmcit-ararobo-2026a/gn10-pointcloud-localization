# gn10-pointcloud-localization

## 目次

1. [概要](#1-概要)
2. [ドキュメント](#2-ドキュメント)
3. [コントリビューション](#3-コントリビューション)
4. [ビルド・使い方](#4-ビルド使い方)
5. [システム構成](#5-システム構成)
6. [ライセンス](#6-ライセンス)

## 1. 概要
MID360S LiDARを用いた高専ロボコン向けの自己位置推定パッケージ。
フィールドが静的で平坦で、MID360Sの取り付け高さが固定であることを前提としており、CUDAを用いた高速化が行われている。
また、オドメトリ、初期位置の情報を用いずにMID360Sから得られるPointCloud2とIMUのみで自己位置推定を行うことができる。

動作環境：
Ubuntu 22.04
ROS2 Humble
CUDA 12.0以上

## 2. ドキュメント

| ドキュメント | 説明 |
| :-: | :-: |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | 開発フロー・コミット規約・コーディング規約 |
| [docs/coding-rules.md](./docs/coding-rules.md) | コーディング規約の詳細 |
| [docs/uml/](./docs/uml/) | UML図 |

## 3. コントリビューション

[CONTRIBUTING.md](./CONTRIBUTING.md) を参照してください。

## 4. ビルド・使い方

MID360SのFAST-LIOオドメトリとフィールドマッチングを統合する構成は
[docs/fast_lio_fusion.md](docs/fast_lio_fusion.md)を参照。

必要なパッケージをインストール

```bash
sudo apt update
sudo apt install -y libceres-dev libeigen3-dev nlohmann-json3-dev
```

CUDAのパスを通す(普通は通ってると思うが、私はパスを常時通すことを嫌うので毎回通すようにしている)
CUDAが/usr/local/cudaにある場合:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

rosdepで依存関係をインストール

```bash
rosdep update --rosdistro humble
rosdep install --from-paths . --ignore-src -y --rosdistro humble
```

ビルド

```bash
colcon build --symlink-install --package-select gn10_pointcloud_localization
```

読み込み

```bash
source install/setup.bash
```

実行

```bash
ros2 launch gn10_pointcloud_localization localization.launch.py
```

static tfを配信：

```bash
ros2 run tf2_ros static_transform_publisher --x 0.2 --y -0.25 --z 1.09 --yaw 0.0 --pitch -0.273 --roll 3.13 --frame-id base_link --child-frame-id livox_frame
```

## 5. システム構成

このパッケージは、以下の手順で自己位置推定を行う。
1. 点群のフィルタリングで半径12mを抽出
2. 平面抽出して床面を除く
3. フィールドの囲いと中央の教壇、各オブジェクトを一致させて自己位置を割り出し

## 6. ライセンス

本リポジトリは [MITライセンス](./LICENSE) のもとで公開されています。
