# gn10-pointcloud-localization

## 目次

1. [概要](#1-概要)
2. [ドキュメント](#2-ドキュメント)
3. [コントリビューション](#3-コントリビューション)
4. [ビルド・使い方](#4-ビルド使い方)
5. [システム構成](#5-システム構成)
6. [ライセンス](#6-ライセンス)

## 1. 概要
MID360S LiDARを用いた高専ロボコン向けの自己位置推定パッケージです。
フィールドが静的で平坦であることを前提としており、CUDAを用いた高速化が行われています。

## 2. ドキュメント

| ドキュメント | 説明 |
| :-: | :-: |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | 開発フロー・コミット規約・コーディング規約 |
| [docs/coding-rules.md](./docs/coding-rules.md) | コーディング規約の詳細 |
| [docs/uml/](./docs/uml/) | UML図 |

## 3. コントリビューション

[CONTRIBUTING.md](./CONTRIBUTING.md) を参照してください。

## 4. ビルド・使い方

必要なパッケージをインストール

```bash
sudo apt update
sudo apt install -y libceres-dev libeigen3-dev nlohmann-json3-dev
```

CUDAのパスを通す（CUDAが/usr/local/cudaにある場合）

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```



## 5. システム構成

<!-- ROS2ノード構成・STM32との通信方式・ハードウェア構成 等 -->
<!-- 依存クラスが3つ以上ある場合は docs/uml/ にUML図を作成し、ここにリンクする -->

## 6. ライセンス

本リポジトリは [MITライセンス](./LICENSE) のもとで公開されています。
