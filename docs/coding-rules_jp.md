# コーディング規約

[English](coding-rules.md) | [日本語 (Japanese)](coding-rules_jp.md)

本規約はこのリポジトリ内のすべてのソースコード、設定ファイル、およびドキュメントに適用されます。

## 目次

1. [命名規則](#1-命名規則)
2. [制約](#2-制約)
3. [コメント・ドキュメント規約](#3-コメントドキュメント規約)

## 1. 命名規則

### 1.1 基本

- 名前と処理内容が一致していることを確認する。
- ネストされたループや複雑な走査では、`point_index`, `voxel_index` 等の意味のある名前を使用する。
- 物理量には単位を含める（例: `distance_m`, `window_sec`, `stamp_ns`）。
- 誤解されない名前を選ぶ。複数の意味を持つ単語には注意する。
- リポジトリ名は英小文字＋ハイフンで構成すること。
- ファイル名は英小文字＋アンダースコアで構成すること。

### 1.2 スタイル

| 対象 | 規則 | 例 |
| :--- | :--- | :--- |
| クラス / 構造体 | `PascalCase` | `PoseSolver`, `ESDFMap`, `YawHistory` |
| 関数 / 変数 | `snake_case` / `camelCase` | `get_angle_at()`, `target_velocity` |
| 定数 / 列挙値 | `ALL_CAPS` | `CYLINDER`, `VISUAL_BOX` |
| プライベートメンバ変数 | 末尾にアンダースコア `_` | `last_known_pose_`, `is_lost_` |

### 1.3 省略の許可と禁止

- 独自の短縮形（`pos`, `cnt`, `mgr` 等）の新規導入は避ける。
- 誤解を生まず広く知られる略語は許可する: `ESDF`, `SDF`, `IMU`, `TF`, `PCD`, `NMS`, `CUDA`, `GPU`, `FOV`, `EMA`, `PID`。
- クラス名の場合は `ESDFMap`, `IMUSample` のように略語部分を大文字とする。

## 2. 制約

### 2.1 共通

- 原則として Google C++ Style Guide に準拠する。
- `do-while` 文、`goto` 文の使用は禁止する。
- コンパイル時定数には `constexpr` を使用する。

### 2.2 組み込みシステム（STM32 / ESP32 等）

- 動的メモリの禁止: `new`, `malloc`, `std::vector`, `std::string` 等の使用を禁止し、`std::array` や固定長配列を使用する。
- コールバック内はフラグ操作のみに留める。
- オーバーヘッド回避のため、`try-catch` は使用しない。

### 2.3 ROS 2 / PC・Jetson 側（Linux）

- `std::unique_ptr` 等のスマートポインタと RAII を活用し、CUDA デバイスメモリ・ピン留めメモリはコンストラクタ／デストラクタで確実に管理する。
- 名前空間でスコープを明確にし、グローバル汚染を防ぐ。
- **ログ出力:** ROS 2 ノード内では `std::cout` ではなく `RCLCPP_INFO` 等の ROS ロガーを使用する。

## 3. コメント・ドキュメント規約

- **`gn10-pointcloud-localization` のソースコード（C++ / CUDA / ヘッダ / 設定ファイル / Launch / テスト）内のコメントはすべて英語で記述する。**
- 関数やカーネルには Doxygen 形式のコメントを付与し、どのようなアルゴリズムで自己位置推定が行われ、各パラメータがどう作用するかがコードから直感的に把握できるようにする。
- 5行以上の処理が続く場合は論理ブロックごとに分け、1行の要約コメントを追加する。
- **日英バイリンガルドキュメント:** 重要なドキュメント（`README.md` / `README_jp.md`、`docs/parameter_tuning_guide.md` / `docs/parameter_tuning_guide_jp.md`、`docs/coding-rules.md` / `docs/coding-rules_jp.md`、および PlantUML `.pu` 図）は英語と日本語を別ファイルで作成し、アルゴリズムやパラメータの変更時に両方を最新の状態に保つ。
