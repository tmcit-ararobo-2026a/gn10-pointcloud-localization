#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path

# ヘッダー情報の設定
COPYRIGHT_TEXT = "Copyright 2026 Gento Aiba and contributors"
SPDX_TEXT = "SPDX-License-Identifier: Apache-2.0"

# C/C++/CUDA 向けヘッダー (//)
HEADER_SLASH = f"// {COPYRIGHT_TEXT}\n// {SPDX_TEXT}\n\n"

# Python/CMake 向けヘッダー (#)
HEADER_HASH = f"# {COPYRIGHT_TEXT}\n# {SPDX_TEXT}\n\n"

# 対象とする拡張子・ファイル名とコメント形式のマッピング
TARGET_FILES = {
    ".cpp": HEADER_SLASH,
    ".hpp": HEADER_SLASH,
    ".cu": HEADER_SLASH,
    ".cuh": HEADER_SLASH,
    ".py": HEADER_HASH,
    "CMakeLists.txt": HEADER_HASH,
}

# 処理対象外とするディレクトリ名
EXCLUDE_DIRS = {
    ".git",
    "build",
    "install",
    "log",
    "rosbag",
    "map",
    "rviz",
    "docs",
}


def add_license_header(file_path: Path):
    header = TARGET_FILES.get(file_path.suffix) or TARGET_FILES.get(
        file_path.name
    )
    if not header:
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 既に著作権表示やSPDXタグが存在する場合はスキップ
        if "Copyright" in content or "SPDX-License-Identifier" in content:
            print(f"[SKIP (Already present)] {file_path}")
            return

        # Pythonファイル等で1行目に #! (Shebang) がある場合の挿入処理
        if content.startswith("#!"):
            first_line_end = content.find("\n") + 1
            new_content = (
                content[:first_line_end] + header + content[first_line_end:]
            )
        else:
            new_content = header + content

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)

        print(f"[UPDATED] {file_path}")

    except Exception as e:
        print(f"[ERROR] Could not process {file_path}: {e}")


def main():
    root_dir = Path.cwd()
    print(f"Scanning for source files in: {root_dir}\n")

    for root, dirs, files in os.walk(root_dir):
        # スキップ対象ディレクトリを除外
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]

        for file in files:
            file_path = Path(root) / file
            if (
                file_path.suffix in TARGET_FILES
                or file_path.name in TARGET_FILES
            ):
                add_license_header(file_path)


if __name__ == "__main__":
    main()