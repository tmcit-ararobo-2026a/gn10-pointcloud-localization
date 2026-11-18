#!/usr/bin/env python3
# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0
"""Convert nhk2026_map.json geometric primitives into a uniform surface-sampled PCD."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

import numpy as np


def sample_box_surface(
    cx: float,
    cy: float,
    hx: float,
    hy: float,
    z_min: float,
    z_max: float,
    step: float,
) -> list[tuple[float, float, float]]:
    """Sample all 6 faces (plus outer perimeter density) of an axis-aligned box."""
    pts: list[tuple[float, float, float]] = []
    xs = np.linspace(cx - hx, cx + hx, max(2, int(round((2.0 * hx) / step)) + 1))
    ys = np.linspace(cy - hy, cy + hy, max(2, int(round((2.0 * hy) / step)) + 1))
    zs = np.linspace(z_min, z_max, max(2, int(round((z_max - z_min) / step)) + 1))

    # Vertical side faces (+/- y and +/- x)
    for x in xs:
        for z in zs:
            pts.append((float(x), float(cy - hy), float(z)))
            pts.append((float(x), float(cy + hy), float(z)))
    for y in ys:
        for z in zs:
            pts.append((float(cx - hx), float(y), float(z)))
            pts.append((float(cx + hx), float(y), float(z)))
    # Top face
    for x in xs:
        for y in ys:
            pts.append((float(x), float(y), float(z_max)))
    return pts


def sample_cylinder_surface(
    cx: float,
    cy: float,
    radius: float,
    z_min: float,
    z_max: float,
    step: float,
) -> list[tuple[float, float, float]]:
    """Sample vertical mantle and top cap of a vertical cylinder."""
    pts: list[tuple[float, float, float]] = []
    n_theta = max(12, int(math.ceil(2.0 * math.pi * radius / step)))
    zs = np.linspace(z_min, z_max, max(2, int(round((z_max - z_min) / step)) + 1))
    thetas = np.linspace(0.0, 2.0 * math.pi, n_theta, endpoint=False)
    for th in thetas:
        x = cx + radius * math.cos(float(th))
        y = cy + radius * math.sin(float(th))
        for z in zs:
            pts.append((x, y, float(z)))
    # Top cap
    n_r = max(1, int(round(radius / step)))
    for ir in range(n_r + 1):
        rr = radius * ir / max(1, n_r)
        nt = 1 if ir == 0 else max(6, int(math.ceil(2.0 * math.pi * rr / step)))
        for it in range(nt):
            th = 2.0 * math.pi * it / nt
            pts.append((cx + rr * math.cos(th), cy + rr * math.sin(th), float(z_max)))
    return pts


def write_binary_pcd(path: Path, pts: np.ndarray) -> None:
    """Write float32 XYZ binary PCD v0.7."""
    n = len(pts)
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        "VERSION 0.7\n"
        "FIELDS x y z\n"
        "SIZE 4 4 4\n"
        "TYPE F F F\n"
        "COUNT 1 1 1\n"
        f"WIDTH {n}\n"
        "HEIGHT 1\n"
        "VIEWPOINT 0 0 0 1 0 0 0\n"
        f"POINTS {n}\n"
        "DATA binary\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(header)
        f.write(pts.astype(np.float32, copy=False).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json", type=Path)
    ap.add_argument("output_pcd", type=Path)
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--include-visual", action="store_true")
    ap.add_argument("--include-floor", action="store_true")
    ap.add_argument("--floor-step", type=float, default=0.10)
    args = ap.parse_args()

    objs = json.loads(args.input_json.read_text(encoding="utf-8"))
    pts: list[tuple[float, float, float]] = []
    for obj in objs:
        t = obj.get("type", "BOX")
        if t == "VISUAL_BOX" and not args.include_visual:
            continue
        if t in ("BOX", "VISUAL_BOX"):
            pts.extend(
                sample_box_surface(
                    float(obj["x"]),
                    float(obj["y"]),
                    float(obj["half_x"]),
                    float(obj["half_y"]),
                    float(obj["z_min"]),
                    float(obj["z_max"]),
                    args.step,
                )
            )
        elif t == "CYLINDER":
            pts.extend(
                sample_cylinder_surface(
                    float(obj["x"]),
                    float(obj["y"]),
                    float(obj["radius"]),
                    float(obj["z_min"]),
                    float(obj["z_max"]),
                    args.step,
                )
            )

    if args.include_floor:
        xs = np.arange(-5.4, 5.41, args.floor_step)
        ys = np.arange(-5.8, 5.81, args.floor_step)
        gx, gy = np.meshgrid(xs, ys)
        floor_pts = np.column_stack([gx.ravel(), gy.ravel(), np.zeros(gx.size)])
        arr = np.vstack([np.array(pts, dtype=np.float32), floor_pts.astype(np.float32)])
    else:
        arr = np.array(pts, dtype=np.float32)

    # Deduplicate at step resolution using voxel hashing
    vox = np.round(arr / args.step).astype(np.int32)
    _, uniq_idx = np.unique(vox, axis=0, return_index=True)
    arr = arr[np.sort(uniq_idx)]
    write_binary_pcd(args.output_pcd, arr)
    print(f"Wrote {len(arr)} points -> {args.output_pcd}")


if __name__ == "__main__":
    main()
