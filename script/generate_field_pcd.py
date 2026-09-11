#!/usr/bin/env python3
import numpy as np

def generate_box(x_min, x_max, y_min, y_max, z_min, z_max, step=0.01):
    """直方体の表面（6面）に点を生成"""
    points = []
    # X面 (前・後)
    for x in [x_min, x_max]:
        y = np.arange(y_min, y_max + step, step)
        z = np.arange(z_min, z_max + step, step)
        Y, Z = np.meshgrid(y, z)
        X = np.full_like(Y, x)
        points.append(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))
    # Y面 (左・右)
    for y in [y_min, y_max]:
        x = np.arange(x_min, x_max + step, step)
        z = np.arange(z_min, z_max + step, step)
        X, Z = np.meshgrid(x, z)
        Y = np.full_like(X, y)
        points.append(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))
    # Z面 (上・下)
    for z in [z_min, z_max]:
        x = np.arange(x_min, x_max + step, step)
        y = np.arange(y_min, y_max + step, step)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, z)
        points.append(np.column_stack((X.ravel(), Y.ravel(), Z.ravel())))
    return np.vstack(points)

def generate_cylinder(cx, cy, radius, z_min, z_max, step=0.01):
    """円柱の側面および上下底面に点を生成"""
    points = []
    height = z_max - z_min
    num_z = int(height / step) + 1
    num_theta = int(2 * np.pi * radius / step) + 1

    # 側面
    z_vals = np.linspace(z_min, z_max, num_z)
    theta_vals = np.linspace(0, 2 * np.pi, num_theta)
    for z in z_vals:
        for t in theta_vals:
            x = cx + radius * np.cos(t)
            y = cy + radius * np.sin(t)
            points.append([x, y, z])

    # 上下天板
    r_vals = np.arange(0, radius, step)
    for z in [z_min, z_max]:
        for r in r_vals:
            for t in theta_vals:
                points.append([cx + r * np.cos(t), cy + r * np.sin(t), z])

    return np.array(points)

def write_pcd(filename, points):
    """PCD (ASCII) ファイルとして書き出し"""
    points = np.unique(points, axis=0) # 重複点の除去
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {len(points)}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {len(points)}
DATA ascii
"""
    with open(filename, 'w') as f:
        f.write(header)
        np.savetxt(f, points, fmt='%.4f %.4f %.4f')
    print(f"Successfully generated {filename} with {len(points)} points.")

def main():
    all_points = []
    step = 0.02  # 点群の生成間隔 (2cm) ※解像度とファイルサイズのバランス調整

    # 1. 外壁 (10.5m x 11.4m, H=0.15m)
    wall_x, wall_y, wall_h = 10.5 / 2.0, 11.4 / 2.0, 0.15
    all_points.append(generate_box(-wall_x, wall_x, -wall_y, wall_y, 0, wall_h, step))

    # 2. 教壇 (X: -5.25~5.25m, Y: -0.3~0.3m, H: 0.2m)
    all_points.append(generate_box(-wall_x, wall_x, -0.3, 0.3, 0, 0.2, step))

    # オブジェクトの領域A/B対称展開用リスト
    objects_spec = []

    # 各オブジェクトの座標定義 (単位: m)
    # バケツ① (φ0.273 x H0.255)
    b1_r = 0.273 / 2.0
    objects_spec.append(('cyl', 0.55, 0.87, b1_r, 0, 0.255))

    # バケツ② (台座 0.3x0.3xH0.6 + バケツ①)
    objects_spec.append(('box', -1.27, 1.48, 0.3, 0.3, 0, 0.6))
    objects_spec.append(('cyl', -1.27, 1.48, b1_r, 0.6, 0.855))

    # バケツ③ (台座 0.3x0.3xH0.3 + バケツ①)
    objects_spec.append(('box', 2.37, 1.48, 0.3, 0.3, 0, 0.3))
    objects_spec.append(('cyl', 2.37, 1.48, b1_r, 0.3, 0.555))

    # 椅子 (座面H0.46, 全高H0.807, W0.36 x D0.4)
    objects_spec.append(('box', 0.55, 4.98, 0.36, 0.40, 0, 0.807))

    # 机 (W0.65 x D0.45 x H0.76) - 4個
    desk_coords = [(-2.295, 3.855), (3.395, 3.855), (-4.895, 5.445), (-4.750, 1.105)]
    for dx, dy in desk_coords:
        objects_spec.append(('box', dx, dy, 0.65, 0.45, 0, 0.76))

    # 旗 (土台 0.39x0.39xH0.18 + 支柱 H3.0 + 旗 W0.6xH1.8)
    objects_spec.append(('box', 0.55, 3.025, 0.39, 0.39, 0, 0.18))
    objects_spec.append(('cyl', 0.55, 3.025, 0.03, 0.18, 3.0)) # 支柱 φ0.06

    # 領域A (+Y) と 領域B (-Y) に配置展開
    for item in objects_spec:
        shape_type = item[0]
        x, y = item[1], item[2]
        
        for y_sign in [1.0, -1.0]:
            cy = y * y_sign
            if shape_type == 'box':
                w, d, z_min, z_max = item[3], item[4], item[5], item[6]
                all_points.append(generate_box(x - w/2, x + w/2, cy - d/2, cy + d/2, z_min, z_max, step))
            elif shape_type == 'cyl':
                r, z_min, z_max = item[3], item[4], item[5]
                all_points.append(generate_cylinder(x, cy, r, z_min, z_max, step))

    # 全点群の結合と保存
    full_cloud = np.vstack(all_points)
    write_pcd("robocon2026_field.pcd", full_cloud)

if __name__ == "__main__":
    main()