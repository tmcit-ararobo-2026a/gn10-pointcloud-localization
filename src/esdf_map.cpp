#include "gn10_pointcloud_localization/esdf_map.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>

bool ESDFMap::saveBinary(const std::string& file_path) const
{
    if (grid_.empty()) return false;
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "[ESDFMap] Failed to open file for writing: " << file_path << std::endl;
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(&header_), sizeof(ESDFHeader));
    ofs.write(reinterpret_cast<const char*>(grid_.data()), grid_.size() * sizeof(float));
    return true;
}

bool ESDFMap::loadBinary(const std::string& file_path)
{
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        return false;
    }
    ifs.read(reinterpret_cast<char*>(&header_), sizeof(ESDFHeader));
    if (!ifs) return false;

    size_t total_elements = static_cast<size_t>(header_.size_x) * header_.size_y * header_.size_z;
    grid_.resize(total_elements);
    ifs.read(reinterpret_cast<char*>(grid_.data()), total_elements * sizeof(float));
    return ifs.good() ||
           ifs.gcount() == static_cast<std::streamsize>(total_elements * sizeof(float));
}

bool ESDFMap::buildFromPoints(
    const std::vector<float>& points_xyz, float resolution, float max_dist
)
{
    size_t num_pts = points_xyz.size() / 3;
    if (num_pts == 0 || resolution <= 0.0f) return false;

    float min_x = std::numeric_limits<float>::max(), max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max(), max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max(), max_z = std::numeric_limits<float>::lowest();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->points.reserve(num_pts);

    for (size_t i = 0; i < num_pts; ++i) {
        float x = points_xyz[i * 3 + 0];
        float y = points_xyz[i * 3 + 1];
        float z = points_xyz[i * 3 + 2];
        if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;

        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        min_z = std::min(min_z, z);
        max_z = std::max(max_z, z);

        cloud->points.emplace_back(x, y, z);
    }

    if (cloud->empty()) return false;

    // マージン付与 (max_dist 分外側まで拡張)
    min_x -= max_dist;
    max_x += max_dist;
    min_y -= max_dist;
    max_y += max_dist;
    min_z -= max_dist;
    max_z += max_dist;

    int nx = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / resolution)));
    int ny = std::max(1, static_cast<int>(std::ceil((max_y - min_y) / resolution)));
    int nz = std::max(1, static_cast<int>(std::ceil((max_z - min_z) / resolution)));

    header_.resolution      = resolution;
    header_.size_x          = nx;
    header_.size_y          = ny;
    header_.size_z          = nz;
    header_.min_x           = min_x;
    header_.min_y           = min_y;
    header_.min_z           = min_z;
    header_.max_dist_thresh = max_dist;

    size_t total_voxels = static_cast<size_t>(nx) * ny * nz;
    grid_.assign(total_voxels, max_dist);

    // 高速化: PCL KdTreeFLANN を利用して点群から各ボクセルへの最近傍探索を行う
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

// OpenMP 並列化で各スライス (Z) を並行処理
#pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < nz; ++z) {
        float wz = min_z + (z + 0.5f) * resolution;
        std::vector<int> pointIdxNKNSearch(1);
        std::vector<float> pointNKNSquaredDistance(1);

        for (int y = 0; y < ny; ++y) {
            float wy = min_y + (y + 0.5f) * resolution;
            for (int x = 0; x < nx; ++x) {
                float wx = min_x + (x + 0.5f) * resolution;
                pcl::PointXYZ searchPoint(wx, wy, wz);

                // 半径 max_dist 以内の最近傍探索
                if (kdtree.radiusSearch(
                        searchPoint, max_dist, pointIdxNKNSearch, pointNKNSquaredDistance, 1
                    ) > 0) {
                    float dist = std::sqrt(pointNKNSquaredDistance[0]);
                    size_t idx = static_cast<size_t>(x) + static_cast<size_t>(y) * nx +
                                 static_cast<size_t>(z) * nx * ny;
                    grid_[idx] = std::min(max_dist, dist);
                }
            }
        }
    }

    return true;
}

bool ESDFMap::buildFromPCD(const std::string& pcd_file_path, float resolution, float max_dist)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    if (pcl::io::loadPCDFile<pcl::PointXYZ>(pcd_file_path, *cloud) == -1) {
        std::cerr << "[ESDFMap] Failed to load PCD file: " << pcd_file_path << std::endl;
        return false;
    }

    std::vector<float> pts;
    pts.reserve(cloud->points.size() * 3);
    for (const auto& pt : cloud->points) {
        pts.push_back(pt.x);
        pts.push_back(pt.y);
        pts.push_back(pt.z);
    }

    return buildFromPoints(pts, resolution, max_dist);
}

// 3D BOX (直方体) 表面までの最短距離計算
static inline float distToBoxSurface3D_Host(
    float px,
    float py,
    float pz,
    float cx,
    float cy,
    float cz_min,
    float cz_max,
    float half_w,
    float half_d
)
{
    float cz_center = (cz_min + cz_max) * 0.5f;
    float half_h    = (cz_max - cz_min) * 0.5f;

    float dx = std::fabs(px - cx) - half_w;
    float dy = std::fabs(py - cy) - half_d;
    float dz = std::fabs(pz - cz_center) - half_h;

    float ax = std::max(dx, 0.0f);
    float ay = std::max(dy, 0.0f);
    float az = std::max(dz, 0.0f);

    float outside_dist = std::sqrt(ax * ax + ay * ay + az * az);
    float inside_dist  = std::min(std::max(dx, std::max(dy, dz)), 0.0f);

    return std::fabs(outside_dist + inside_dist);
}

// 3D CYLINDER (高さ制限あり円柱) 表面までの最短距離計算
static inline float distToCylinder3D_Host(
    float px, float py, float pz, float cx, float cy, float cz_min, float cz_max, float radius
)
{
    float dx   = px - cx;
    float dy   = py - cy;
    float d_xy = std::max(std::sqrt(dx * dx + dy * dy) - radius, 0.0f);

    float cz_center = (cz_min + cz_max) * 0.5f;
    float half_h    = (cz_max - cz_min) * 0.5f;
    float d_z       = std::max(std::fabs(pz - cz_center) - half_h, 0.0f);

    return std::sqrt(d_xy * d_xy + d_z * d_z);
}

bool ESDFMap::buildFromFieldObjects(
    const std::vector<FieldObject>& objects,
    float resolution,
    float min_x,
    float max_x,
    float min_y,
    float max_y,
    float min_z,
    float max_z,
    float max_dist
)
{
    if (objects.empty() || resolution <= 0.0f) return false;

    min_x -= max_dist;
    max_x += max_dist;
    min_y -= max_dist;
    max_y += max_dist;
    min_z -= max_dist;
    max_z += max_dist;

    int nx = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / resolution)));
    int ny = std::max(1, static_cast<int>(std::ceil((max_y - min_y) / resolution)));
    int nz = std::max(1, static_cast<int>(std::ceil((max_z - min_z) / resolution)));

    header_.resolution      = resolution;
    header_.size_x          = nx;
    header_.size_y          = ny;
    header_.size_z          = nz;
    header_.min_x           = min_x;
    header_.min_y           = min_y;
    header_.min_z           = min_z;
    header_.max_dist_thresh = max_dist;

    size_t total_voxels = static_cast<size_t>(nx) * ny * nz;
    grid_.assign(total_voxels, max_dist);

#pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < nz; ++z) {
        float wz = min_z + (z + 0.5f) * resolution;
        for (int y = 0; y < ny; ++y) {
            float wy = min_y + (y + 0.5f) * resolution;
            for (int x = 0; x < nx; ++x) {
                float wx    = min_x + (x + 0.5f) * resolution;
                float min_d = max_dist;

                for (const auto& obj : objects) {
                    if (obj.type == VISUAL_BOX) continue;

                    // AABB（軸並行バウンディングボックス）拡張範囲による事前フィルタリング
                    float bound_x = (obj.type == BOX) ? obj.param1 : obj.param1;
                    float bound_y = (obj.type == BOX) ? obj.param2 : obj.param1;

                    if (wx < obj.center_x - bound_x - max_dist ||
                        wx > obj.center_x + bound_x + max_dist ||
                        wy < obj.center_y - bound_y - max_dist ||
                        wy > obj.center_y + bound_y + max_dist || wz < obj.z_min - max_dist ||
                        wz > obj.z_max + max_dist) {
                        continue;
                    }

                    float d = max_dist;
                    if (obj.type == CYLINDER) {
                        d = distToCylinder3D_Host(
                            wx, wy, wz, obj.center_x, obj.center_y, obj.z_min, obj.z_max, obj.param1
                        );
                    } else if (obj.type == BOX) {
                        d = distToBoxSurface3D_Host(
                            wx,
                            wy,
                            wz,
                            obj.center_x,
                            obj.center_y,
                            obj.z_min,
                            obj.z_max,
                            obj.param1,
                            obj.param2
                        );
                    }

                    if (d < min_d) min_d = d;
                }

                size_t idx = static_cast<size_t>(x) + static_cast<size_t>(y) * nx +
                             static_cast<size_t>(z) * nx * ny;
                grid_[idx] = min_d;
            }
        }
    }

    return true;
}

float ESDFMap::getDistance(float wx, float wy, float wz) const
{
    if (grid_.empty() || !std::isfinite(wx) || !std::isfinite(wy) || !std::isfinite(wz) ||
        wx < header_.min_x || wy < header_.min_y || wz < header_.min_z ||
        wx >= header_.min_x + header_.size_x * header_.resolution ||
        wy >= header_.min_y + header_.size_y * header_.resolution ||
        wz >= header_.min_z + header_.size_z * header_.resolution) {
        return header_.max_dist_thresh;
    }

    // Match CUDA's linear interpolation and clamp at the outer texel centers.
    const float fx = std::clamp(
        (wx - header_.min_x) / header_.resolution - 0.5f, 0.0f, float(header_.size_x - 1)
    );
    const float fy = std::clamp(
        (wy - header_.min_y) / header_.resolution - 0.5f, 0.0f, float(header_.size_y - 1)
    );
    const float fz = std::clamp(
        (wz - header_.min_z) / header_.resolution - 0.5f, 0.0f, float(header_.size_z - 1)
    );
    const int x0 = int(fx), y0 = int(fy), z0 = int(fz);
    const int x1   = std::min(x0 + 1, header_.size_x - 1);
    const int y1   = std::min(y0 + 1, header_.size_y - 1);
    const int z1   = std::min(z0 + 1, header_.size_z - 1);
    const float tx = fx - x0, ty = fy - y0, tz = fz - z0;
    const auto sample = [&](int x, int y, int z) {
        return grid_
            [size_t(x) + size_t(y) * header_.size_x + size_t(z) * header_.size_x * header_.size_y];
    };
    const auto lerp = [](float a, float b, float t) { return a + t * (b - a); };
    return lerp(
        lerp(
            lerp(sample(x0, y0, z0), sample(x1, y0, z0), tx),
            lerp(sample(x0, y1, z0), sample(x1, y1, z0), tx),
            ty
        ),
        lerp(
            lerp(sample(x0, y0, z1), sample(x1, y0, z1), tx),
            lerp(sample(x0, y1, z1), sample(x1, y1, z1), tx),
            ty
        ),
        tz
    );
}

/**
 * @brief 可視化用のボクセル点群を抽出する
 *
 * 指定された最大距離閾値未満のボクセルをワールド座標系における点群として抽出します。
 *
 * @param max_distance_m 抽出する最大距離[m] (負値の場合は header_.max_dist_thresh 未満)
 * @param stride ボクセルの間引きステップ (1以上の整数)
 * @return std::vector<ESDFMap::ESDFVoxelPoint> 抽出されたボクセル点群
 */
std::vector<ESDFMap::ESDFVoxelPoint> ESDFMap::extractVoxelPoints(
    float max_distance_m, int stride
) const
{
    std::vector<ESDFVoxelPoint> voxel_points;
    if (grid_.empty()) {
        return voxel_points;
    }

    // 間引きステップの正規化
    int effective_stride = stride;
    if (effective_stride < 1) {
        effective_stride = 1;
    }

    // 抽出距離しきい値の決定
    float distance_threshold_m = header_.max_dist_thresh;
    if (max_distance_m >= 0.0f && max_distance_m < header_.max_dist_thresh) {
        distance_threshold_m = max_distance_m;
    }

    const float resolution_m = header_.resolution;
    const int size_x         = header_.size_x;
    const int size_y         = header_.size_y;
    const int size_z         = header_.size_z;
    const float min_x_m      = header_.min_x;
    const float min_y_m      = header_.min_y;
    const float min_z_m      = header_.min_z;

    // ボクセル配列を走査してしきい値内の点群を抽出
    for (int z_index = 0; z_index < size_z; z_index += effective_stride) {
        const float world_z_m = min_z_m + (static_cast<float>(z_index) + 0.5f) * resolution_m;
        const size_t z_offset = static_cast<size_t>(z_index) * size_x * size_y;

        for (int y_index = 0; y_index < size_y; y_index += effective_stride) {
            const float world_y_m = min_y_m + (static_cast<float>(y_index) + 0.5f) * resolution_m;
            const size_t y_offset = static_cast<size_t>(y_index) * size_x;

            for (int x_index = 0; x_index < size_x; x_index += effective_stride) {
                const size_t voxel_index = static_cast<size_t>(x_index) + y_offset + z_offset;
                const float distance_m   = grid_[voxel_index];

                // 未観測・クリップ上限（何もない空間）を除外し、障害物表面近傍のみ抽出
                if (distance_m < distance_threshold_m && distance_m >= 0.0f) {
                    const float world_x_m =
                        min_x_m + (static_cast<float>(x_index) + 0.5f) * resolution_m;
                    voxel_points.push_back({world_x_m, world_y_m, world_z_m, distance_m});
                }
            }
        }
    }

    return voxel_points;
}