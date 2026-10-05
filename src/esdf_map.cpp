#include "gn10_pointcloud_localization/esdf_map.hpp"

#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <limits>
#include <queue>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>

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
    return ifs.good() || ifs.gcount() == static_cast<std::streamsize>(total_elements * sizeof(float));
}

bool ESDFMap::buildFromPoints(
    const std::vector<float>& points_xyz,
    float resolution,
    float max_dist
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

        min_x = std::min(min_x, x); max_x = std::max(max_x, x);
        min_y = std::min(min_y, y); max_y = std::max(max_y, y);
        min_z = std::min(min_z, z); max_z = std::max(max_z, z);

        cloud->points.emplace_back(x, y, z);
    }

    if (cloud->empty()) return false;

    // マージン付与 (max_dist 分外側まで拡張)
    min_x -= max_dist; max_x += max_dist;
    min_y -= max_dist; max_y += max_dist;
    min_z -= max_dist; max_z += max_dist;

    int nx = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / resolution)));
    int ny = std::max(1, static_cast<int>(std::ceil((max_y - min_y) / resolution)));
    int nz = std::max(1, static_cast<int>(std::ceil((max_z - min_z) / resolution)));

    header_.resolution = resolution;
    header_.size_x = nx;
    header_.size_y = ny;
    header_.size_z = nz;
    header_.min_x = min_x;
    header_.min_y = min_y;
    header_.min_z = min_z;
    header_.max_dist_thresh = max_dist;

    size_t total_voxels = static_cast<size_t>(nx) * ny * nz;
    grid_.assign(total_voxels, max_dist);

    // 高速化: PCL KdTreeFLANN を利用して点群から各ボクセルへの最近傍探索を行う
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    float max_dist_sq = max_dist * max_dist;

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
                if (kdtree.radiusSearch(searchPoint, max_dist, pointIdxNKNSearch, pointNKNSquaredDistance, 1) > 0) {
                    float dist = std::sqrt(pointNKNSquaredDistance[0]);
                    size_t idx = static_cast<size_t>(x) + static_cast<size_t>(y) * nx + static_cast<size_t>(z) * nx * ny;
                    grid_[idx] = std::min(max_dist, dist);
                }
            }
        }
    }

    return true;
}

bool ESDFMap::buildFromPCD(
    const std::string& pcd_file_path,
    float resolution,
    float max_dist
)
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

static inline float distToBoxSurface2D_Host(
    float px, float py, float cx, float cy, float half_w, float half_d
)
{
    float dx = std::fabs(px - cx) - half_w;
    float dy = std::fabs(py - cy) - half_d;
    float ax = std::max(dx, 0.0f);
    float ay = std::max(dy, 0.0f);
    return std::fabs(std::sqrt(ax * ax + ay * ay) + std::min(std::max(dx, dy), 0.0f));
}

static inline float distToCylinder2D_Host(float px, float py, float cx, float cy, float radius)
{
    float dx = px - cx;
    float dy = py - cy;
    float dist_center = std::sqrt(dx * dx + dy * dy);
    return std::fabs(dist_center - radius);
}

bool ESDFMap::buildFromFieldObjects(
    const std::vector<FieldObject>& objects,
    float resolution,
    float min_x, float max_x,
    float min_y, float max_y,
    float min_z, float max_z,
    float max_dist
)
{
    if (objects.empty() || resolution <= 0.0f) return false;

    min_x -= max_dist; max_x += max_dist;
    min_y -= max_dist; max_y += max_dist;
    min_z -= max_dist; max_z += max_dist;

    int nx = std::max(1, static_cast<int>(std::ceil((max_x - min_x) / resolution)));
    int ny = std::max(1, static_cast<int>(std::ceil((max_y - min_y) / resolution)));
    int nz = std::max(1, static_cast<int>(std::ceil((max_z - min_z) / resolution)));

    header_.resolution = resolution;
    header_.size_x = nx;
    header_.size_y = ny;
    header_.size_z = nz;
    header_.min_x = min_x;
    header_.min_y = min_y;
    header_.min_z = min_z;
    header_.max_dist_thresh = max_dist;

    size_t total_voxels = static_cast<size_t>(nx) * ny * nz;
    grid_.assign(total_voxels, max_dist);

    #pragma omp parallel for schedule(dynamic)
    for (int z = 0; z < nz; ++z) {
        float wz = min_z + (z + 0.5f) * resolution;
        for (int y = 0; y < ny; ++y) {
            float wy = min_y + (y + 0.5f) * resolution;
            for (int x = 0; x < nx; ++x) {
                float wx = min_x + (x + 0.5f) * resolution;
                float min_d = max_dist;

                for (const auto& obj : objects) {
                    if (obj.type == VISUAL_BOX) continue;
                    if (wz >= (obj.z_min - 0.05f) && wz <= (obj.z_max + 0.05f)) {
                        float d = max_dist;
                        if (obj.type == CYLINDER) {
                            d = distToCylinder2D_Host(wx, wy, obj.center_x, obj.center_y, obj.param1);
                        } else if (obj.type == BOX) {
                            d = distToBoxSurface2D_Host(wx, wy, obj.center_x, obj.center_y, obj.param1, obj.param2);
                        }
                        if (d < min_d) min_d = d;
                    }
                }

                size_t idx = static_cast<size_t>(x) + static_cast<size_t>(y) * nx + static_cast<size_t>(z) * nx * ny;
                grid_[idx] = min_d;
            }
        }
    }

    return true;
}

float ESDFMap::getDistance(float wx, float wy, float wz) const
{
    if (grid_.empty()) return header_.max_dist_thresh;

    float fx = (wx - header_.min_x) / header_.resolution;
    float fy = (wy - header_.min_y) / header_.resolution;
    float fz = (wz - header_.min_z) / header_.resolution;

    int gx = std::clamp(static_cast<int>(fx), 0, header_.size_x - 1);
    int gy = std::clamp(static_cast<int>(fy), 0, header_.size_y - 1);
    int gz = std::clamp(static_cast<int>(fz), 0, header_.size_z - 1);

    size_t idx = static_cast<size_t>(gx) + static_cast<size_t>(gy) * header_.size_x + static_cast<size_t>(gz) * header_.size_x * header_.size_y;
    return grid_[idx];
}
