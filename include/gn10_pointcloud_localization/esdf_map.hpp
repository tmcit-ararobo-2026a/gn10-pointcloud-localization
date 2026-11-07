// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gn10_pointcloud_localization/cuda/field_objects.cuh"

#pragma pack(push, 1)
struct ESDFHeader {
    float resolution{0.05f};
    int32_t size_x{0};
    int32_t size_y{0};
    int32_t size_z{0};
    float min_x{0.0f};
    float min_y{0.0f};
    float min_z{0.0f};
    float max_dist_thresh{0.5f};
};
#pragma pack(pop)

struct ESDFCropBounds {
    bool enabled{false};
    float min_x{-5.5f};
    float max_x{5.5f};
    float min_y{-6.0f};
    float max_y{6.0f};
    float min_z{-0.2f};
    float max_z{4.0f};
};

class ESDFMap
{
public:
    ESDFMap()  = default;
    ~ESDFMap() = default;

    const ESDFHeader& header() const
    {
        return header_;
    }
    const std::vector<float>& data() const
    {
        return grid_;
    }
    bool empty() const
    {
        return grid_.empty();
    }

    void setHeader(const ESDFHeader& header)
    {
        header_ = header;
    }
    void setData(std::vector<float>&& data)
    {
        grid_ = std::move(data);
    }

    // Binary file serialization and deserialization (.esdf format)
    bool saveBinary(const std::string& file_path) const;
    bool loadBinary(const std::string& file_path);

    // Build 3D ESDF from a raw point cloud (interleaved x, y, z float array) with ground filtering
    bool buildFromPoints(
        const std::vector<float>& points_xyz,
        float resolution,
        float max_dist                    = 0.5f,
        const ESDFCropBounds& crop_bounds = ESDFCropBounds{},
        float ground_z_thresh             = 0.04f
    );

    // Build 3D ESDF from a PCD file (with automatic .esdf binary caching and ground filtering)
    bool buildFromPCD(
        const std::string& pcd_file_path,
        float resolution,
        float max_dist                    = 0.5f,
        const ESDFCropBounds& crop_bounds = ESDFCropBounds{},
        float ground_z_thresh             = 0.04f
    );

    // Build 3D ESDF from analytic field object primitives (backward compatibility)
    bool buildFromFieldObjects(
        const std::vector<FieldObject>& objects,
        float resolution,
        float min_x,
        float max_x,
        float min_y,
        float max_y,
        float min_z,
        float max_z,
        float max_dist = 0.5f
    );

    // Query signed/unsigned Euclidean distance at world coordinates (wx, wy, wz) on the host
    float getDistance(float wx, float wy, float wz) const;

    /**
     * @brief Projects the 3D ESDF along the Z axis (minimum distance within [z_min, z_max])
     *        to generate a 2D OccupancyGrid-compatible costmap.
     *
     * @param z_min Lower Z bound of robot collision height [m]
     * @param z_max Upper Z bound of robot collision height [m]
     * @param robot_radius Physical robot radius [m] (cells within this distance get lethal cost 100)
     * @param inflation_margin Inflation decay margin [m] (cost decays linearly from 99 to 1)
     * @param out_width Output 2D map width [cells]
     * @param out_height Output 2D map height [cells]
     * @param out_origin_x Output 2D map origin X [m]
     * @param out_origin_y Output 2D map origin Y [m]
     * @param out_costmap Output 2D costmap array (values in [0, 100])
     * @return true if projection succeeds
     */
    bool sliceTo2DCostmap(
        float z_min,
        float z_max,
        float robot_radius,
        float inflation_margin,
        int& out_width,
        int& out_height,
        float& out_origin_x,
        float& out_origin_y,
        std::vector<int8_t>& out_costmap
    ) const;

    /**
     * @brief ESDF voxel point structure for RViz2 visualization
     */
    struct ESDFVoxelPoint {
        float x;
        float y;
        float z;
        float distance_m;
    };

    /**
     * @brief Extracts voxel center points from the 3D ESDF for visualization.
     *
     * @param max_distance_m Maximum distance threshold [m] to extract (if negative, uses header_.max_dist_thresh)
     * @param stride Voxel subsampling stride (>= 1)
     * @return std::vector<ESDFVoxelPoint> Extracted voxel points
     */
    std::vector<ESDFVoxelPoint> extractVoxelPoints(
        float max_distance_m = -1.0f, int stride = 1
    ) const;

private:
    ESDFHeader header_{};
    std::vector<float> grid_;
};
