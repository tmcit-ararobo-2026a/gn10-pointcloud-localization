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

    // ファイル保存・読込
    bool saveBinary(const std::string& file_path) const;
    bool loadBinary(const std::string& file_path);

    // 点群 (x, y, z float 配列) からの高速 ESDF 構築 (床面フィルタ対応)
    bool buildFromPoints(
        const std::vector<float>& points_xyz,
        float resolution,
        float max_dist                    = 0.5f,
        const ESDFCropBounds& crop_bounds = ESDFCropBounds{},
        float ground_z_thresh             = 0.04f
    );

    // PCD ファイルからの構築（自動キャッシュ機能・床面フィルタ対応）
    bool buildFromPCD(
        const std::string& pcd_file_path,
        float resolution,
        float max_dist                    = 0.5f,
        const ESDFCropBounds& crop_bounds = ESDFCropBounds{},
        float ground_z_thresh             = 0.04f
    );

    // 従来のフィールド定義オブジェクトからの ESDF 構築（互換性用）
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

    // 座標 (wx, wy, wz) に対するホスト側での距離取得（デバッグ・検証用）
    float getDistance(float wx, float wy, float wz) const;

    /**
     * @brief 3D ESDF から Z 軸最小値射影を行い、2D コストマップ（OccupancyGrid 準拠）を生成する
     *
     * @param z_min ロボット干渉範囲の下限 [m]
     * @param z_max ロボット干渉範囲の上限 [m]
     * @param robot_radius ロボット物理半径 [m] (この距離以内は不可侵 100)
     * @param inflation_margin インフレーション波及マージン [m] (この範囲でコスト 99 から 1 へ減衰)
     * @param out_width 出力 2D マップの幅 (セル数)
     * @param out_height 出力 2D マップの高さ (セル数)
     * @param out_origin_x 出力 2D マップの原点 X [m]
     * @param out_origin_y 出力 2D マップの原点 Y [m]
     * @param out_costmap 出力 2D コストマップ配列 (0..100)
     * @return true 射影成功時
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
     * @brief 可視化用 ESDF ボクセル点データ構造体
     */
    struct ESDFVoxelPoint {
        float x;
        float y;
        float z;
        float distance_m;
    };

    /**
     * @brief 可視化用のボクセル点群を抽出する
     *
     * @param max_distance_m 抽出する最大距離[m] (負値の場合は header_.max_dist_thresh 未満)
     * @param stride ボクセルの間引きステップ (1以上の整数)
     * @return std::vector<ESDFVoxelPoint> 抽出されたボクセル点群
     */
    std::vector<ESDFVoxelPoint> extractVoxelPoints(
        float max_distance_m = -1.0f, int stride = 1
    ) const;

private:
    ESDFHeader header_{};
    std::vector<float> grid_;
};
