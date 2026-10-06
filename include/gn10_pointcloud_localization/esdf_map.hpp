#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
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

class ESDFMap {
public:
    ESDFMap() = default;
    ~ESDFMap() = default;

    const ESDFHeader& header() const { return header_; }
    const std::vector<float>& data() const { return grid_; }
    bool empty() const { return grid_.empty(); }

    void setHeader(const ESDFHeader& header) { header_ = header; }
    void setData(std::vector<float>&& data) { grid_ = std::move(data); }

    // ファイル保存・読込
    bool saveBinary(const std::string& file_path) const;
    bool loadBinary(const std::string& file_path);

    // 点群 (x, y, z float 配列) からの高速 ESDF 構築
    bool buildFromPoints(
        const std::vector<float>& points_xyz,
        float resolution,
        float max_dist = 0.5f
    );

    // PCD ファイルからの構築（自動キャッシュ機能対応）
    bool buildFromPCD(
        const std::string& pcd_file_path,
        float resolution,
        float max_dist = 0.5f
    );

    // 従来のフィールド定義オブジェクトからの ESDF 構築（互換性用）
    bool buildFromFieldObjects(
        const std::vector<FieldObject>& objects,
        float resolution,
        float min_x, float max_x,
        float min_y, float max_y,
        float min_z, float max_z,
        float max_dist = 0.5f
    );

    // 座標 (wx, wy, wz) に対するホスト側での距離取得（デバッグ・検証用）
    float getDistance(float wx, float wy, float wz) const;

private:
    ESDFHeader header_{};
    std::vector<float> grid_;
};
