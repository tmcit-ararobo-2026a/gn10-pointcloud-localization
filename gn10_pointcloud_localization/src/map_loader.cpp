#include "gn10_pointcloud_localization/map_loader.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

using json = nlohmann::json;

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

#include "gn10_pointcloud_localization/map_loader.hpp"

using json = nlohmann::json;

std::vector<FieldObject> MapLoader::loadFromJSON(const std::string& file_path)
{
    std::vector<FieldObject> map;
    std::ifstream f(file_path);
    if (!f.is_open()) {
        std::cerr << "[MapLoader] Failed to open map file: " << file_path << std::endl;
        return map;
    }

    try {
        json j = json::parse(f);
        for (const auto& item : j) {
            FieldObject obj;
            std::string type_str = item.at("type").get<std::string>();
            obj.type             = (type_str == "CYLINDER") ? CYLINDER : BOX;
            obj.center_x         = item.at("x").get<float>();
            obj.center_y         = item.at("y").get<float>();
            obj.z_min            = item.at("z_min").get<float>();
            obj.z_max            = item.at("z_max").get<float>();
            obj.param1           = item.at("param1").get<float>();
            obj.param2           = item.at("param2").get<float>();

            // JSON内に "weight" キーが存在すれば取得、なければ 1.0f をデフォルト値とする
            obj.weight = item.value("weight", 1.0f);

            map.push_back(obj);
        }
    } catch (const std::exception& e) {
        std::cerr << "[MapLoader] JSON parse error: " << e.what() << std::endl;
    }

    return map;
}

std::vector<FieldObject> MapLoader::loadFromParams(const std::vector<std::string>& param_strings)
{
    // 形式: "TYPE,x,y,z_min,z_max,param1,param2[,weight]"
    std::vector<FieldObject> map;
    for (const auto& s : param_strings) {
        std::stringstream ss(s);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() >= 7) {
            FieldObject obj;
            obj.type     = (tokens[0] == "CYLINDER") ? CYLINDER : BOX;
            obj.center_x = std::stof(tokens[1]);
            obj.center_y = std::stof(tokens[2]);
            obj.z_min    = std::stof(tokens[3]);
            obj.z_max    = std::stof(tokens[4]);
            obj.param1   = std::stof(tokens[5]);
            obj.param2   = std::stof(tokens[6]);

            // 8つ目の要素が渡されていればパースし、無ければ 1.0f
            obj.weight = (tokens.size() >= 8) ? std::stof(tokens[7]) : 1.0f;

            map.push_back(obj);
        }
    }
    return map;
}

std::vector<FieldObject> MapLoader::createNHK2026FieldMap()
{
    std::vector<FieldObject> map;

    // 重みの目安:
    // 0.3 ~ 0.5 : 外壁・教壇（非常に強い拘束力を持たせる）
    // 1.0       : バケツ・旗土台など
    // 1.5 ~ 2.0 : 机・椅子など（位置ずれ・歪み・透過の影響を受けやすい）
    constexpr float W_WALL  = 0.4f;
    constexpr float W_STAGE = 0.4f;
    constexpr float W_PROP  = 1.0f;

    // 1. 外壁 (10.5m x 11.4m, H=0.15m)
    map.push_back({BOX, 0.000f, 5.700f, 0.000f, 0.150f, 5.250f, 0.050f, W_WALL});   // 上壁
    map.push_back({BOX, 0.000f, -5.700f, 0.000f, 0.150f, 5.250f, 0.050f, W_WALL});  // 下壁
    map.push_back({BOX, -5.250f, 0.000f, 0.000f, 0.150f, 0.050f, 5.700f, W_WALL});  // 左壁
    map.push_back({BOX, 5.250f, 0.000f, 0.000f, 0.150f, 0.050f, 5.700f, W_WALL});   // 右壁

    // 2. 教壇
    map.push_back({BOX, 0.000f, 0.000f, 0.000f, 0.200f, 5.250f, 0.300f, W_STAGE});

    struct ObjectSpec {
        ObjectType type;
        float x, y;
        float param1, param2;
        float z_min, z_max;
        float weight;
    };

    constexpr float bucket_radius = 0.273f / 2.0f;
    std::vector<ObjectSpec> base_specs;

    // バケツ①
    base_specs.push_back({CYLINDER, 0.550f, 0.870f, bucket_radius, 0.000f, 0.000f, 0.255f, W_PROP});

    // バケツ②
    base_specs.push_back({BOX, -1.270f, 1.480f, 0.150f, 0.150f, 0.000f, 0.600f, W_PROP});
    base_specs.push_back(
        {CYLINDER, -1.270f, 1.480f, bucket_radius, 0.000f, 0.600f, 0.855f, W_PROP}
    );

    // バケツ③
    base_specs.push_back({BOX, 2.370f, 1.480f, 0.150f, 0.150f, 0.000f, 0.300f, W_PROP});
    base_specs.push_back({CYLINDER, 2.370f, 1.480f, bucket_radius, 0.000f, 0.300f, 0.555f, W_PROP});

    // 椅子・机
    base_specs.push_back({BOX, 0.550f, 4.980f, 0.180f, 0.200f, 0.000f, 0.807f, W_PROP});

    constexpr float desk_coords[4][2] = {
        {-2.295f, 3.855f},
        { 3.395f, 3.855f},
        {-4.895f, 5.445f},
        {-4.750f, 1.105f}
    };
    for (const auto& coord : desk_coords) {
        base_specs.push_back({BOX, coord[0], coord[1], 0.325f, 0.225f, 0.000f, 0.760f, W_PROP});
    }

    // 旗
    base_specs.push_back({BOX, 0.550f, 3.025f, 0.195f, 0.195f, 0.000f, 0.180f, W_PROP});
    base_specs.push_back({CYLINDER, 0.550f, 3.025f, 0.030f, 0.000f, 0.180f, 3.000f, W_PROP});

    for (const auto& spec : base_specs) {
        for (const float y_sign : {1.0f, -1.0f}) {
            map.push_back(
                {spec.type,
                 spec.x,
                 spec.y * y_sign,
                 spec.z_min,
                 spec.z_max,
                 spec.param1,
                 spec.param2,
                 spec.weight}
            );
        }
    }

    return map;
}