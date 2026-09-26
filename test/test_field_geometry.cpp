#include "gn10_pointcloud_localization/map_loader.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifndef TEST_FIELD_MAP_PATH
#error TEST_FIELD_MAP_PATH must name the checked-in field map
#endif

namespace {
void require(bool condition)
{
    if (!condition) throw std::runtime_error("field geometry check failed");
}

bool same(const FieldObject& a, const FieldObject& b)
{
    const auto near = [](float x, float y) { return std::abs(x - y) < 1e-5f; };
    return a.type == b.type && near(a.center_x, b.center_x) &&
           near(a.center_y, b.center_y) && near(a.z_min, b.z_min) &&
           near(a.z_max, b.z_max) && near(a.param1, b.param1) &&
           near(a.param2, b.param2);
}
}  // namespace

int main()
{
    const auto json = MapLoader::loadFromJSON(TEST_FIELD_MAP_PATH);
    const auto fallback = MapLoader::createNHK2026FieldMap();
    require(json.size() == 35);
    require(fallback.size() == json.size());
    for (const auto& object : json) {
        require(std::count_if(fallback.begin(), fallback.end(),
                              [&](const FieldObject& other) { return same(object, other); }) == 1);
    }
    const auto has = [&](const FieldObject& expected) {
        return std::any_of(json.begin(), json.end(),
                           [&](const FieldObject& object) { return same(object, expected); });
    };
    require(has({BOX, 0.0f, 0.0f, 0.0f, 0.2f, 5.25f, 0.3f}));
    for (const float sign : {-1.0f, 1.0f}) {
        require(has({CYLINDER, 0.55f, sign * 0.87f, 0.0f, 0.255f, 0.1365f, 0.0f}));
        require(has({BOX, -1.27f, sign * 1.48f, 0.0f, 0.6f, 0.15f, 0.15f}));
        require(has({BOX, 2.37f, sign * 1.48f, 0.0f, 0.3f, 0.15f, 0.15f}));
        require(has({BOX, 0.55f, sign * 4.98f, 0.0f, 0.807f, 0.18f, 0.2f}));
        require(has({BOX, 0.55f, sign * 3.025f, 0.0f, 0.18f, 0.195f, 0.195f}));
        require(has({CYLINDER, 0.55f, sign * 3.025f, 0.18f, 3.0f, 0.03f, 0.0f}));
        require(has({BOX, 0.85f, sign * 3.025f, 1.2f, 3.0f, 0.3f, 0.02f}));
    }

    int base_count = 0;
    int desk_count = 0;
    float min_x = 1e9f, max_x = -1e9f, min_y = 1e9f, max_y = -1e9f;
    for (const auto& object : json) {
        if (object.type == VISUAL_BOX) {
            ++base_count;
            require(std::abs(object.z_max - 0.024f) < 1e-5f);
        }
        if (object.type == BOX && std::abs(object.param1 - 0.325f) < 1e-5f &&
            std::abs(object.param2 - 0.225f) < 1e-5f &&
            std::abs(object.z_max - 0.760f) < 1e-5f) ++desk_count;
        if (object.type == BOX || object.type == VISUAL_BOX) {
            min_x = std::min(min_x, object.center_x - object.param1);
            max_x = std::max(max_x, object.center_x + object.param1);
            min_y = std::min(min_y, object.center_y - object.param2);
            max_y = std::max(max_y, object.center_y + object.param2);
        }
    }
    require(base_count == 4);
    require(desk_count == 8);
    require(std::abs(min_x + 5.4f) < 1e-5f && std::abs(max_x - 5.4f) < 1e-5f);
    require(std::abs(min_y + 5.85f) < 1e-5f && std::abs(max_y - 5.85f) < 1e-5f);
    return 0;
}
