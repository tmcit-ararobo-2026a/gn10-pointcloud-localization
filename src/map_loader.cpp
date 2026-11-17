// Copyright 2026 Gento Aiba and contributors
// SPDX-License-Identifier: Apache-2.0

#include "gn10_pointcloud_localization/map_loader.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <sstream>

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
            std::string type_str = item.at("type").get<std::string>();

            // Skip ZONE / PENALTY_ZONE entries because they are passable regions, not physical obstacles
            if (type_str == "ZONE" || type_str == "PENALTY_ZONE") {
                continue;
            }
            if (item.contains("passable") && item.at("passable").get<bool>()) {
                // Exclude passable zones that are not explicitly marked as VISUAL_BOX
                if (type_str != "VISUAL_BOX") {
                    continue;
                }
            }

            FieldObject obj;
            obj.type     = (type_str == "CYLINDER") ? CYLINDER :
                           (type_str == "VISUAL_BOX") ? VISUAL_BOX : BOX;
            obj.center_x = item.at("x").get<float>();
            obj.center_y = item.at("y").get<float>();
            obj.z_min    = item.at("z_min").get<float>();
            obj.z_max    = item.at("z_max").get<float>();

            // Shape parameters (modern schema: radius, half_x, half_y / legacy schema: param1, param2)
            if (obj.type == CYLINDER) {
                if (item.contains("radius")) {
                    obj.set_radius(item.at("radius").get<float>());
                } else if (item.contains("param1")) {
                    obj.set_radius(item.at("param1").get<float>());
                } else {
                    obj.set_radius(0.0f);
                }
            } else {
                // BOX / VISUAL_BOX
                float hx = 0.0f;
                float hy = 0.0f;
                if (item.contains("half_x")) {
                    hx = item.at("half_x").get<float>();
                } else if (item.contains("half_width")) {
                    hx = item.at("half_width").get<float>();
                } else if (item.contains("size_x")) {
                    hx = item.at("size_x").get<float>() * 0.5f;
                } else if (item.contains("param1")) {
                    hx = item.at("param1").get<float>();
                }

                if (item.contains("half_y")) {
                    hy = item.at("half_y").get<float>();
                } else if (item.contains("half_depth")) {
                    hy = item.at("half_depth").get<float>();
                } else if (item.contains("size_y")) {
                    hy = item.at("size_y").get<float>() * 0.5f;
                } else if (item.contains("param2")) {
                    hy = item.at("param2").get<float>();
                }
                obj.set_half_sizes(hx, hy);
            }

            map.push_back(obj);
        }
    } catch (const std::exception& e) {
        std::cerr << "[MapLoader] JSON parse error: " << e.what() << std::endl;
    }

    return map;
}

std::vector<FieldObject> MapLoader::loadFromParams(const std::vector<std::string>& param_strings)
{
    // Format: "TYPE,x,y,z_min,z_max,param1,param2"
    std::vector<FieldObject> map;
    for (const auto& s : param_strings) {
        std::stringstream ss(s);
        std::string token;
        std::vector<std::string> tokens;
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        if (tokens.size() == 7) {
            FieldObject obj;
            obj.type     = (tokens[0] == "CYLINDER") ? CYLINDER :
                           (tokens[0] == "VISUAL_BOX") ? VISUAL_BOX : BOX;
            obj.center_x = std::stof(tokens[1]);
            obj.center_y = std::stof(tokens[2]);
            obj.z_min    = std::stof(tokens[3]);
            obj.z_max    = std::stof(tokens[4]);
            obj.param1   = std::stof(tokens[5]);
            obj.param2   = std::stof(tokens[6]);
            map.push_back(obj);
        }
    }
    return map;
}
