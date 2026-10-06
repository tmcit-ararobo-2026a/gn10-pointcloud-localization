#include <iostream>
#include <string>
#include "gn10_pointcloud_localization/esdf_map.hpp"

int main(int argc, char** argv)
{
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.pcd> <output.esdf> <resolution_m> [max_dist_m]" << std::endl;
        std::cerr << "Example: " << argv[0] << " map.pcd map.esdf 0.05 0.5" << std::endl;
        return 1;
    }

    std::string pcd_path = argv[1];
    std::string esdf_path = argv[2];
    float res = std::stof(argv[3]);
    float max_dist = (argc >= 5) ? std::stof(argv[4]) : 0.5f;

    std::cout << "[Converter] Building ESDF from " << pcd_path << " (resolution=" << res << "m, max_dist=" << max_dist << "m)..." << std::endl;

    ESDFMap map;
    if (!map.buildFromPCD(pcd_path, res, max_dist)) {
        std::cerr << "[Converter] Error building ESDF from PCD file." << std::endl;
        return 1;
    }

    const auto& h = map.header();
    std::cout << "[Converter] Generated Grid Dimensions: " << h.size_x << " x " << h.size_y << " x " << h.size_z 
              << " (" << (map.data().size() * sizeof(float)) / (1024 * 1024) << " MB)" << std::endl;

    if (!map.saveBinary(esdf_path)) {
        std::cerr << "[Converter] Failed to save ESDF to " << esdf_path << std::endl;
        return 1;
    }

    std::cout << "[Converter] Successfully saved ESDF to " << esdf_path << std::endl;
    return 0;
}