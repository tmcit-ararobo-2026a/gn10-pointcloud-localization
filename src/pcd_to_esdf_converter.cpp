#include <iostream>
#include <string>

#include "gn10_pointcloud_localization/esdf_map.hpp"

int main(int argc, char** argv)
{
    if (argc < 4 || (argc != 4 && argc != 5 && argc != 10 && argc != 11)) {
        std::cerr << "Usage: " << argv[0]
                  << " <input.pcd> <output.esdf> <resolution_m> [max_dist_m]"
                  << " [min_x max_x min_y max_y min_z max_z]" << std::endl;
        std::cerr << "Example: " << argv[0] << " map.pcd map.esdf 0.05 0.5"
                  << " -5.5 5.5 -6.0 6.0 -0.2 4.0" << std::endl;
        return 1;
    }

    std::string pcd_path  = argv[1];
    std::string esdf_path = argv[2];
    float res             = std::stof(argv[3]);
    float max_dist        = (argc >= 5) ? std::stof(argv[4]) : 0.5f;
    ESDFCropBounds crop_bounds;
    const bool has_crop = argc == 10 || argc == 11;
    if (has_crop) {
        const int crop_start = argc == 10 ? 4 : 5;
        crop_bounds.enabled  = true;
        crop_bounds.min_x    = std::stof(argv[crop_start + 0]);
        crop_bounds.max_x    = std::stof(argv[crop_start + 1]);
        crop_bounds.min_y    = std::stof(argv[crop_start + 2]);
        crop_bounds.max_y    = std::stof(argv[crop_start + 3]);
        crop_bounds.min_z    = std::stof(argv[crop_start + 4]);
        crop_bounds.max_z    = std::stof(argv[crop_start + 5]);
    }

    std::cout << "[Converter] Building ESDF from " << pcd_path << " (resolution=" << res
              << "m, max_dist=" << max_dist << "m)..." << std::endl;
    if (has_crop) {
        std::cout << "[Converter] Crop bounds: x=[" << crop_bounds.min_x << ", "
                  << crop_bounds.max_x << "], y=[" << crop_bounds.min_y << ", " << crop_bounds.max_y
                  << "], z=[" << crop_bounds.min_z << ", " << crop_bounds.max_z << "]" << std::endl;
    }

    ESDFMap map;
    if (!map.buildFromPCD(pcd_path, res, max_dist, crop_bounds)) {
        std::cerr << "[Converter] Error building ESDF from PCD file." << std::endl;
        return 1;
    }

    const auto& h = map.header();
    std::cout << "[Converter] Generated Grid Dimensions: " << h.size_x << " x " << h.size_y << " x "
              << h.size_z << " (" << (map.data().size() * sizeof(float)) / (1024 * 1024) << " MB)"
              << std::endl;

    if (!map.saveBinary(esdf_path)) {
        std::cerr << "[Converter] Failed to save ESDF to " << esdf_path << std::endl;
        return 1;
    }

    std::cout << "[Converter] Successfully saved ESDF to " << esdf_path << std::endl;
    return 0;
}