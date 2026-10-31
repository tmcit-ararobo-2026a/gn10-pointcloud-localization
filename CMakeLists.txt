# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.18)
project(gn10_pointcloud_localization LANGUAGES C CXX)
option(BUILD_RUNTIME_NODES "Build localization runtime nodes" ON)
option(BUILD_GPU_NODES "Build CUDA matcher (CPU monitor can be built separately)" ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 72 86 87 89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

if(BUILD_RUNTIME_NODES)
if(BUILD_GPU_NODES)
enable_language(CUDA)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(PCL REQUIRED QUIET COMPONENTS common io kdtree search)
find_package(OpenMP REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# CUDA ライブラリ
add_library(gn10_cuda_lib
  src/cuda/ground_filter.cu
  src/cuda/field_objects.cu
  src/cuda/esdf_matcher.cu
)
set_target_properties(gn10_cuda_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
)

# nvcc 内部で生成されるコードの -Wpedantic 警告を抑制
target_compile_options(gn10_cuda_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-pedantic>
)

# 共通 ESDF ライブラリ
add_library(gn10_esdf_lib
  src/esdf_map.cpp
)
target_link_libraries(gn10_esdf_lib
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
)

# C++ ノードのビルド
add_executable(gn10_pointcloud_localization_node
  src/map_loader.cpp
  src/pose_solver.cpp
  src/global_searcher.cpp
  src/localization_node.cpp
  src/main.cpp
)

# C++ (g++) 側のみにコンパイルオプションを適用
target_compile_options(gn10_pointcloud_localization_node PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

target_link_libraries(gn10_pointcloud_localization_node
  gn10_cuda_lib
  gn10_esdf_lib
  nlohmann_json::nlohmann_json
  ${PCL_LIBRARIES}
  OpenMP::OpenMP_CXX
  CUDA::cudart
)

# PCD -> ESDF 変換 CLI ツール
add_executable(pcd_to_esdf_converter
  src/pcd_to_esdf_converter.cpp
)
target_link_libraries(pcd_to_esdf_converter
  gn10_esdf_lib
  ${PCL_LIBRARIES}
)

# ESDF -> 2D Costmap 変換 CLI ツール
add_executable(esdf_to_costmap
  src/esdf_to_costmap.cpp
)
target_link_libraries(esdf_to_costmap
  gn10_esdf_lib
  OpenMP::OpenMP_CXX
)

ament_target_dependencies(gn10_pointcloud_localization_node
  ament_index_cpp
  message_filters
  rclcpp
  std_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  visualization_msgs
  nlohmann_json
)

# インストール設定 (Launch, Config, Node, Libraries)
install(TARGETS 
  gn10_cuda_lib
  gn10_esdf_lib
  gn10_pointcloud_localization_node
  pcd_to_esdf_converter
  esdf_to_costmap
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  enable_testing()
  add_executable(dynamic_filter_test tests/test_dynamic_filter.cu)
  target_include_directories(dynamic_filter_test PRIVATE include)
  target_link_libraries(dynamic_filter_test gn10_cuda_lib CUDA::cudart)
  target_compile_options(dynamic_filter_test PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-UNDEBUG>)
  add_test(NAME dynamic_filter COMMAND dynamic_filter_test)
  set_tests_properties(dynamic_filter PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(angle_and_imu_bounds_test tests/test_angle_and_imu_bounds.cpp)
  target_include_directories(angle_and_imu_bounds_test PRIVATE include)
  target_link_libraries(angle_and_imu_bounds_test gn10_cuda_lib gn10_esdf_lib CUDA::cudart)
  target_compile_options(angle_and_imu_bounds_test PRIVATE -UNDEBUG)
  add_test(NAME angle_and_imu_bounds COMMAND angle_and_imu_bounds_test)

  add_executable(scan_2d_deskew_test tests/test_scan_2d_deskew.cpp)
  target_include_directories(scan_2d_deskew_test PRIVATE include)
  target_link_libraries(scan_2d_deskew_test gn10_cuda_lib gn10_esdf_lib CUDA::cudart)
  ament_target_dependencies(scan_2d_deskew_test sensor_msgs rclcpp)
  target_compile_options(scan_2d_deskew_test PRIVATE -UNDEBUG)
  add_test(NAME scan_2d_deskew COMMAND scan_2d_deskew_test)

  add_executable(independent_2d_cost_test tests/test_independent_2d_cost.cpp)
  target_include_directories(independent_2d_cost_test PRIVATE include)
  target_link_libraries(independent_2d_cost_test gn10_cuda_lib gn10_esdf_lib CUDA::cudart)
  target_compile_options(independent_2d_cost_test PRIVATE -UNDEBUG)
  add_test(NAME independent_2d_cost COMMAND independent_2d_cost_test)
  set_tests_properties(independent_2d_cost PROPERTIES SKIP_RETURN_CODE 77)
endif()

# 他のノードや環境変数のためにライブラリをエクスポート
ament_export_libraries(gn10_cuda_lib gn10_esdf_lib)

endif()
endif()

install(DIRECTORY launch config rviz map
  DESTINATION share/${PROJECT_NAME}
)

ament_package()