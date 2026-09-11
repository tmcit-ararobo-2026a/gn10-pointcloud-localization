cmake_minimum_required(VERSION 3.18)
project(gn10_pointcloud_localization LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87 89)
endif()

project(gn10_pointcloud_localization LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
)

# CUDA ライブラリのビルド
add_library(gn10_cuda_lib
  src/cuda/ground_filter.cu
)
set_target_properties(gn10_cuda_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# C++ ノードのビルド
add_executable(gn10_pointcloud_localization_node
  src/map_loader.cpp
  src/pose_solver.cpp
  src/localization_node.cpp
  src/main.cpp
)

target_link_libraries(gn10_pointcloud_localization_node
  gn10_cuda_lib
  ${CERES_LIBRARIES}
)

ament_target_dependencies(gn10_pointcloud_localization_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

# インストール設定 (Launch, Config, Node)
install(TARGETS 
  gn10_pointcloud_localization_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
