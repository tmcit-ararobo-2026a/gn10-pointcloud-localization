cmake_minimum_required(VERSION 3.18)
project(gn10_pointcloud_localization LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 87 89)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(CUDAToolkit REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${CUDAToolkit_INCLUDE_DIRS}
)

# CUDA ライブラリ
add_library(gn10_cuda_lib
  src/cuda/ground_filter.cu
  src/cuda/field_objects.cu
)
set_target_properties(gn10_cuda_lib PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# nvcc 内部で生成されるコードの -Wpedantic 警告を抑制
target_compile_options(gn10_cuda_lib PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wno-pedantic>
)

# C++ ノードのビルド
add_executable(gn10_pointcloud_localization_node
  src/localization_node.cpp
  src/main.cpp
)

# C++ (g++) 側のみにコンパイルオプションを適用
target_compile_options(gn10_pointcloud_localization_node PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
)

target_link_libraries(gn10_pointcloud_localization_node
  gn10_cuda_lib
  ${CERES_LIBRARIES}
  CUDA::cudart
)

ament_target_dependencies(gn10_pointcloud_localization_node
  rclcpp
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  tf2_geometry_msgs
  visualization_msgs
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