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
find_package(livox_ros_driver2 REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(diagnostic_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)

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
  nlohmann_json
)

add_executable(livox_pointcloud_bridge src/livox_pointcloud_bridge.cpp)
ament_target_dependencies(livox_pointcloud_bridge rclcpp sensor_msgs livox_ros_driver2)

add_library(gn10_pose_fusion_filter src/pose_fusion_filter.cpp src/odom_projection.cpp)
target_include_directories(gn10_pose_fusion_filter PUBLIC include ${EIGEN3_INCLUDE_DIRS})

add_executable(gn10_pose_fusion_node src/pose_fusion_node.cpp)
target_link_libraries(gn10_pose_fusion_node gn10_pose_fusion_filter)
ament_target_dependencies(gn10_pose_fusion_node
  rclcpp nav_msgs geometry_msgs diagnostic_msgs tf2_ros tf2_eigen
)

if(BUILD_TESTING)
  enable_testing()
  add_executable(test_pose_fusion_filter test/test_pose_fusion_filter.cpp)
  target_link_libraries(test_pose_fusion_filter gn10_pose_fusion_filter)
  add_test(NAME pose_fusion_filter COMMAND test_pose_fusion_filter)
  add_executable(test_odom_projection test/test_odom_projection.cpp)
  target_link_libraries(test_odom_projection gn10_pose_fusion_filter)
  add_test(NAME odom_projection COMMAND test_odom_projection)
endif()

# インストール設定 (Launch, Config, Node)
install(TARGETS 
  gn10_pointcloud_localization_node
  livox_pointcloud_bridge
  gn10_pose_fusion_node
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch config rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
