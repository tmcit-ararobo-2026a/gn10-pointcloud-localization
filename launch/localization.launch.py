# Copyright 2026 Gento Aiba and contributors
# SPDX-License-Identifier: Apache-2.0

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def _create_localization_node(context):
    pkg_dir = get_package_share_directory('gn10_pointcloud_localization')
    common_param_file = os.path.join(pkg_dir, 'config', 'common_params.yaml')

    use_sim_time_str = LaunchConfiguration('use_sim_time').perform(context).lower()
    use_sim_time = use_sim_time_str in ('true', '1')

    params_file = LaunchConfiguration('params_file').perform(context)
    default_param_file = os.path.join(pkg_dir, 'config', 'localization_params.yaml')

    if not params_file:
        params_file = default_param_file
    elif not os.path.isabs(params_file) and not os.path.exists(params_file):
        candidate = os.path.join(pkg_dir, 'config', params_file)
        if os.path.exists(candidate):
            params_file = candidate

    overrides = {'use_sim_time': use_sim_time}

    # 1. Map file & source type overrides
    map_arg = LaunchConfiguration('map').perform(context).strip()
    map_type_arg = LaunchConfiguration('map_type').perform(context).strip().lower()

    if map_arg:
        if not os.path.isabs(map_arg) and not os.path.exists(map_arg):
            candidate = os.path.join(pkg_dir, 'map', map_arg)
            if os.path.exists(candidate):
                map_arg = candidate
        overrides['map_file_path'] = map_arg
        if not map_type_arg:
            ext = os.path.splitext(map_arg)[1].lower().lstrip('.')
            if ext in ('pcd', 'json', 'esdf'):
                map_type_arg = ext

    if map_type_arg:
        overrides['map_source_type'] = map_type_arg

    # 2. Matching parameters overrides
    def _override_float(arg_name, param_key):
        val = LaunchConfiguration(arg_name).perform(context).strip()
        if val:
            try:
                overrides[param_key] = float(val)
            except ValueError:
                pass

    def _override_bool(arg_name, param_key):
        val = LaunchConfiguration(arg_name).perform(context).strip().lower()
        if val in ('true', '1'):
            overrides[param_key] = True
        elif val in ('false', '0'):
            overrides[param_key] = False

    _override_float('max_dist_thresh', 'matching_params.max_dist_thresh')
    _override_float('cost_threshold', 'matching_params.cost_threshold')
    _override_float('dynamic_dist_thresh', 'matching_params.dynamic_dist_thresh')

    # 3. Sensor integrations
    _override_bool('lidar_2d', 'lidar_2d.enable')
    _override_bool('realsense', 'realsense.enable')

    # 4. Initial pose overrides
    _override_bool('initial_local_search', 'initial_pose.use_for_local_search')
    _override_float('initial_x', 'initial_pose.x')
    _override_float('initial_y', 'initial_pose.y')
    _override_float('initial_yaw', 'initial_pose.yaw')

    # 5. Benchmark profiler overrides
    _override_bool('benchmark', 'benchmark.enable')
    bench_csv = LaunchConfiguration('benchmark_csv').perform(context).strip()
    if bench_csv:
        overrides['benchmark.csv_output_path'] = bench_csv

    return [
        Node(
            package='gn10_pointcloud_localization',
            executable='gn10_pointcloud_localization_node',
            name='gn10_pointcloud_localization_node',
            output='screen',
            parameters=[
                common_param_file,
                params_file,
                overrides
            ],
        )
    ]

def generate_launch_description():
    pkg_dir = get_package_share_directory('gn10_pointcloud_localization')
    default_param_file = os.path.join(pkg_dir, 'config', 'localization_params.yaml')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo/Rosbag) clock if true'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_param_file,
        description='Full path or filename in config/ of profile YAML to overlay on common_params.yaml'
    )

    # --- Map Arguments ---
    declare_map = DeclareLaunchArgument(
        'map',
        default_value='',
        description='Path or filename in map/ of map file (json/pcd/esdf). Overrides map_file_path.'
    )

    declare_map_type = DeclareLaunchArgument(
        'map_type',
        default_value='',
        description='Map source type (json, pcd, esdf). Inferred from file extension if empty.'
    )

    # --- Matching Threshold Overrides ---
    declare_max_dist_thresh = DeclareLaunchArgument(
        'max_dist_thresh',
        default_value='',
        description='Override matching_params.max_dist_thresh'
    )

    declare_cost_threshold = DeclareLaunchArgument(
        'cost_threshold',
        default_value='',
        description='Override matching_params.cost_threshold'
    )

    declare_dynamic_dist_thresh = DeclareLaunchArgument(
        'dynamic_dist_thresh',
        default_value='',
        description='Override matching_params.dynamic_dist_thresh'
    )

    # --- Sensor switches ---
    declare_realsense = DeclareLaunchArgument(
        'realsense',
        default_value='true',
        description='Enable RealSense ESDF dynamic background filter (default: true)'
    )

    declare_lidar_2d = DeclareLaunchArgument(
        'lidar_2d',
        default_value='',
        description='Enable 2D LiDAR (Lakibeam) integration (true/false)'
    )

    declare_lakibeam_tf = DeclareLaunchArgument(
        'start_lakibeam_tf',
        default_value='false',
        description='Publish static TF for rear Lakibeam 1 LiDAR (base_link -> lakibeam_frame)'
    )

    # --- Initial Pose Overrides ---
    declare_initial_local_search = DeclareLaunchArgument(
        'initial_local_search',
        default_value='',
        description='Override initial_pose.use_for_local_search (true/false)'
    )

    declare_initial_x = DeclareLaunchArgument(
        'initial_x',
        default_value='',
        description='Override initial_pose.x'
    )

    declare_initial_y = DeclareLaunchArgument(
        'initial_y',
        default_value='',
        description='Override initial_pose.y'
    )

    declare_initial_yaw = DeclareLaunchArgument(
        'initial_yaw',
        default_value='',
        description='Override initial_pose.yaw'
    )

    # --- Benchmark Profiler Overrides ---
    declare_benchmark = DeclareLaunchArgument(
        'benchmark',
        default_value='',
        description='Enable benchmark profiler (true/false)'
    )

    declare_benchmark_csv = DeclareLaunchArgument(
        'benchmark_csv',
        default_value='',
        description='Path to benchmark CSV output file'
    )

    localization_node = OpaqueFunction(function=_create_localization_node)

    livox_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='livox_tf',
        arguments=[
            '--x', '0.24',
            '--y', '-0.25',
            '--z', '1.09',
            '--yaw', '0.0',
            '--pitch', '-0.273',
            '--roll', '3.13',
            '--frame-id', 'base_link',
            '--child-frame-id', 'livox_frame'
        ]
    )

    lakibeam_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='lakibeam_tf',
        condition=IfCondition(LaunchConfiguration('start_lakibeam_tf')),
        arguments=[
            '--x', '-0.4',
            '--y', '0.0',
            '--z', '0.08',
            '--yaw', '3.14159',
            '--pitch', '0.0',
            '--roll', '0.0',
            '--frame-id', 'base_link',
            '--child-frame-id', 'lakibeam_frame'
        ]
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_params_file,
        declare_map,
        declare_map_type,
        declare_max_dist_thresh,
        declare_cost_threshold,
        declare_dynamic_dist_thresh,
        declare_realsense,
        declare_lidar_2d,
        declare_lakibeam_tf,
        declare_initial_local_search,
        declare_initial_x,
        declare_initial_y,
        declare_initial_yaw,
        declare_benchmark,
        declare_benchmark_csv,
        localization_node,
        livox_tf_node,
        lakibeam_tf_node
    ])