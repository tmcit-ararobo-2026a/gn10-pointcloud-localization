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
    use_sim_time = use_sim_time_str == 'true'
    realsense_str = LaunchConfiguration('realsense').perform(context).lower()
    enable_realsense = realsense_str == 'true'

    params_file = LaunchConfiguration('params_file').perform(context)

    # If user enabled realsense and did not explicitly specify a non-default params_file:
    default_param_file = os.path.join(pkg_dir, 'config', 'localization_params.yaml')
    if enable_realsense and params_file in ('', default_param_file, 'localization_params.yaml'):
        target_name = 'unity_realsense_filter_params.yaml' if use_sim_time else 'realsense_filter_params.yaml'
        params_file = os.path.join(pkg_dir, 'config', target_name)
    elif not os.path.isabs(params_file) and not os.path.exists(params_file):
        candidate = os.path.join(pkg_dir, 'config', params_file)
        if os.path.exists(candidate):
            params_file = candidate

    overrides = {'use_sim_time': use_sim_time}
    if enable_realsense:
        overrides['realsense.enable'] = True

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

    declare_realsense = DeclareLaunchArgument(
        'realsense',
        default_value='true',
        description='Enable RealSense ESDF dynamic background filter (default: true)'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_param_file,
        description='Full path or filename in config/ of parameter YAML to use'
    )

    declare_lakibeam_tf = DeclareLaunchArgument(
        'start_lakibeam_tf',
        default_value='false',
        description='Publish static TF for rear Lakibeam 1 LiDAR (base_link -> lakibeam_frame)'
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
        declare_realsense,
        declare_params_file,
        declare_lakibeam_tf,
        localization_node,
        livox_tf_node,
        lakibeam_tf_node
    ])