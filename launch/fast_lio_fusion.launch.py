"""Run MID360 PointCloud2 -> FAST-LIO -> field matching -> map pose fusion."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('gn10_pointcloud_localization')
    use_sim_time = LaunchConfiguration('use_sim_time')
    start_fast_lio = LaunchConfiguration('start_fast_lio')
    matcher_config = LaunchConfiguration('matcher_config')
    fusion_config = LaunchConfiguration('fusion_config')
    fast_lio_config = LaunchConfiguration('fast_lio_config')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('start_fast_lio', default_value='true'),
        DeclareLaunchArgument('matcher_config', default_value=os.path.join(
            share, 'config', 'localization_params.yaml')),
        DeclareLaunchArgument('fusion_config', default_value=os.path.join(
            share, 'config', 'fusion_params.yaml')),
        DeclareLaunchArgument('fast_lio_config', default_value=os.path.join(
            share, 'config', 'fast_lio_mid360.yaml')),
        Node(
            package='gn10_pointcloud_localization', executable='livox_pointcloud_bridge',
            name='livox_pointcloud_bridge', output='screen',
            parameters=[{'use_sim_time': use_sim_time}], condition=IfCondition(start_fast_lio),
        ),
        Node(
            package='fast_lio', executable='fastlio_mapping', name='fast_lio',
            output='screen', parameters=[fast_lio_config, {'use_sim_time': use_sim_time}],
            condition=IfCondition(start_fast_lio),
        ),
        Node(
            package='gn10_pointcloud_localization',
            executable='gn10_pointcloud_localization_node',
            name='gn10_pointcloud_localization_node', output='screen',
            parameters=[matcher_config, {
                'use_sim_time': use_sim_time,
                'publish_tf': False,
                'topics.output_pose': '/platform_constraint_raw',
                'topics.fused_prior': '/platform_constraint',
                'fusion.use_prior': True,
            }],
        ),
        Node(
            package='gn10_pointcloud_localization', executable='gn10_pose_fusion_node',
            name='gn10_pose_fusion_node', output='screen',
            parameters=[fusion_config, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='tf2_ros', executable='static_transform_publisher',
            name='livox_tf',
            arguments=[
                '--x', '0.2', '--y', '-0.25', '--z', '1.09',
                '--yaw', '0.0', '--pitch', '-0.273', '--roll', '3.13',
                '--frame-id', 'base_link', '--child-frame-id', 'livox_frame',
            ],
        ),
    ])
