"""Field localization with configured RViz and native Livox display conversion."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('gn10_pointcloud_localization')
    input_type = LaunchConfiguration('input_cloud_type')
    input_topic = LaunchConfiguration('input_cloud_topic')
    use_sim_time = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('rviz_config', default_value=os.path.join(share, 'rviz', 'fusion.rviz')),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'fast_lio_fusion.launch.py'))),
        Node(
            package='gn10_pointcloud_localization', executable='livox_rviz_bridge',
            name='livox_rviz_bridge', output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'input_topic': input_topic}],
            condition=IfCondition(PythonExpression(["'", input_type, "' == 'custom_msg'"])),
        ),
        Node(
            package='rviz2', executable='rviz2', name='rviz2', output='screen',
            arguments=['-d', LaunchConfiguration('rviz_config')],
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[('/gn10/rviz_cloud', PythonExpression([
                "'", input_topic, "' if '", input_type,
                "' == 'pointcloud2' else '/gn10/rviz_cloud'"
            ]))],
        ),
    ])
