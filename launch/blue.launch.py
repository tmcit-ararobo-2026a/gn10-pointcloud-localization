import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # パッケージの share ディレクトリを取得
    pkg_dir = get_package_share_directory('gn10_pointcloud_localization')
    
    # デフォルトのパラメータファイルパス
    default_param_file = os.path.join(pkg_dir, 'config', 'blue_params.yaml')

    # Launch 引数の定義
    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='false',
        description='Use simulation (Gazebo/Rosbag) clock if true'
    )

    declare_params_file = DeclareLaunchArgument(
        'params_file',
        default_value=default_param_file,
        description='Full path to the ROS2 parameters file to use'
    )

    # ノードの設定
    localization_node = Node(
        package='gn10_pointcloud_localization',
        executable='gn10_pointcloud_localization_node',
        name='gn10_pointcloud_localization_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
        # 必要に応じてトピックのリマップを追加
        # remapping=[
        #     ('/livox/lidar', '/custom/lidar'),
        # ]
    )

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
        localization_node,
        livox_tf_node,
        lakibeam_tf_node
    ])