from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    input_topic=LaunchConfiguration('input_topic'); map_frame=LaunchConfiguration('map_frame'); model_backend=LaunchConfiguration('model_backend'); checkpoint=LaunchConfiguration('checkpoint')
    return LaunchDescription([
        DeclareLaunchArgument('input_topic', default_value='/carla/ego_vehicle/lidar'),
        DeclareLaunchArgument('map_frame', default_value='map'),
        DeclareLaunchArgument('model_backend', default_value='auto'),
        DeclareLaunchArgument('checkpoint', default_value=''),
        Node(package='ffem_lidar_mapping', executable='ffem_node', name='ffem_mapper', output='screen', parameters=[{'input_topic': input_topic, 'map_frame': map_frame, 'model_backend': model_backend, 'checkpoint': checkpoint, 'use_tf': True, 'enable_rerun': False}]),
        Node(package='ffem_lidar_mapping', executable='ffem_rerun_logger', name='ffem_rerun_logger', output='screen', parameters=[{'input_topic': '/ffem_mapper/map/elevation', 'moving_topic': '/ffem_mapper/map/moving_points', 'metrics_topic': '/ffem_mapper/metrics'}]),
    ])
