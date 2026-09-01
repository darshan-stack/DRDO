from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='ffem_lidar_mapping', executable='ffem_node', name='ffem_mapper', output='screen')
    ])
