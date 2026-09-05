from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    input_topic = LaunchConfiguration('input_topic')
    map_frame = LaunchConfiguration('map_frame')
    model_backend = LaunchConfiguration('model_backend')
    checkpoint = LaunchConfiguration('checkpoint')
    use_tf = LaunchConfiguration('use_tf')
    enable_rerun = LaunchConfiguration('enable_rerun')
    recording = LaunchConfiguration('recording')

    return LaunchDescription([
        DeclareLaunchArgument('input_topic', default_value='/carla/hero/lidar/point_cloud'),
        # Native CARLA ROS 2 sensors do not require the official bridge. For the
        # native demo we keep the LiDAR frame as the map frame and disable TF.
        # Set use_tf:=true and map_frame:=map when a valid TF tree is available.
        DeclareLaunchArgument('map_frame', default_value='hero/lidar'),
        DeclareLaunchArgument('model_backend', default_value='auto'),
        DeclareLaunchArgument('checkpoint', default_value=''),
        DeclareLaunchArgument('use_tf', default_value='false'),
        DeclareLaunchArgument('enable_rerun', default_value='true'),
        DeclareLaunchArgument('recording', default_value='outputs/carla_ffem.rrd'),
        Node(
            package='ffem_lidar_mapping',
            executable='ffem_node',
            name='ffem_mapper',
            output='screen',
            parameters=[{
                'input_topic': input_topic,
                'map_frame': map_frame,
                'model_backend': model_backend,
                'checkpoint': checkpoint,
                'use_tf': use_tf,
                'enable_rerun': enable_rerun,
                'recording': recording,
            }],
        ),
    ])
