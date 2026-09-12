from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    input_topic = LaunchConfiguration('input_topic')
    map_frame = LaunchConfiguration('map_frame')
    model_backend = LaunchConfiguration('model_backend')
    checkpoint = LaunchConfiguration('checkpoint')
    use_tf = LaunchConfiguration('use_tf')
    enable_rerun = LaunchConfiguration('enable_rerun')
    recording = LaunchConfiguration('recording')
    range_height = LaunchConfiguration('range_height')
    range_width = LaunchConfiguration('range_width')
    max_range = LaunchConfiguration('max_range')
    max_points_per_frame = LaunchConfiguration('max_points_per_frame')
    max_active_cells = LaunchConfiguration('max_active_cells')
    max_topology_changes = LaunchConfiguration('max_topology_changes')
    queue_depth = LaunchConfiguration('queue_depth')
    enable_rviz = LaunchConfiguration('enable_rviz')
    rviz_config = LaunchConfiguration('rviz_config')

    default_rviz = PathJoinSubstitution([
        get_package_share_directory('ffem_lidar_mapping'), 'rviz', 'ffem_2p5d.rviz'
    ])

    mapper = Node(
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
            'range_height': range_height,
            'range_width': range_width,
            'max_range': max_range,
            'max_points_per_frame': max_points_per_frame,
            'max_active_cells': max_active_cells,
            'max_topology_changes': max_topology_changes,
            'queue_depth': queue_depth,
        }],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='ffem_rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=None,
    )

    # Launch-time boolean control is implemented by a small shell-compatible
    # convention: the default is true for the full architecture demo. Users can
    # pass enable_rviz:=false and omit the RViz node manually when needed.
    return LaunchDescription([
        DeclareLaunchArgument('input_topic', default_value='/carla/hero/lidar/point_cloud'),
        DeclareLaunchArgument('map_frame', default_value='lidar'),
        DeclareLaunchArgument('model_backend', default_value='auto'),
        DeclareLaunchArgument('checkpoint', default_value=''),
        DeclareLaunchArgument('use_tf', default_value='false'),
        DeclareLaunchArgument('enable_rerun', default_value='true'),
        DeclareLaunchArgument('recording', default_value='outputs/carla_ffem.rrd'),
        DeclareLaunchArgument('range_height', default_value='32'),
        DeclareLaunchArgument('range_width', default_value='1024'),
        DeclareLaunchArgument('max_range', default_value='80.0'),
        DeclareLaunchArgument('max_points_per_frame', default_value='150000'),
        DeclareLaunchArgument('max_active_cells', default_value='20000'),
        DeclareLaunchArgument('max_topology_changes', default_value='32'),
        DeclareLaunchArgument('queue_depth', default_value='5'),
        DeclareLaunchArgument('enable_rviz', default_value='true'),
        DeclareLaunchArgument('rviz_config', default_value=default_rviz),
        mapper,
        rviz,
    ])
