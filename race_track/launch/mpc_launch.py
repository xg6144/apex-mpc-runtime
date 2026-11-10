"""Launch only the MPC node against an already running vehicle/simulator."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('race_track_mpc')
    config = os.path.join(package_share, 'config', 'mpc.yaml')
    default_raceline = os.path.join(
        package_share, 'race_track',
        'track_file/ifac_roboracer_raceline_apex.csv')
    default_map = os.path.join(
        package_share, 'race_track', 'track_file/ifac_roboracer.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('raceline_path', default_value=default_raceline),
        DeclareLaunchArgument('map_yaml_path', default_value=default_map),
        DeclareLaunchArgument('raceline_editor', default_value='false'),
        Node(
            package='race_track_mpc',
            executable='mpc_controller',
            name='mpc_controller',
            output='screen',
            parameters=[config, {
                'raceline_path': LaunchConfiguration('raceline_path'),
                'map_yaml_path': LaunchConfiguration('map_yaml_path'),
                'enable_raceline_editor': LaunchConfiguration('raceline_editor'),
            }],
        ),
    ])
