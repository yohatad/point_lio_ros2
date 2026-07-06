from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the RViz argument
    rviz_arg = DeclareLaunchArgument(
        'rviz', default_value='true',
        description='Flag to launch RViz.')

    # Node parameters, including those from the YAML configuration file
    laser_mapping_params = [
        PathJoinSubstitution([
            FindPackageShare('point_lio'),
            'config', 'l2lidar_node.yaml'
        ]),
        {
            'use_imu_as_input': False,  # Change to True to use IMU as input of Point-LIO
            'prop_at_freq_of_imu': True,
            'check_satu': True,
            'init_map_size': 10,
            'point_filter_num': 1,  # Options: 1, 3
            'space_down_sample': True,
            'filter_size_surf': 0.1,  # Options: 0.5, 0.3, 0.2, 0.15, 0.1
            'filter_size_map': 0.1,  # Options: 0.5, 0.3, 0.15, 0.1
            'cube_side_length': 1000.0,  # Option: 1000
            'runtime_pos_log_enable': False,  # Option: True
            # 'odom_header_frame_id' feeds the point cloud/path frame_id too,
            # so it must stay "odom". Do NOT set 'odom_child_frame_id' to
            # "l2lidar_imu": Point-LIO's own tf broadcast (laserMapping.cpp,
            # unconditional, no publish_tf-style disable flag) would then
            # fight the static l2lidar_frame -> l2lidar_imu transform for a
            # parent. Left at its "aft_mapped" default (unclaimed frame,
            # harmless orphan branch) -- lio_map_odom_bridge.py below does
            # the real odom -> base_footprint republish instead.
            'odom_header_frame_id': 'odom',
        }
    ]

    # Node definition for laserMapping with Point-LIO
    laser_mapping_node = Node(
        package='point_lio',
        executable='pointlio_mapping',
        name='laserMapping',
        output='screen',
        parameters=laser_mapping_params,
        # prefix='gdb -ex run --args'
    )

    # Republishes Point-LIO's odometry (odom -> aft_mapped, i.e. odom -> l2lidar_imu
    # in physical terms) as odom -> base_footprint, reusing the same bridge
    # FAST-LIO uses -- see FAST_LIO_ROS2/scripts/lio_map_odom_bridge.py for the
    # full explanation of why this indirection exists.
    odom_bridge_node = Node(
        package='fast_lio',
        executable='lio_map_odom_bridge.py',
        name='lio_map_odom_bridge',
        output='screen',
        parameters=[{
            'odom_topic': '/aft_mapped_to_init',
        }],
    )

    # Conditional RViz node launch
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', PathJoinSubstitution([
            FindPackageShare('point_lio'),
            'rviz_cfg', 'loam_livox.rviz'
        ])],
        condition=IfCondition(LaunchConfiguration('rviz')),
        prefix='nice'
    )

    # Assemble the launch description
    ld = LaunchDescription([
        rviz_arg,
        laser_mapping_node,
        odom_bridge_node,
        GroupAction(
            actions=[rviz_node],
            condition=IfCondition(LaunchConfiguration('rviz'))
        ),
    ])

    return ld
