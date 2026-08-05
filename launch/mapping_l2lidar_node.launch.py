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
    rviz_cfg_arg = DeclareLaunchArgument(
        'rviz_cfg',
        default_value=PathJoinSubstitution([
            FindPackageShare('point_lio'), 'rviz_cfg', 'loam_livox.rviz']),
        description='RViz config file path. Callers that add their own '
                    'displays (e.g. pointlio_lc_l2.launch.py\'s PGO view) '
                    'should override this rather than get the plain default.'
    )

    use_sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='true for bag replay (ros2 bag play --clock); false on the '
                    'robot. Reaches laserMapping and the odom bridge below -- '
                    'without it they run on wall time while the rest of the '
                    'stack runs on the bag clock.')

    flatten_base_frame_arg = DeclareLaunchArgument(
        'flatten_base_frame', default_value='false',
        description='Zero the leveled z/roll/pitch of odom -> base_footprint '
                    'every cycle (keep x, y, yaw) -- a hard flat-floor '
                    'assumption, not a sensor-fused correction. Off by '
                    'default: only correct on robots that are always on '
                    'genuinely flat floor.'
    )
    level_frame_as_child_arg = DeclareLaunchArgument(
        'level_frame_as_child', default_value='false',
        description='Publish the leveling transform as odom_lidar -> odom '
                    '(child) instead of odom -> odom_lidar (parent). Use with '
                    'bridge_level_frame:=true when a localizer already owns '
                    'map -> odom, so odom still exists without giving '
                    'odom two parents.'
    )
    bridge_level_frame_arg = DeclareLaunchArgument(
        'bridge_level_frame', default_value='true',
        description='Have lio_map_odom_bridge publish the static odom -> '
                    'odom leveling frame. Set false when a higher layer owns '
                    'odom (e.g. PGO publishing map -> odom), so odom does not '
                    'end up with two parents (odom AND map).'
    )

    # Node parameters, including those from the YAML configuration file
    laser_mapping_params = [
        PathJoinSubstitution([
            FindPackageShare('point_lio'),
            'config', 'l2lidar_node.yaml'
        ]),
        {
            'use_imu_as_input': True,  # input model (FAST-LIO-style propagation): more robust to the L2's vibration-heavy IMU than the output model
            'prop_at_freq_of_imu': True,
            'check_satu': True,
            'init_map_size': 10,
            'point_filter_num': 3,  # was 1: decimate 3x so processing keeps up in real time (dropped scans made the filter diverge)
            'space_down_sample': True,
            'filter_size_surf': 0.25,  # was 0.1: coarser voxel, large real-time headroom gain
            'filter_size_map': 0.25,  # was 0.1: coarser map voxel, cheaper insertion/search
            'cube_side_length': 1000.0,  # Option: 1000
            'runtime_pos_log_enable': False,  # Option: True
            # 'odom_header_frame_id' feeds the point cloud/path frame_id too,
            # so it must stay "odom". Do NOT set 'odom_child_frame_id' to
            # "l2lidar_frame_imu": Point-LIO's own tf broadcast (laserMapping.cpp,
            # unconditional, no publish_tf-style disable flag) would then
            # fight the static l2lidar_frame -> l2lidar_frame_imu transform for a
            # parent. Left at its "aft_mapped" default (unclaimed frame,
            # harmless orphan branch) -- lio_map_odom_bridge.py below does
            # the real odom -> base_footprint republish instead.
            'odom_header_frame_id': 'odom_lidar',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
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

    # Republishes Point-LIO's odometry (odom -> aft_mapped, i.e. odom -> l2lidar_frame_imu
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
            'flatten_base_frame': LaunchConfiguration('flatten_base_frame'),
            'publish_level_frame': LaunchConfiguration('bridge_level_frame'),
            'level_frame_as_child': LaunchConfiguration('level_frame_as_child'),
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
    )

    # Conditional RViz node launch
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz',
        arguments=['-d', LaunchConfiguration('rviz_cfg')],
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        condition=IfCondition(LaunchConfiguration('rviz')),
        prefix='nice'
    )

    # Assemble the launch description
    ld = LaunchDescription([
        rviz_arg,
        rviz_cfg_arg,
        use_sim_time_arg,
        flatten_base_frame_arg,
        level_frame_as_child_arg,
        bridge_level_frame_arg,
        laser_mapping_node,
        odom_bridge_node,
        GroupAction(
            actions=[rviz_node],
            condition=IfCondition(LaunchConfiguration('rviz'))
        ),
    ])

    return ld
