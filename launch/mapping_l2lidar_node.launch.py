import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import GroupAction, DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the RViz argument
    config_file_arg = DeclareLaunchArgument(
        'config_file', default_value='l2lidar_rsimu.yaml',
        description='Config under point_lio/config. l2lidar_node.yaml uses the '
                    "L2's own IMU; l2lidar_rsimu.yaml uses the RealSense's "
                    '(/camera/imu) -- see utils/L2_IMU/REPORT.md for why.'
    )
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

    # Node parameters, including those from the YAML configuration file
    laser_mapping_params = [
        PathJoinSubstitution([
            FindPackageShare('point_lio'),
            'config', LaunchConfiguration('config_file')
        ]),
        {
            # Output model (IMU as measurement, Point-LIO's native mode). A/B on
            # slam_20260823 vs the input model: APE 1.06 -> 0.31 m, RPE 4.9 -> 1.0 %/10 m.
            # The earlier note that the input model was more robust to the L2's
            # IMU predates the RealSense-IMU config; re-check if l2lidar_node.yaml is used.
            'use_imu_as_input': False,
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
            # harmless orphan branch) -- lio_odom_bridge below does
            # the real odom -> base_footprint republish instead.
            'odom_header_frame_id': 'lio_init',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }
    ]

    # Node definition for laserMapping with Point-LIO
    laser_mapping_node = Node(
        package='point_lio',
        executable='pointlio_mapping',
        # Standard odometry topic across every LIO variant: FAST-LIO publishes
        # /Odometry natively, Point-LIO and FAST-LIVO2 /aft_mapped_to_init.
        # Each mapping launch remaps its own to /odom_lio so consumers need not
        # know which estimator is running.
        remappings=[('/aft_mapped_to_init', '/odom_lio')],
        name='laserMapping',
        output='screen',
        parameters=laser_mapping_params,
        # prefix='gdb -ex run --args'
    )

    # Republishes Point-LIO's odometry (lio_init -> <body frame>) as
    # odom -> base_footprint through pepper_slam's lio_odom_bridge.launch.py,
    # the same include FAST-LIO's callers use, so both estimators get one
    # bridge setup: parameters from pepper_slam_configuration.yaml, the body
    # frame from lidar_imu_frame (default camera_imu_optical_frame, the frame
    # l2lidar_rsimu.yaml names -- an L2-IMU run with l2lidar_node.yaml must
    # pass lidar_imu_frame:=l2lidar_frame_imu, as with FAST-LIO), and the
    # overrides it declares (flatten_base_frame, bridge_level_frame,
    # level_frame_as_child, ...) taken from the caller when set and from the
    # configuration file otherwise.
    odom_bridge_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('pepper_slam'),
            'launch', 'lio_odom_bridge.launch.py')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'config_path': PathJoinSubstitution([
                FindPackageShare('point_lio'), 'config']),
            'config_file': LaunchConfiguration('config_file'),
        }.items(),
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
        config_file_arg,
        rviz_arg,
        rviz_cfg_arg,
        use_sim_time_arg,
        laser_mapping_node,
        odom_bridge_launch,
        GroupAction(
            actions=[rviz_node],
            condition=IfCondition(LaunchConfiguration('rviz'))
        ),
    ])

    return ld
