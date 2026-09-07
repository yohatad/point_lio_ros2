# POINT-LIO LOCALIZATION on the L2 rig: localize against a prior map built by
# fastlio_lc_pgo and converted with the ScanContext map converter.
#
# The Point-LIO twin of fast_lio/launch/localization_l2.launch.py -- same map
# format, same arguments, same defaults, so the two backends can be compared by
# swapping one launch file. See point_lio/src/laserLocalization.cpp's header for
# what differs inside.
#
#   ros2 launch point_lio localization_l2.launch.py
#
# then, for a bag:
#   ros2 bag play <bag> --clock \
#       --qos-profile-overrides-path config/play_qos.yaml \
#       --read-ahead-queue-size 2000
#
# The QoS overrides are REQUIRED: /imu/data and /camera/imu were recorded
# BEST_EFFORT and a RELIABLE subscriber matches nothing against them.
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('point_lio')

    args = [
        # Split deliberately: the pose file is small and git-tracked; the
        # keyframe clouds are 75 MB of gitignored binary.
        DeclareLaunchArgument('map_dir',
            default_value=os.path.join(
                get_package_share_directory('pepper_navigation'), 'pcd'),
            description='Directory holding the pose file.'),
        DeclareLaunchArgument('map_pose_file', default_value='sc_pose_20260823.json',
            description='Pose file within map_dir (or an absolute path).'),
        # Named after the RUN, not a bare pcd/: the pose file indexes these by
        # number, so two undifferentiated folders would be silently
        # interchangeable.
        DeclareLaunchArgument('map_scan_dir',
            default_value=os.path.join(
                get_package_share_directory('pepper_navigation'),
                'pcd', 'sc_pcd_20260823'),
            description='Directory holding the per-keyframe <N>.pcd clouds.'),
        DeclareLaunchArgument('config_file', default_value='l2lidar_rsimu.yaml',
            description='Point-LIO config. Must be the SAME one the map was '
                        'built with -- the lidar/IMU extrinsic enters the '
                        'initial pose composition.'),
        DeclareLaunchArgument('use_sim_time', default_value='true',
            description='true for bag replay (this launch is bag-oriented).'),
        DeclareLaunchArgument('rviz', default_value='true',
            description='Open RViz2 pre-configured for this stack.'),
        DeclareLaunchArgument('tf_child_frame', default_value='base_footprint',
            description='Child of the broadcast map edge. The body->child '
                        'extrinsic is read from /tf_static once and cached.'),
        # ScanContext descriptor geometry, sized to the L2 rather than
        # upstream's 64-beam car lidar.
        DeclareLaunchArgument('sc_max_radius', default_value='10.0',
            description='ScanContext descriptor radius, metres.'),
        DeclareLaunchArgument('sc_num_ring', default_value='12',
            description='ScanContext descriptor ring count.'),
        DeclareLaunchArgument('sc_num_sector', default_value='40',
            description='ScanContext descriptor sector count.'),
        DeclareLaunchArgument('sc_lidar_height', default_value='0.5',
            description='Lidar height above ground, metres.'),
        DeclareLaunchArgument('sc_dist_thres', default_value='0.15',
            description='ScanContext match distance threshold.'),
        # A single ScanContext hit in a corridor is not evidence.
        DeclareLaunchArgument('init_agree_count', default_value='2',
            description='Independent ScanContext locks required to agree.'),
        DeclareLaunchArgument('init_agree_dist', default_value='2.0',
            description='Metres within which agreeing locks must match.'),
        # Agreement alone can't catch two matches that agree on the SAME wrong
        # place. require_motion forces the two estimates apart in space so
        # agreement means something; off by default since a seeded /initialpose
        # start doesn't need it. Turn on for unattended startup with no seed.
        DeclareLaunchArgument('init_require_motion', default_value='false',
            description='Require motion between agreeing estimates before '
                        'accepting a lock.'),
        DeclareLaunchArgument('init_motion_min', default_value='0.50',
            description='Metres of odometry required between the agreeing '
                        'estimates. Only used when init_require_motion.'),
        DeclareLaunchArgument('init_min_overlap', default_value='0.70',
            description='Minimum fraction of the scan that must overlap the '
                        'map at the proposed pose.'),
        DeclareLaunchArgument('init_overlap_dist', default_value='0.20',
            description='Metres. Keep TIGHT -- looser values let a wrong lock '
                        'still score a high overlap.'),
        # Post-lock health check: re-scores the live pose against the map,
        # because a wrong-but-self-consistent lock has no other symptom.
        DeclareLaunchArgument('health_min_overlap', default_value='0.45',
            description='Overlap below this counts as unhealthy.'),
        DeclareLaunchArgument('health_bad_duration', default_value='5.0',
            description='Seconds of sustained unhealthy overlap before the '
                        'search is re-armed. Short dips are normal.'),
        DeclareLaunchArgument('health_check_period', default_value='1.0',
            description='Seconds between health checks.'),
        DeclareLaunchArgument('auto_relocalize', default_value='true',
            description='Re-arm the search automatically on sustained bad '
                        'overlap. Off leaves it to /relocalize.'),
    ]

    node = Node(
        package='point_lio', executable='pointlio_localization',
        name='point_lio_localization', output='screen',
        parameters=[
            os.path.join(share, 'config', 'l2lidar_rsimu.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time'),
             'publish.tf_child_frame': LaunchConfiguration('tf_child_frame'),
             'localization.map_dir': LaunchConfiguration('map_dir'),
             'localization.map_scan_dir': LaunchConfiguration('map_scan_dir'),
             'localization.map_pose_file': LaunchConfiguration('map_pose_file'),
             'localization.sc_max_radius': LaunchConfiguration('sc_max_radius'),
             'localization.sc_num_ring': LaunchConfiguration('sc_num_ring'),
             'localization.sc_num_sector': LaunchConfiguration('sc_num_sector'),
             'localization.sc_lidar_height': LaunchConfiguration('sc_lidar_height'),
             'localization.sc_dist_thres': LaunchConfiguration('sc_dist_thres'),
             'localization.init_agree_count': LaunchConfiguration('init_agree_count'),
             'localization.init_agree_dist': LaunchConfiguration('init_agree_dist'),
             # ParameterValue with an explicit type: a bare LaunchConfiguration
             # arrives as a string, which the node's bool parameter rejects.
             'localization.init_require_motion': ParameterValue(
                 LaunchConfiguration('init_require_motion'), value_type=bool),
             'localization.init_motion_min': LaunchConfiguration('init_motion_min'),
             'localization.init_min_overlap': LaunchConfiguration('init_min_overlap'),
             'localization.init_overlap_dist': LaunchConfiguration('init_overlap_dist'),
             'localization.health_min_overlap': LaunchConfiguration('health_min_overlap'),
             'localization.health_bad_duration': LaunchConfiguration('health_bad_duration'),
             'localization.health_check_period': LaunchConfiguration('health_check_period'),
             'localization.auto_relocalize': ParameterValue(
                 LaunchConfiguration('auto_relocalize'), value_type=bool)},
        ])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        arguments=['-d', os.path.join(share, 'rviz_cfg', 'loam_livox.rviz')])

    return LaunchDescription(args + [node, rviz])
