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
        # MUST match the config's publish.body_frame: the map -> base edge is
        # composed through body -> base, so a mismatched body frame either
        # fails to resolve (no TF at all) or composes through the wrong mount.
        DeclareLaunchArgument('body_frame', default_value='camera_imu_optical_frame',
            description='Frame the filter estimates. camera_imu_optical_frame '
                        'for l2lidar_rsimu.yaml, l2lidar_frame_imu for '
                        'l2lidar_node.yaml.'),
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
        DeclareLaunchArgument('sc_dist_thres', default_value='0.12',
            description='ScanContext match distance threshold.'),
        # A single ScanContext hit in a corridor is not evidence.
        DeclareLaunchArgument('init_agree_count', default_value='2',
            description='Independent ScanContext locks required to agree.'),
        DeclareLaunchArgument('init_agree_dist', default_value='1.0',
            description='Metres within which agreeing locks must match.'),
        DeclareLaunchArgument('lock_verify_scans', default_value='5',
            description='Consecutive live scans a candidate must clear '
                        'init_min_overlap on before the filter is teleported '
                        'and the prior map swapped in. Until then nothing is '
                        'committed and a bad candidate costs nothing.'),
        DeclareLaunchArgument('max_relock_attempts', default_value='2',
            description='Locks claimed and then lost before the automatic '
                        'search gives up and waits for a manual /initialpose. '
                        '0 retries forever.'),
        DeclareLaunchArgument('max_speed', default_value='1.0',
            description="Metres/second the platform cannot exceed (Pepper is "
                        "~0.55). Above it the estimate is diverging, not "
                        "moving, and velocity plus the IMU states are zeroed."),
        DeclareLaunchArgument('no_match_duration', default_value='1.0',
            description='Seconds of zero effective points before a lock is '
                        'dropped and a live map restored.'),
        DeclareLaunchArgument('odom_init_frame', default_value='lio_init',
            description="/Odometry's frame_id before a lock, when the filter "
                        "is in its own frame rather than the map's."),
        # Used twice: as the admission gate on a fresh registration, and as
        # the per-scan bar during verification. Kept at 0.70: tightening
        # init_overlap_dist already lowers every score, so raising this too
        # double-counted it and starved the search (MEASURED on FAST_LIO).
        DeclareLaunchArgument('init_min_overlap', default_value='0.70',
            description='Minimum fraction of the scan that must overlap the '
                        'map at the proposed pose.'),
        # The single most important number for not locking to the wrong
        # place: the radius within which a scan point counts as "on the map".
        # At 0.20 a pose the filter could not register against at all still
        # scored 78-99%, because in a structured room a wrong pose puts most
        # points within 20 cm of SOMETHING.
        DeclareLaunchArgument('init_overlap_dist', default_value='0.12',
            description='Metres. Keep TIGHT -- looser values let a wrong lock '
                        'still score a high overlap.'),
        # Post-lock health check: re-scores the live pose against the map,
        # because a wrong-but-self-consistent lock has no other symptom.
        DeclareLaunchArgument('health_min_overlap', default_value='0.45',
            description='Overlap below this counts as unhealthy.'),
        DeclareLaunchArgument('health_bad_duration', default_value='5.0',
            description='Seconds of sustained unhealthy overlap before the '
                        'diagnostic escalates to ERROR. Report-only: recovery '
                        'is /initialpose, /relocalize, or the Nav2 watchdog.'),
        DeclareLaunchArgument('health_check_period', default_value='1.0',
            description='Seconds between health checks.'),
    ]

    node = Node(
        package='point_lio', executable='pointlio_localization',
        name='point_lio_localization', output='screen',
        parameters=[
            os.path.join(share, 'config', 'l2lidar_rsimu.yaml'),
            {'use_sim_time': LaunchConfiguration('use_sim_time'),
             'publish.tf_child_frame': LaunchConfiguration('tf_child_frame'),
             # After the lock the estimate is in the prior map's frame, so the
             # published header frame is 'map', not Point-LIO's own start frame.
             'odom_header_frame_id': 'map',
             'odom_child_frame_id': LaunchConfiguration('body_frame'),
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
             # arrives as a string, which a typed parameter rejects.
             'localization.lock_verify_scans': ParameterValue(
                 LaunchConfiguration('lock_verify_scans'), value_type=int),
             'localization.max_relock_attempts': ParameterValue(
                 LaunchConfiguration('max_relock_attempts'), value_type=int),
             'localization.max_speed': ParameterValue(
                 LaunchConfiguration('max_speed'), value_type=float),
             'localization.no_match_duration': ParameterValue(
                 LaunchConfiguration('no_match_duration'), value_type=float),
             'publish.odom_init_frame': LaunchConfiguration('odom_init_frame'),
             'localization.init_min_overlap': LaunchConfiguration('init_min_overlap'),
             'localization.init_overlap_dist': LaunchConfiguration('init_overlap_dist'),
             'localization.health_min_overlap': LaunchConfiguration('health_min_overlap'),
             'localization.health_bad_duration': LaunchConfiguration('health_bad_duration'),
             'localization.health_check_period': LaunchConfiguration('health_check_period')},
        ])

    rviz = Node(
        package='rviz2', executable='rviz2', name='rviz2', output='screen',
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time')}],
        # The localization view (prior map, candidate being tested, live scan
        # red while searching / green once locked), not the mapping one.
        arguments=['-d', os.path.join(share, 'rviz_cfg', 'pointlio_localization.rviz')])

    return LaunchDescription(args + [node, rviz])
