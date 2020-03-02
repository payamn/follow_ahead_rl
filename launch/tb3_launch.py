import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, LogInfo)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, TextSubstitution


def generate_launch_description():
    # Get the launch directory
    follow_ahead_dir = get_package_share_directory('follow_ahead_rl')
    launch_dir = os.path.join(follow_ahead_dir, 'launch')

    # Names and poses of the robots
    robots = [
        {'name': 'robot1', 'x_pose': 0.0, 'y_pose': 0.5, 'z_pose': 0.01},
        {'name': 'robot2', 'x_pose': 0.0, 'y_pose': -0.5, 'z_pose': 0.01}]

    # Simulation settings
    world = LaunchConfiguration('world')
    simulator = LaunchConfiguration('simulator')

    # On this example all robots are launched with the same settings
    map_yaml_file = LaunchConfiguration('map')

    bt_xml_file = LaunchConfiguration('bt_xml_file')
    autostart = LaunchConfiguration('autostart')
    rviz_config_file = LaunchConfiguration('rviz_config')
    use_robot_state_pub = LaunchConfiguration('use_robot_state_pub')
    use_rviz = LaunchConfiguration('use_rviz')
    log_settings = LaunchConfiguration('log_settings', default='true')

    # Declare the launch arguments
    declare_world_cmd = DeclareLaunchArgument(
        'world',
        default_value=os.path.join(follow_ahead_dir, 'worlds', 'world_only.model'),
        description='Full path to world file to load')

    declare_simulator_cmd = DeclareLaunchArgument(
        'simulator',
        default_value='gazebo',
        description='The simulator to use (gazebo or gzserver)')

    declare_map_yaml_cmd = DeclareLaunchArgument(
        'map',
        default_value=os.path.join(follow_ahead_dir, 'maps', 'turtlebot3_world.yaml'),
        description='Full path to map file to load')

    declare_robot1_params_file_cmd = DeclareLaunchArgument(
        'robot1_params_file',
        default_value=os.path.join(follow_ahead_dir, 'params', 'nav2_multirobot_params_1.yaml'),
        description='Full path to the ROS2 parameters file to use for robot1 launched nodes')

    declare_robot2_params_file_cmd = DeclareLaunchArgument(
        'robot2_params_file',
        default_value=os.path.join(follow_ahead_dir, 'params', 'nav2_multirobot_params_2.yaml'),
        description='Full path to the ROS2 parameters file to use for robot2 launched nodes')

    declare_rviz_config_file_cmd = DeclareLaunchArgument(
        'rviz_config',
        default_value=os.path.join(follow_ahead_dir, 'rviz', 'nav2_namespaced_view.rviz'),
        description='Full path to the RVIZ config file to use.')

    declare_use_robot_state_pub_cmd = DeclareLaunchArgument(
        'use_robot_state_pub',
        default_value='True',
        description='Whether to start the robot state publisher')

    declare_use_rviz_cmd = DeclareLaunchArgument(
        'use_rviz',
        default_value='True',
        description='Whether to start RVIZ')

    # Start Gazebo with plugin providing the robot spawing service
    start_gazebo_cmd = ExecuteProcess(
        cmd=[simulator, '--verbose', '-s', 'libgazebo_ros_factory.so', world],
        output='screen')

    # Define commands for spawing the robots into Gazebo
    spawn_robots_cmds = []
    for robot in robots:
        spawn_robots_cmds.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(os.path.join(follow_ahead_dir, 'launch',
                                                           'spawn_tb3_launch.py')),
                launch_arguments={
                                  'x_pose': TextSubstitution(text=str(robot['x_pose'])),
                                  'y_pose': TextSubstitution(text=str(robot['y_pose'])),
                                  'z_pose': TextSubstitution(text=str(robot['z_pose'])),
                                  'robot_name': robot['name'],
                                  'turtlebot_type': TextSubstitution(text='burger')
                                  }.items()))

   # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_simulator_cmd)
    ld.add_action(declare_world_cmd)
    ld.add_action(declare_map_yaml_cmd)
    ld.add_action(declare_robot1_params_file_cmd)
    ld.add_action(declare_robot2_params_file_cmd)
    ld.add_action(declare_use_rviz_cmd)
    ld.add_action(declare_rviz_config_file_cmd)
    ld.add_action(declare_use_robot_state_pub_cmd)

    # Add the actions to start gazebo, robots and simulations
    ld.add_action(start_gazebo_cmd)

    for spawn_robot_cmd in spawn_robots_cmds:
        ld.add_action(spawn_robot_cmd)


    return ld
