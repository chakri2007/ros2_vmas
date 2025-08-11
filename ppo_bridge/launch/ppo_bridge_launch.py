from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    robot_names = ['tb1','tb2']   # match your tb3_multi_robot config
    return LaunchDescription([
        Node(
            package='ppo_bridge',
            executable='ppo_bridge',
            name='ppo_bridge',
            parameters=[{
                'robot_names': robot_names,
                'model_path': '/home/USER/robot_ws/src/ppo_bridge/models/marl_ppo_inference.pth',
                'control_rate': 10.0,
                'lidar_proc_size': 24,
                'max_linear': 0.22,
                'max_angular': 1.0,
                'use_torchscript': False
            }],
            output='screen'
        )
    ])
