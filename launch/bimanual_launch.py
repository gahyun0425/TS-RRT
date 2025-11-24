# 파일 이름 예: multi_nodes.launch.py

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import Node


def generate_launch_description():

    # 1. basic_spawn_node.py 실행
    basic_spawn_node = Node(
        package='new_bimanual_pkg',          
        executable='basic_node',   
        name='basic_spawn_node',
        output='screen'
    )

    # 2. start_node.py 실행 (basic_spawn_node 시작 후)
    start_node = Node(
        package='new_bimanual_pkg',
        executable='start_node',
        name='start_node',
        output='screen'
    )

    # 3. path_master_node.py 실행 (start_node 시작 후)
    path_master_node = Node(
        package='new_bimanual_pkg',
        executable='path_node',
        name='path_master_node',
        output='screen'
    )

    # basic_spawn_node 가 시작되면 start_node 실행
    start_after_basic = RegisterEventHandler(
        OnProcessStart(
            target_action=basic_spawn_node,
            on_start=[start_node]
        )
    )

    # start_node 가 시작되면 path_master_node 실행
    start_after_start_node = RegisterEventHandler(
        OnProcessStart(
            target_action=start_node,
            on_start=[path_master_node]
        )
    )

    ld = LaunchDescription()
    ld.add_action(basic_spawn_node)
    ld.add_action(start_after_basic)
    ld.add_action(start_after_start_node)

    return ld