# MuJoCo에 모델 스폰 (ctrl 기반)

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from ament_index_python.packages import get_package_share_directory
import mujoco
import mujoco.viewer


class MujocoROSNode(Node):
    def __init__(self, model, data):
        super().__init__('mujoco_node')
        self.model = model
        self.data = data

        # actuator 이름 → ctrl 인덱스 매핑
        self.actuator_name_to_id = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(model.nu)
        }

        self.subscription = self.create_subscription(
            JointState,
            '/desired_joint_angles',
            self.joint_callback,
            10
        )
        self.get_logger().info('Subscribed to /desired_joint_angles')

    def joint_callback(self, msg: JointState):

        for name, position in zip(msg.name, msg.position):
            if name in self.actuator_name_to_id:
                act_id = self.actuator_name_to_id[name]
                self.data.ctrl[act_id] = position  # actuator 입력으로 전달
        # ctrl은 mj_step() 할 때 physics에 적용됨


def main():
    rclpy.init()

    # 모델 로딩
    pkg_path = get_package_share_directory('new_bimanual_pkg')
    model_path = os.path.join(pkg_path, 'mujoco_models', 'bimanual.xml')
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    node = MujocoROSNode(model, data)

    # 뷰어 실행
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while rclpy.ok() and viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            rclpy.spin_once(node, timeout_sec=0.001)

    node.destroy_node()
    rclpy.shutdown()
    print("[INFO] Viewer closed.")
