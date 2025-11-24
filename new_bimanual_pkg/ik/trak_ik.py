# trak ik 호출

import rclpy 
import numpy as np
import time

from moveit_msgs.srv import GetPositionIK
from geometry_msgs.msg import PoseStamped
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState

def compute_ik_via_moveit(self, group, frame_id, ik_link_name, pos, quat,
                            seed_names, seed_values, timeout=0.2, attempts=8,
                            avoid_collisions=False, wait_mode="spin"):
    
    # 서비스 대기
    if not self.ik_cli.wait_for_service(timeout_sec=5.0):
        self.get_logger().error("'/compute_ik' service not available")
        return None

    # 안전 캐스팅
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    qx, qy, qz, qw = float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])

    # 시드는 리스트로 작업
    seed_names = list(seed_names)
    base_seed  = [float(v) for v in seed_values]

    self.get_logger().info(f"[IK] group={group}, pos={pos}, quat={quat}")

    for k in range(max(1, int(attempts))):
        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.ik_link_name = ik_link_name
        req.ik_request.avoid_collisions = bool(avoid_collisions)

        ps = PoseStamped()
        ps.header.frame_id = frame_id            # ← MoveIt planning frame과 맞추세요 (보통 robot_description의 base)
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x = px
        ps.pose.position.y = py
        ps.pose.position.z = pz
        ps.pose.orientation.x = qx
        ps.pose.orientation.y = qy
        ps.pose.orientation.z = qz
        ps.pose.orientation.w = qw
        req.ik_request.pose_stamped = ps

        req.ik_request.timeout = Duration(sec=2)

        # 시드 (k>0일 때 약간 섞어서 재시도)
        if k == 0:
            seed = base_seed
        else:
            jitter = np.random.normal(scale=1e-3, size=len(base_seed))
            seed = (np.array(base_seed) + jitter).tolist()

        req.ik_request.robot_state.joint_state = JointState()
        req.ik_request.robot_state.joint_state.name = seed_names
        req.ik_request.robot_state.joint_state.position = seed

        # ROS2는 timeout만 지원(필수)
        req.ik_request.timeout = Duration(
            sec=int(timeout),
            nanosec=int((timeout - int(timeout)) * 1e9)
        )

        future = self.ik_cli.call_async(req)

        if wait_mode == "spin":
            # executor가 아직 스핀 전(예: __init__ 단계)에서 사용
            rclpy.spin_until_future_complete(self, future)
        else:
            # 콜백/타이머 안에서 사용 (교착 방지)
            while rclpy.ok() and not future.done():
                time.sleep(0.002)

        res = future.result()

        if res is None:
            self.get_logger().error("IK service call failed (no response)")
            return None

        if res.error_code.val == res.error_code.SUCCESS:
            names = list(res.solution.joint_state.name)
            vals  = list(res.solution.joint_state.position)
            return dict(zip(names, vals))

        # 실패 로그는 첫/마지막 시도에만 간단히
        if k == 0 or k == attempts - 1:
            self.get_logger().warn(f"IK attempt {k+1}/{attempts} failed (error_code={res.error_code.val})")

        # 짧게 쉬고 재시도(선택)
        time.sleep(0.01)

    return None