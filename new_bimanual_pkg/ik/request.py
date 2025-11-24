from moveit_msgs.srv import GetPositionIK
from builtin_interfaces.msg import Duration
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

def _ik_request_async(self, group, frame_id, ik_link_name, pos, quat,
                        seed_names, seed_values, timeout=0.8, avoid_collisions=True):
        req = GetPositionIK.Request()
        req.ik_request.group_name = group
        req.ik_request.ik_link_name = ik_link_name
        req.ik_request.avoid_collisions = bool(avoid_collisions)

        ps = PoseStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = map(float, pos)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, quat)
        req.ik_request.pose_stamped = ps

        # MoveIt의 IK 내부 타임아웃
        req.ik_request.timeout = Duration(
            sec=int(timeout),
            nanosec=int((timeout - int(timeout)) * 1e9)
        )

        # 시드
        js = JointState()
        js.name = list(seed_names)
        js.position = [float(v) for v in seed_values]
        req.ik_request.robot_state.joint_state = js

        # 바로 비동기 요청만 보내고, 여기서는 절대 기다리지 않음
        return self.ik_cli.call_async(req)
