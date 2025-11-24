# constraint 환경 검사
# joint limit & collision detection

import numpy as np
import rclpy

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState, Constraints # MoveIt robot state
from moveit_msgs.srv import GetStateValidity # 충돌 여부

# 노드 초기화 단계 -> 함수가 여러번 호출되어도 node/client를 매번 만들지 않기 위해
_NODE = None
_CLI = None

# move_group 실행 확인
def _ensure_client(timeout: float = 2.0):
    global _NODE, _CLI
    if not rclpy.ok():                 # 아직 init 안 됐으면
        rclpy.init(args=None)
    if _NODE is None:
        _NODE = rclpy.create_node("state_validity_client")
    if _CLI is None:
        _CLI = _NODE.create_client(GetStateValidity, "/check_state_validity")
        if not _CLI.wait_for_service(timeout_sec=timeout):
            _NODE.get_logger().error("/check_state_validity 서비스가 없습니다. move_group를 먼저 실행하세요.")
            raise RuntimeError("GetStateValidity not available")
        
# joint limit 판단. Moveit planning scene에서 충돌 판단 후 반환
def is_state_valid(
        q: np.ndarray,
        joint_names: list,
        lb: np.ndarray,
        ub: np.ndarray,
        group_name: str = "manipulator",
        timeout: float = 2.0,
        constraints: Constraints | None = None  # ← 추가됨
) -> bool:
    """
    MoveIt의 /check_state_validity 서비스를 통해
    joint limit, collision, 추가 제약(constraints)을 검사한다.
    """
    q = np.asarray(q, dtype=float)
    lb = np.asarray(lb, dtype=float)
    ub = np.asarray(ub, dtype=float)

    # 길이 체크
    if len(joint_names) != len(q) or len(q) != len(lb) or len(lb) != len(ub):
        raise ValueError("joint_names, q, lb, ub 길이가 일치해야 합니다.")

    # 관절 한계 검사
    if (q < lb).any() or (q > ub).any():
        return False

    # 서비스 준비
    _ensure_client(timeout=timeout)

    # 요청 메시지 구성
    js = JointState(name=list(joint_names), position=q.tolist())
    rs = RobotState(joint_state=js)
    req = GetStateValidity.Request(robot_state=rs, group_name=group_name)

    # 추가 제약 조건이 있으면 포함
    if constraints is not None:
        req.constraints = constraints

    # 호출 및 대기
    future = _CLI.call_async(req)
    rclpy.spin_until_future_complete(_NODE, future, timeout_sec=timeout)
    if not future.done() or future.result() is None:
        _NODE.get_logger().warn("GetStateValidity timeout/fail")
        return False

    return bool(future.result().valid)

def shutdown_client():
    global _NODE, _CLI
    if _NODE is not None:
        _NODE.destroy_node()
        _NODE = None
        _CLI = None
    if rclpy.ok():
        rclpy.shutdown()