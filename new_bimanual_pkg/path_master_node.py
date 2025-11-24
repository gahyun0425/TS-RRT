# master

import rclpy
import numpy as np
import pinocchio as pin
import json, time
import math

from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from moveit_msgs.srv import GetPositionIK
from sensor_msgs.msg import JointState
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped, Point

from new_bimanual_pkg.planning.BiRRT import BiRRT
from new_bimanual_pkg.planning.TS_RRT import TangentSpaceRRT
from new_bimanual_pkg.planning.trajectory import TrajectoryPlanner
from new_bimanual_pkg.planning.spline import Spline
from new_bimanual_pkg.constraint.projection import pot_grasp_projector
from new_bimanual_pkg.constraint.object_constraint import ObjectConstraint
from new_bimanual_pkg.planning.path_simplification import simplify_path
from new_bimanual_pkg.constraint.constraint import is_state_valid

from new_bimanual_pkg.utils.joint_limit import load_joint_limits
from new_bimanual_pkg.utils.math import quat_from_rpy
from new_bimanual_pkg.ik.trak_ik import compute_ik_via_moveit
from new_bimanual_pkg.ik.request import _ik_request_async


class PathNode(Node):
    def __init__(self):
        super().__init__('path_node')

        from rclpy.callback_groups import ReentrantCallbackGroup
        self.cbgroup = ReentrantCallbackGroup()


        # --- gripper: MuJoCo direct-publish mode ---
        self.use_mujoco_grip = True  # 컨트롤러 없으니 True
        self.mj_grip_topic   = '/mujoco/gripper_set'  # 너의 브리지 입력 토픽명으로 변경
        self.mj_grip_pub     = self.create_publisher(JointState, self.mj_grip_topic, 10)


        # IK service client
        self.ik_cli = self.create_client(GetPositionIK, '/compute_ik', callback_group=self.cbgroup)
        if not self.ik_cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("'/compute_ik' service not available yet; waiting in background.")


        # pinocchio FK -> 시각화용. IK보다 먼저 준비
        urdf_path = "/home/gaga/bimanual_ws/src/new_bimanual_pkg/mujoco_models/bimanual.urdf"
        full_model = pin.buildModelFromUrdf(urdf_path)

        # EE
        self.ee_frame_l = "gripper_l_rh_p12_rn_base"
        self.ee_frame_r = "gripper_r_rh_p12_rn_base"

        # left, right joint name
        left_names  = ['arm_l_joint1','arm_l_joint2','arm_l_joint3','arm_l_joint4','arm_l_joint5','arm_l_joint6','arm_l_joint7']
        right_names = ['arm_r_joint1','arm_r_joint2','arm_r_joint3','arm_r_joint4','arm_r_joint5','arm_r_joint6','arm_r_joint7']
        self.left_names  = left_names
        self.right_names = right_names
        self.joint_names = left_names + right_names 

        # 두 팔 별 퍼블리셔 (컨트롤러 토픽에 맞게 바꿔도 됨)
        self.traj_pub_left  = self.create_publisher(JointTrajectory, '/birrt/trajectory_left',  10)
        self.traj_pub_right = self.create_publisher(JointTrajectory, '/birrt/trajectory_right', 10)
        
        # 양팔 joint만 남기고 잠그기
        keep = set(self.joint_names)
        lock_ids = []
        for name in full_model.names[1:]:
            if name not in keep:
                jid = full_model.getJointId(name)
                if jid != 0:
                    lock_ids.append(jid)

        q0_full = pin.neutral(full_model)
        self.model = pin.buildReducedModel(full_model, lock_ids, q0_full)
        self.data = self.model.createData()

        # FK에서 사용할 EE
        self.ee_frame_id_l = self.model.getFrameId(self.ee_frame_l)
        self.ee_frame_id_r = self.model.getFrameId(self.ee_frame_r)

        # 기본은 왼팔로 시각화
        self.ee_frame_id = self.ee_frame_id_l

        # 시작자세 (RRT 시작점 및 FK용)
        self.start_q = np.zeros(len(self.joint_names), dtype=float)

        # === 좌/우 목표 pose 고정 값으로 사용 ===
        # 왼쪽 좌표값: 0.5  0.23  1.12
        target_l = np.array([0.5, 0.23, 1.12], dtype=float)

        # 왼쪽 방위값 (deg): 90 0 0
        rL, pL, yL = 90.0, 0.0, 0.0
        rL = math.radians(rL)
        pL = math.radians(pL)
        yL = math.radians(yL)
        quat_l = quat_from_rpy(rL, pL, yL)

        # 오른쪽 좌표값: 0.5  -0.23  1.12
        target_r = np.array([0.5, -0.23, 1.12], dtype=float)

        # 오른쪽 방위값 (deg): -90 0 0
        rR, pR, yR = -90.0, 0.0, 0.0
        rR = math.radians(rR)
        pR = math.radians(pR)
        yR = math.radians(yR)
        quat_r = quat_from_rpy(rR, pR, yR)

        seed_left  = [0.0] * len(left_names)
        seed_right = [0.0] * len(right_names)

        base_frame = 'world' 

        ik_l = compute_ik_via_moveit(
            self,
            group='left_arm',
            frame_id=base_frame,
            ik_link_name=self.ee_frame_l,    
            pos=target_l.tolist(),
            quat=quat_l,
            seed_names=left_names,
            seed_values=seed_left,
            timeout=0.5,
            attempts=10,
            avoid_collisions=True
        )
        if ik_l is None:
            raise RuntimeError("Left IK failed (TRAC-IK)")

        ik_r = compute_ik_via_moveit(
            self,
            group='right_arm',
            frame_id=base_frame,
            ik_link_name=self.ee_frame_r,    
            pos=target_r.tolist(),
            quat=quat_r,
            seed_names=right_names,
            seed_values=seed_right,
            timeout=0.5,
            attempts=10,
            avoid_collisions=True
        )
        if ik_r is None:
            raise RuntimeError("Right IK failed (TRAC-IK)")
        
        # moveit group & joint limit
        self.group_name = 'both_arms'
        
        joint_limits_yaml = '/home/gaga/bimanual_ws/src/bimanual_moveit_config/config/joint_limits.yaml'
        self.lb, self.ub = load_joint_limits(urdf_path, joint_limits_yaml, self.joint_names)

        # collision planner
        self.planner = BiRRT(
            joint_names=self.joint_names,
            lb=self.lb,
            ub=self.ub,
            group_name=self.group_name,
            state_dim=len(self.joint_names),
            max_iter=2000,
            step_size=0.03,
            edge_check_res=0.05,
        )

        # IK 결과를 goal로 결합
        self.goal_q = np.array([*(ik_l[nm] for nm in left_names),
                                *(ik_r[nm] for nm in right_names)], dtype=float)

        self.planner.set_start(self.start_q)
        self.planner.set_goal(self.goal_q)

        # publisher & marker qos
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.marker_pub = self.create_publisher(Marker, '/visualization_marker', qos)
        self.path_pub_full = self.create_publisher(Path, '/birrt_path/full', qos)
        self.path_pub_a = self.create_publisher(Path, '/birrt_path/A', qos)
        self.path_pub_b = self.create_publisher(Path, '/birrt_path/B', qos)

        self.traj_topic = '/birrt/trajectory'
        self.traj_pub = self.create_publisher(JointTrajectory, self.traj_topic, 10)

        # 주기적으로 한 번만 계획/시각화
        self.timer = self.create_timer(1.0, self.plan_once_and_visualize)
        self.done = False

        # --- 도달 판정 상태 ---
        self._awaiting_arrival = False     # 트젝 보낸 뒤 true로, 도달 후 false
        self._arm_goal = {}                # {joint_name: 목표각}
        self._arrive_count = 0             # 연속 만족 카운트
        self._arrive_needed = 10           # 몇 번 연속 만족하면 도달로 볼지 (100Hz면 0.1s)

        # 도달 타임아웃(강제 오픈) 설정
        self._arrival_timeout_timer = None
        self._arrival_timeout_s = 50.0  # ← 5초 타임아웃

        # 토픽에서 최신 JointState 수신
        self.js_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_state, 50
        )

        # 그리퍼 타깃 퍼블리셔(/gripper_target로 보냄)
        self.grip_target_pub = self.create_publisher(JointState, '/gripper_target', 10)
        self.desired_pub = self.create_publisher(JointState, '/desired_joint_angles', 10)

        # 램프 상태 멤버
        self._grip_ramp_timer = None
        self._grip_ramp_t0 = None
        self._grip_ramp_T = 1.0      # 총 램프 시간(초) – 취향대로
        self._grip_ramp_rate = 50.0  # 발행 주기(Hz)
        self._grip_start = {}
        self._grip_goal  = {}
        self._grip_mode = None

        # ==== after existing __init__ content ====
        # 파지 완료 후 회전/플레이스 트리거 상태
        self._grasp_done = False
        self._did_rotate_place = False
        # 파지 완료되면 이 타이머 루프가 회전→플레이스 진행
        self._after_grasp_timer = self.create_timer(0.1, self._after_grasp_loop, callback_group=self.cbgroup)


        self.use_absolute_after_grasp_place = True  # True면 절대 포즈로 바로 이동, False면 기존 rotate→place

        # __init__ 안쪽 (after_grasp_target 쓰던 자리 대체)
        self.after_grasp_targets = [
            {   # 1) Lift
                "frame_id": "world",
                "pos": [0.5, 0.0, 1.12],        # pot 중심 (x,y,z)
                "rpy_deg": [0.0, 0.0, 0.0],    # 회전 없음
                "eps_xy_mm": 5.0,
                "eps_z_mm": 5.0,
                "level_deg": 1.0,
            },
            {   # 2) Rotate
                "frame_id": "world",
                "pos": [0.5, 0.15, 1.12],
                "rpy_deg": [0.0, -20.0, 0.0],  # y축 -10°
                "eps_xy_mm": 5.0,
                "eps_z_mm": 5.0,
                "level_deg": 1.0,
            },
            {   # 3) Place
                "frame_id": "world",
                "pos": [0.6, 0.15, 1.15],
                "rpy_deg": [0.0, -85.0, 0.0],
                "eps_xy_mm": 5.0,
                "eps_z_mm": 5.0,
                "level_deg": 1.0,
            },
        ]



        # 상태 브로드캐스트 (문자열)
        self.cbirrt_status_pub = self.create_publisher(String, '/constraint_birrt/status', 10)

        # __init__ 끝부분 즈음에 추가
        from enum import Enum
        class Stage(Enum):
            IDLE=0; LIFT=1; ROTATE=2; PLACE=3; DONE=4

        self.stage = Stage.IDLE
        self._sequence_running = False
        self._lift_dz = 0.10         # 2단계: 위로 10cm
        self._rotate_deg = -90.0      # 3단계: -90deg (world Y-축 기준)
        self._place_target_world = pin.SE3(  # 4단계: 테이블 위 목표 pot pose
            np.eye(3),
            np.array([0.60, 0.00, 0.82], dtype=float)  # (예시) 테이블 상면+여유
        )
        self._place_target_world.rotation[:] = np.eye(3)  # 수평 유지 (필요시 수정)

        self._arrival_mode = 'close'

        # --- gripper keepalive ---
        self._grip_keepalive_hz = 300.0   # 필요시 5~20Hz 정도로
        self._grip_keepalive_timer = self.create_timer(
            1.0/self._grip_keepalive_hz, self._grip_keepalive_tick,
            callback_group=self.cbgroup
        )

        # __init__ 안
        self.gripper_names = [
            'gripper_r_joint1','gripper_r_joint2','gripper_r_joint3','gripper_r_joint4',
            'gripper_l_joint1','gripper_l_joint2','gripper_l_joint3','gripper_l_joint4'
        ]

        # 닫힘(타겟) 값과 허용 오차 (모델에 맞게 조정)
        self.grip_close_targets = {
            'gripper_r_joint1': 1.1, 'gripper_r_joint2': 1.0,
            'gripper_r_joint3': 1.1, 'gripper_r_joint4': 1.0,
            'gripper_l_joint1': 1.1, 'gripper_l_joint2': 1.0,
            'gripper_l_joint3': 1.1, 'gripper_l_joint4': 1.0,
        }

        self.grip_pos_tol   = 0.02   # 위치 오차
        self.grip_vel_tol   = 0.02   # 속도 오차
        self.grip_min_hold  = 0.15   # 최소 유지 시간(s) – 접촉/슬립 안정화
        self.grip_max_hold  = 3.0    # 안전 상한(s) – 과도한 퍼블리시 방지

        # 최신 조인트 상태 맵
        self._last_js_map = {}  # name -> (pos, vel)

        self._last_js_wall = 0.0

    def _check_valid(self, q, name: str, constraints=None) -> bool:
        try:
            ok = is_state_valid(
                q=np.asarray(q, float),
                joint_names=self.joint_names,
                lb=self.lb, ub=self.ub,
                group_name=self.group_name,
                timeout=2.0,
                constraints=constraints
            )
        except Exception as e:
            self.get_logger().warn(f"[validity] {name}: exception {e}")
            return False

        if ok:
            pass
        else:
            self.get_logger().warn(f"[validity] {name}: ❌ invalid (limit/collision/constraints)")
        return ok

    # FK (joint space -> 3D point for current ee_frame_id)
    def q_to_point(self, q: np.ndarray) -> Point:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.ee_frame_id]
        p = Point()
        p.x, p.y, p.z = pose.translation.tolist()
        return p

    def _base_marker(self, mid: int, mtype: int, ns: str) -> Marker:
        m = Marker()
        m.header.frame_id = 'world'
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns = ns
        m.id = mid
        m.type = mtype
        m.action = Marker.ADD
        m.lifetime.sec = 0
        m.pose.orientation.w = 1.0
        return m

    def q_to_posestamped(self, q: np.ndarray) -> PoseStamped:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        T = self.data.oMf[self.ee_frame_id]
        R = T.rotation
        t = T.translation
        quat = pin.Quaternion(R).coeffs()  # [x,y,z,w]

        ps = PoseStamped()
        ps.header.frame_id = 'world'
        ps.header.stamp = self.get_clock().now().to_msg()
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = t.tolist()
        ps.pose.orientation.x = float(quat[0])
        ps.pose.orientation.y = float(quat[1])
        ps.pose.orientation.z = float(quat[2])
        ps.pose.orientation.w = float(quat[3])
        return ps

    def make_polyline_marker(self, points, mid: int, ns:str, color:str, width:float):
        m = self._base_marker(mid, Marker.LINE_STRIP, ns)
        m.scale.x = width
        m.color.a = 1.0
        if color == 'r': m.color.r = 1.0
        elif color == 'g': m.color.g = 1.0
        elif color == 'b': m.color.b = 1.0
        else: m.color.r = m.color.g = 0.8
        m.points = points
        return m

    def make_tree_nodes_marker(self, points, mid: int, ns: str, color='g'):
        m = self._base_marker(mid, Marker.POINTS, ns)
        m.scale.x = 0.01; m.scale.y = 0.01
        m.color.a = 1.0
        if color == 'g':   m.color.g = 0.9
        elif color == 'b': m.color.b = 0.9
        else:              m.color.r = 0.9
        m.points = points
        return m

    def make_tree_edges_marker(self, edges, mid: int, ns: str, color='g'):
        m = self._base_marker(mid, Marker.LINE_LIST, ns)
        m.scale.x = 0.003
        m.color.a = 1.0
        if color == 'g':   m.color.g = 0.7
        elif color == 'b': m.color.b = 0.7
        else:              m.color.r = 0.7
        m.points = [pt for seg in edges for pt in seg]  # 두 점씩 한 선
        return m

    def make_sphere_marker(self, point, mid: int, ns: str, color='y'):
        m = self._base_marker(mid, Marker.SPHERE, ns)
        m.scale.x = m.scale.y = m.scale.z = 0.05
        m.color.a = 1.0
        if color == 'y': m.color.r, m.color.g = 0.9, 0.9
        elif color == 'g': m.color.g = 1.0
        elif color == 'b': m.color.b = 1.0
        else: m.color.r = 1.0
        m.pose.position = point
        return m

    def tree_to_points_edges(self, tree):
        pts, edges = [], []
        for n in tree:
            pts.append(self.q_to_point(n['q']))
        for idx, n in enumerate(tree):
            p = n['parent']
            if p is None:
                continue
            pa = self.q_to_point(tree[p]['q'])
            ch = self.q_to_point(n['q'])
            edges.append((pa, ch))
        return pts, edges

    def plan_once_and_visualize(self):
        if self.done:
            return
        self.done = True

        self.get_logger().info('Planning (Bi-RRT)...')
        result = self.planner.solve(max_time=5.0)

        ok = False; full = None
        path_a = getattr(self.planner, 'last_path_a', None)
        path_b = getattr(self.planner, 'last_path_b', None)
        if isinstance(result, tuple):
            if len(result) == 4:
                ok, full, path_a, path_b = result
            elif len(result) == 2:
                ok, full = result

        if not ok or full is None:
            self.get_logger().warning('Failed to find path.')
            return
        
        # --- Path simplification (조용한 유효성 검사 사용) ---
        full = simplify_path(
            full,
            node=self,                # ★ OMPL PathSimplifier 경로로 들어감 (MoveIt 유효성 사용)
            constraints=None,
            max_step=0.02,            # 유효성 샘플 분해능 느낌과 매칭
            red_passes=1,             # 폴백 경로일 때만 의미, OMPL에선 내부 reduce/smooth 사용
            shortcut_attempts=200,    # 폴백 경로일 때만 의미
            smooth_iters=0            # 폴백 경로일 때만 의미 (OMPL은 자체 B-spline smooth 사용)
        )
        
        # full shape: (N, 14)  ← N: 웨이포인트 수
        dof_each = 7
        full = np.asarray(full)
        Q_L = full[:, :dof_each]               # (N,7) 왼팔
        Q_R = full[:, dof_each:2*dof_each]     # (N,7) 오른팔

        # 팔 별 속도/가속 상한 (필요시 파라미터화 가능)
        qd_max_L  = np.array([1.5]*dof_each)
        qdd_max_L = np.array([3.0]*dof_each)
        qd_max_R  = np.array([1.5]*dof_each)
        qdd_max_R = np.array([3.0]*dof_each)

        # TOPP→스플라인→균일샘플 (왼팔)
        tp_L = TrajectoryPlanner(
            ds=0.005,
            sdot_start=0.0,
            stop_window_s=0.05,
            alpha_floor=1.5,
            v_min_time=1e-4,
            sample_hz=10.0,
            max_points=100000,
        )

        outL = tp_L.plan(Q_L, qd_max_L, qdd_max_L)


        # TOPP→스플라인→균일샘플 (오른팔)
        tp_R = TrajectoryPlanner(
            ds=0.005,
            sdot_start=0.0,
            stop_window_s=0.05,
            alpha_floor=1.5,
            v_min_time=1e-4,
            sample_hz=10.0,
            max_points=100000,
        )

        outR = tp_R.plan(Q_R, qd_max_R, qdd_max_R)


        # JointTrajectory 메시지로 변환 & 퍼블리시 (두 팔 따로)
        trajL = TrajectoryPlanner.make_joint_trajectory_msg(
            joint_names=self.left_names,
            t0=float(outL["t_samples"][0]),
            t_samples=outL["t_samples"],
            Q_samples=outL["Q_samples"],
            Qd_samples=outL["Qd_samples"],
        )
        trajR = TrajectoryPlanner.make_joint_trajectory_msg(
            joint_names=self.right_names,
            t0=float(outR["t_samples"][0]),
            t_samples=outR["t_samples"],
            Q_samples=outR["Q_samples"],
            Qd_samples=outR["Qd_samples"],
        )

        # --- 도달 판정을 위한 목표각 기록 & 감시 시작 ---
        goal_map = {}
        if trajL and len(trajL.points) > 0:
            lastL = trajL.points[-1].positions
            for nm, q in zip(self.left_names, lastL):
                goal_map[nm] = float(q)

        if trajR and len(trajR.points) > 0:
            lastR = trajR.points[-1].positions
            for nm, q in zip(self.right_names, lastR):
                goal_map[nm] = float(q)

        self._arm_arrival(goal_map, mode='close')

        self.get_logger().info(f'Arrival monitoring ON for {len(goal_map)} joints.')


        now = self.get_clock().now().to_msg()
        if trajL is not None:
            trajL.header.frame_id = 'world'
            trajL.header.stamp = now
            self.traj_pub_left.publish(trajL)
        if trajR is not None:
            trajR.header.frame_id = 'world'
            trajR.header.stamp = now
            self.traj_pub_right.publish(trajR)

        self.get_logger().info(
            f"Published: left({len(trajL.points) if trajL else 0}) right({len(trajR.points) if trajR else 0})"
        )

        controller_hz = 100.0  # ← 실제 joint_trajectory_controller 주기
        dt = 1.0/controller_hz

        splineL = Spline(outL["t_knots"], Q_L, outL["qdot_knots"])
        splineR = Spline(outR["t_knots"], Q_R, outR["qdot_knots"])

        t0L, t1L = float(outL["t_samples"][0]), float(outL["t_samples"][-1])
        t0R, t1R = float(outR["t_samples"][0]), float(outR["t_samples"][-1])
        t0 = min(t0L, t0R); t1 = max(t1L, t1R)
        time_scale = 1.0  # 필요시 0.8 등으로 조절
        T = (t1 - t0) * time_scale

        t_common = t0 + np.arange(0.0, T + 0.5*dt, dt)

        QL_sync, QLd_sync, QR_sync, QRd_sync = [], [], [], []
        for t in t_common:
            qL, qLd = Spline.clamp_eval(splineL, t)
            qR, qRd = Spline.clamp_eval(splineR, t)
            QL_sync.append(qL);  QLd_sync.append(qLd)
            QR_sync.append(qR);  QRd_sync.append(qRd)

        QL_sync  = np.vstack(QL_sync)
        QLd_sync = np.vstack(QLd_sync)
        QR_sync  = np.vstack(QR_sync)
        QRd_sync = np.vstack(QRd_sync)

        Q_sync  = np.hstack([QL_sync,  QR_sync])   # (M,14)
        Qd_sync = np.hstack([QLd_sync, QRd_sync])  # (M,14)

        traj_all = TrajectoryPlanner.make_joint_trajectory_msg(
            joint_names=(self.left_names + self.right_names),
            t0=t0,
            t_samples=(t_common - t0),   # time_from_start 기준이면 0부터
            Q_samples=Q_sync,
            Qd_samples=Qd_sync
        )
        if traj_all is not None:
            traj_all.header.frame_id = 'world'
            traj_all.header.stamp = self.get_clock().now().to_msg()
            self.traj_pub.publish(traj_all)

        # FULL PATH
        full_points = [self.q_to_point(q) for q in full]
        if len(full_points) >= 2:
            self.marker_pub.publish(self.make_polyline_marker(full_points, 0, 'birrt_path_full', 'r', 0.01))
            path_msg = Path()
            path_msg.header.frame_id = 'world'
            path_msg.header.stamp = self.get_clock().now().to_msg()
            path_msg.poses = [self.q_to_posestamped(q) for q in full]
            self.path_pub_full.publish(path_msg)

        # PARTIAL PATHS A/B
        def publish_partial(name, arr, mid, color, pub):
            if arr is None or len(arr) < 2:
                return
            pts = [self.q_to_point(q) for q in arr]
            self.marker_pub.publish(self.make_polyline_marker(pts, mid, name, color, 0.02))
            msg = Path()
            msg.header.frame_id = 'world'
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.poses = [self.q_to_posestamped(q) for q in arr]
            pub.publish(msg)

        publish_partial('birrt_path_A', path_a, 10, 'g', self.path_pub_a)
        publish_partial('birrt_path_B', path_b, 11, 'b', self.path_pub_b)

        # 트리 마커
        try:
            start_pts, start_edges = self.tree_to_points_edges(self.planner.start_tree)
            goal_pts,  goal_edges  = self.tree_to_points_edges(self.planner.goal_tree)

            self.marker_pub.publish(self.make_tree_nodes_marker(start_pts, 1, 'start_tree_nodes', 'g'))
            self.marker_pub.publish(self.make_tree_edges_marker(start_edges, 2, 'start_tree_edges', 'g'))
            self.marker_pub.publish(self.make_tree_nodes_marker(goal_pts,  3, 'goal_tree_nodes',  'b'))
            self.marker_pub.publish(self.make_tree_edges_marker(goal_edges, 4, 'goal_tree_edges',  'b'))
        except Exception as e:
            self.get_logger().warn(f"Tree visualization skipped: {e}")

        # 시작/목표 위치
        self.marker_pub.publish(self.make_sphere_marker(self.q_to_point(self.start_q), 5, 'start_pose', 'g'))
        self.marker_pub.publish(self.make_sphere_marker(self.q_to_point(self.goal_q),  6, 'goal_pose',  'b'))

        len_a = len(path_a) if path_a is not None else 0
        len_b = len(path_b) if path_b is not None else 0
        self.get_logger().info(f'Path found! full={len(full)} A={len_a} B={len_b} (markers published)')

    def publish_joint_trajectory(self, q_path: np.ndarray, dt: float = 0.1):
        if q_path is None or len(q_path) == 0:
            self.get_logger().warn("No path to publish.")
            return
        dof = len(self.joint_names)
        if q_path.shape[1] != dof:
            self.get_logger().error(f"Path DOF({q_path.shape[1]}) != joint_names({dof})")
            return

        traj = JointTrajectory()
        traj.header.frame_id = 'world'
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = list(self.joint_names)

        t = 0.0
        for q in q_path:
            pt = JointTrajectoryPoint()
            pt.positions = [float(x) for x in q]
            t += dt
            pt.time_from_start.sec = int(t)
            pt.time_from_start.nanosec = int((t - int(t)) * 1e9)
            traj.points.append(pt)

        self.traj_pub.publish(traj)
        self.get_logger().info(f"Published JointTrajectory to {self.traj_topic} ({len(traj.points)} points)")

    def _is_gripper_closed(self) -> bool:
        # 모든 그리퍼 조인트가 타겟 근처 && 거의 정지인지 확인
        ok_all = True
        for n in self.gripper_names:
            if n not in self._last_js_map: 
                ok_all = False
                break
            pos, vel = self._last_js_map[n]
            tgt = self.grip_close_targets.get(n, 0.0)
            if abs(pos - tgt) > self.grip_pos_tol:
                ok_all = False
                break
            if abs(vel) > self.grip_vel_tol:
                ok_all = False
                break
        return ok_all


    def close_grippers(self, force: bool=False):
        # OPEN 중이면, 강제 지시가 아니면 닫기 억제
        if getattr(self, '_grip_mode', None) == 'open' and not force:
            self.get_logger().info("Close suppressed (OPEN in progress).")
            return

        self._grip_mode = 'close'
        # 어떤 모니터/타이머가 있어도 닫기 전에 싹 정지
        self._awaiting_arrival = False
        self._arm_goal = {}
        self._arrive_count = 0
        self._cancel_arrival_timeout()
        self._stop_all_grip_timers()

        self._set_grip_mode('close')

        names = [
            'gripper_r_joint1','gripper_r_joint2','gripper_r_joint3','gripper_r_joint4',
            'gripper_l_joint1','gripper_l_joint2','gripper_l_joint3','gripper_l_joint4'
        ]
        targets = [1.1,1.0,1.1,1.0, 1.1,1.0,1.1,1.0]

        msg = JointState()
        msg.name = names
        msg.position = targets

        if not hasattr(self, "_grip_hold_timer"):
            self._grip_hold_timer = None
        self._grip_hold_count = 0

        if self._grip_hold_timer:
            self._grip_hold_timer.cancel()
            self._grip_hold_timer = None

        self._grip_close_t0 = self.get_clock().now().nanoseconds * 1e-9

        def _stop_publish():
            if self._grip_hold_timer:
                self._grip_hold_timer.cancel()
                self._grip_hold_timer = None
            self._grip_mode = None
            self.get_logger().info("Gripper close: publish stopped.")
            self._set_grip_mode(None)

        def _tick():
            t_now = self.get_clock().now().nanoseconds * 1e-9
            elapsed = t_now - self._grip_close_t0

            msg.header.stamp = self.get_clock().now().to_msg()
            # 필요한 곳에만 퍼블리시 (중복 퍼블 제거 권장)
            self.desired_pub.publish(msg)
            self.grip_target_pub.publish(msg)
            if getattr(self, 'use_mujoco_grip', False) and hasattr(self, 'mj_grip_pub'):
                self.mj_grip_pub.publish(msg)

            self._grip_hold_count += 1

            # 조건 1) 최소 유지시간 경과 AND 닫힘 확인 → 중단
            if elapsed >= self.grip_min_hold and self._is_gripper_closed():
                self.get_logger().info(f"Gripper closed confirmed (elapsed={elapsed:.2f}s).")
                _stop_publish()
                self._grasp_done = True
                return

            # 조건 2) 안전 상한 시간 경과 → 강제 중단
            if elapsed >= self.grip_max_hold:
                self.get_logger().warn(f"Gripper close timeout {elapsed:.2f}s → stop publishing.")
                _stop_publish()
                self._grasp_done = True
                return

        self._grip_hold_timer = self.create_timer(1.0/100.0, _tick)

    def _set_grip_mode(self, m):
        if getattr(self, '_grip_mode', None) != m:
            self.get_logger().info(f"[GRIP] mode -> {m}")
        self._grip_mode = m


    def open_grippers(self, force: bool=False):
        # 열기로 선점: arrival/타이머/모드 모두 리셋
        self._awaiting_arrival = False
        self._arm_goal = {}
        self._arrive_count = 0
        self._cancel_arrival_timeout()
        self._stop_all_grip_timers()

        self._set_grip_mode('open') 

        # 이미 OPEN 모드면 강제 아니면 재설정만
        if getattr(self, '_grip_mode', None) != 'open' or force:
            self.get_logger().info("[GRIP] mode -> open")
        self._grip_mode = 'open'

        names = [
            'gripper_r_joint1','gripper_r_joint2','gripper_r_joint3','gripper_r_joint4',
            'gripper_l_joint1','gripper_l_joint2','gripper_l_joint3','gripper_l_joint4'
        ]
        targets = [0.0]*8  # 너 모델에서 0.0이 ‘열림’

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = targets

        self.desired_pub.publish(msg)
        self.grip_target_pub.publish(msg)
        if getattr(self, 'use_mujoco_grip', False) and hasattr(self, 'mj_grip_pub'):
            self.mj_grip_pub.publish(msg)


            
    def _on_joint_state(self, msg: JointState):
        self._last_js_wall = time.time()
        vel_dict = {}
        if len(msg.velocity) == len(msg.name):
            vel_dict = {n: msg.velocity[i] for i, n in enumerate(msg.name)}
        for i, n in enumerate(msg.name):
            self._last_js_map[n] = (msg.position[i], vel_dict.get(n, 0.0))

        # 도달 감시 중이 아니면 여기서 종료
        if not self._awaiting_arrival or not self._arm_goal:
            return

        # ↓ 이하 기존 도달판정 로직 그대로 유지
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        missing = [n for n in self._arm_goal.keys() if n not in name_to_idx]
        if missing:
            now_s = self.get_clock().now().nanoseconds*1e-9
            if (getattr(self, "_missing_log_t", 0.0) + 0.5) < now_s:
                self._missing_log_t = now_s
                self.get_logger().warn(f"[ARRIVAL] missing joints in /joint_states: {missing[:4]}{' ...' if len(missing)>4 else ''}")
            return
        
        self._last_joint_state = [msg.position[msg.name.index(n)] for n in self.joint_names if n in msg.name]

        # tol 설정(라디안 / 라디안/초)
        pos_tol = 0.02    # 위치 오차 범위 0.57° 
        vel_tol = 0.02    # 
        # JointState.velocity 길이가 0이거나 일부만 있으면 속도 판정 생략 가능
        have_vel = (len(msg.velocity) == len(msg.name) and len(msg.velocity) > 0)

        max_pos_err = 0.0
        max_abs_vel = 0.0

        for jn, q_goal in self._arm_goal.items():
            idx = name_to_idx[jn]
            q_cur = msg.position[idx]
            err = abs(q_cur - q_goal)
            if err > max_pos_err:
                max_pos_err = err
            if have_vel:
                v = abs(msg.velocity[idx])
                if v > max_abs_vel:
                    max_abs_vel = v

        # 판정: 위치가 충분히 가깝고(필수), 속도도 충분히 작으면(있으면) OK
        pos_ok = (max_pos_err <= pos_tol)
        vel_ok = (not have_vel) or (max_abs_vel <= vel_tol)

        # 0.5초에 한 번 디버그 출력
        now_s = self.get_clock().now().nanoseconds*1e-9
        if (getattr(self, "_arr_dbg_t", 0.0) + 0.5) < now_s:
            self._arr_dbg_t = now_s
            self.get_logger().info(
                f"[ARRIVAL] mode={self._arrival_mode} "
                f"count={self._arrive_count}/{self._arrive_needed} "
                f"pos_err={max_pos_err:.4f} tol={pos_tol:.4f} "
                f"vel={max_abs_vel:.4f} tol={vel_tol:.4f} "
                f"(pos_ok={pos_ok}, vel_ok={vel_ok}) "
                f"goals={len(self._arm_goal)}"
            )


        if pos_ok and vel_ok:
            self._arrive_count += 1
        else:
            self._arrive_count = 0

        # 연속 만족 시 그리퍼 동작 후 감시 종료
        if self._arrive_count >= self._arrive_needed:
            self._awaiting_arrival = False
            self._cancel_arrival_timeout()
            self.get_logger().info(
                f'Arrival confirmed: max_pos_err={max_pos_err:.4f}, max_abs_vel={max_abs_vel:.4f}'
            )

            if getattr(self, "_arrival_mode", "close") == "open":
                # 동일 방식으로 '열기'
                self.open_grippers()
                self.get_logger().info("Arrival action: OPEN grippers (hold publish).")
            else:
                # 동일 방식으로 '닫기'
                self.close_grippers()
                self.get_logger().info("Arrival action: CLOSE grippers (hold publish).")


    def _ease_smoothstep(self, s: float) -> float:
        # 0~1 -> 0~1, 가감속이 부드러운 프로파일
        s = max(0.0, min(1.0, s))
        return s*s*(3.0 - 2.0*s)

    def _grip_ramp_tick(self):
        # 진행률
        now = self.get_clock().now()
        s = (now - self._grip_ramp_t0).nanoseconds * 1e-9 / max(1e-6, self._grip_ramp_T)
        s = max(0.0, min(1.0, s))
        s_ease = self._ease_smoothstep(s)

        # 현재 보간 값 만들기 (원 코드)
        names = list(self._grip_goal.keys())
        pos = []
        for n in names:
            q0 = self._grip_start.get(n, 0.0)
            q1 = self._grip_goal[n]
            pos.append(q0 + (q1 - q0) * s_ease)

        # 퍼블리시 (원 코드)
        msg = JointState()
        msg.header.stamp = now.to_msg()
        msg.name = names
        msg.position = [float(v) for v in pos]
        self.grip_target_pub.publish(msg)

        # 종료 처리 (원 코드)
        if s >= 1.0 and self._grip_ramp_timer is not None:
            self._grip_ramp_timer.cancel()
            self._grip_ramp_timer = None
            self.get_logger().info("Gripper ramp done.")
            self._grasp_done = True  

    def _after_grasp_loop(self):
        if not self._grasp_done or self._did_rotate_place:
            return
        self._did_rotate_place = True
        try:
            self._after_grasp_timer.cancel()
        except Exception:
            pass

        # 여기! 시퀀스 호출
        self.get_logger().info(">>> Grasp done. Running 3-step constrained sequence (lift→rotate→place)...")
        self._cbirrt_sequence_and_publish()

    
    def _make_grasp_projector_once(self):
        # 시작 관절(q_start)
        if hasattr(self, "_last_joint_state"):
            q_start = np.array(self._last_joint_state, float)
        else:
            q_start = np.asarray(self.goal_q, float)

        # FK
        pin.forwardKinematics(self.model, self.data, q_start)
        pin.updateFramePlacements(self.model, self.data)
        fidL = self.model.getFrameId(self.ee_frame_l)
        fidR = self.model.getFrameId(self.ee_frame_r)
        TWL = self.data.oMf[fidL]   # world->left_ee
        TWR = self.data.oMf[fidR]   # world->right_ee

        # 파지 직후 pot 초기 world pose (너가 쓰던 값 그대로)
        T_pot0 = pin.SE3(np.eye(3), np.array([0.5, 0.0, 1.0051]))

        # '측정된' EE의 pot-기준 상대자세 (파지 유지 핵심)
        T_l_in_pot_meas = T_pot0.inverse() * TWL
        T_r_in_pot_meas = T_pot0.inverse() * TWR

        self._T_l_in_pot_meas = T_l_in_pot_meas
        self._T_r_in_pot_meas = T_r_in_pot_meas
        self._T_pot0 = T_pot0

        # projector 생성 (재사용)
        self._grasp_proj = pot_grasp_projector(
            self.model, self.data,
            ee_frame_l=self.ee_frame_l,
            ee_frame_r=self.ee_frame_r,
            T_pot0=T_pot0,
            T_l_in_pot=T_l_in_pot_meas,
            T_r_in_pot=T_r_in_pot_meas,
            tol=1.0, max_iters=2000, damping=1e-3
        )
        self.get_logger().info("[CBiRRT] Grasp projector prepared.")


    def _place_absolute_and_publish(self):
        # 0) 준비: 현재(q_start)에서 EE-EE 상대변환 저장
        if hasattr(self, "_last_joint_state"):
            q_start = np.array(self._last_joint_state, float)
        else:
            q_start = np.asarray(self.goal_q, float)

        pin.forwardKinematics(self.model, self.data, q_start)
        pin.updateFramePlacements(self.model, self.data)
        fidL = self.model.getFrameId(self.ee_frame_l)
        fidR = self.model.getFrameId(self.ee_frame_r)
        TWL = self.data.oMf[fidL]   # world->left_ee
        TWR = self.data.oMf[fidR]   # world->right_ee
    
        # 1) pot 목표 pose 구성 (after_grasp_target 기준)
        cfg = self.after_grasp_target
        px, py, pz = [float(v) for v in cfg["pos"]]
        r_deg, p_deg, y_deg = cfg["rpy_deg"]
        r, p, y = math.radians(r_deg), math.radians(p_deg), math.radians(y_deg)

        # RPY -> rotation matrix
        qx, qy, qz, qw = quat_from_rpy(r, p, y)
        R_goal = pin.Quaternion(np.array([qx, qy, qz, qw])).toRotationMatrix()
        T_pot_goal = pin.SE3(R_goal, np.array([px, py, pz], dtype=float))

        # 회전 행렬 만들기 (pot을 손이 ±90° 돌려 잡도록)
        R_target_l = pin.exp3(np.array([0, 0, 0])) 
        R_target_r = pin.exp3(np.array([0, 0, 0])) 

        # 목표 hand→pot 변환
        T_l_in_pot = pin.SE3(R_target_l, np.array([0.0, 0.23, 0.115]))
        T_r_in_pot = pin.SE3(R_target_r, np.array([0.0, -0.23, 0.115]))


        R_err_l = TWL.rotation.T @ T_l_in_pot.rotation
        R_err_r = TWR.rotation.T @ T_r_in_pot.rotation

        print("Left rot error (deg):", pin.log3(R_err_l).T * 180/np.pi)
        print("Right rot error (deg):", pin.log3(R_err_r).T * 180/np.pi)


        # --- 왼팔 목표: after_grasp_target 그대로 ---
        # ❗ 파지 직후 pot의 world pose (projector 만들 때 쓰던 값과 동일하게 맞추세요)
        T_pot0 = pin.SE3(np.eye(3), np.array([0.5, 0.0, 1.0051]))

        # 현재(파지 직후)의 world→EE, world→pot 로부터 'EE의 pot 기준 상대자세' 추정
        T_l_in_pot_meas = T_pot0.inverse() * TWL   # left  EE pose in pot frame
        T_r_in_pot_meas = T_pot0.inverse() * TWR   # right EE pose in pot frame

        print("=== Using measured transforms for projector ===")
        print("T_l_in_pot:", T_l_in_pot_meas)
        print("T_r_in_pot:", T_r_in_pot_meas)

        # 새 pot 목표에서의 EE 목표 = pot 목표 * (EE의 pot기준 상대자세)
        TWL_goal = T_pot_goal * T_l_in_pot_meas
        TWR_goal = T_pot_goal * T_r_in_pot_meas


        self._log_goal_pose("left",  TWL_goal)
        self._log_goal_pose("right", TWR_goal)

        # 2) SE3 -> PoseStamped (제약 생성용)
        def se3_to_ps(T: pin.SE3) -> PoseStamped:
            ps = PoseStamped()
            ps.header.frame_id = cfg.get("frame_id", "world")
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = T.translation.tolist()
            q = pin.Quaternion(T.rotation).coeffs()  # [x,y,z,w]
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
            return ps

        ps_left  = se3_to_ps(TWL_goal)
        ps_right = se3_to_ps(TWR_goal)

                # pot 초기 pose (world 좌표)
        T_pot0 = pin.SE3(np.eye(3), np.array([0.5, 0.0, 1.0051]))
        proj_spec = pot_grasp_projector(
            self.model, self.data,
            ee_frame_l="gripper_l_rh_p12_rn_base",
            ee_frame_r="gripper_r_rh_p12_rn_base",
            T_pot0=T_pot0,
            T_l_in_pot=T_l_in_pot_meas,
            T_r_in_pot=T_r_in_pot_meas,
            tol=1.0,
            max_iters=2000,
            damping=1e-3,
        )

        self.get_logger().info("[CBiRRT] grasp projector set (pot handles).")

        obj_cons = ObjectConstraint()
        cons_left = obj_cons.place_constraints(
            ee_link=self.ee_frame_l, place_pose_world=ps_left,
            eps_xy_mm=float(cfg.get("eps_xy_mm", 150.0)),
            eps_z_mm=float(cfg.get("eps_z_mm", 150.0)),
            level_deg=float(cfg.get("level_deg", 10.0)),
        )
        cons_right = obj_cons.place_constraints(
            ee_link=self.ee_frame_r, place_pose_world=ps_right,
            eps_xy_mm=float(cfg.get("eps_xy_mm", 150.0)),
            eps_z_mm=float(cfg.get("eps_z_mm", 150.0)),
            level_deg=float(cfg.get("level_deg", 10.0)),
        )

        from moveit_msgs.msg import Constraints
        combined = Constraints()
        combined.name = "bimanual_place"
        combined.joint_constraints.extend(cons_left.joint_constraints)
        combined.joint_constraints.extend(cons_right.joint_constraints)
        combined.position_constraints.extend(cons_left.position_constraints)
        combined.position_constraints.extend(cons_right.position_constraints)
        combined.orientation_constraints.extend(cons_left.orientation_constraints)
        combined.orientation_constraints.extend(cons_right.orientation_constraints)
        combined.visibility_constraints.extend(cons_left.visibility_constraints)
        combined.visibility_constraints.extend(cons_right.visibility_constraints)

        cplanner = TangentSpaceRRT(
            joint_names=self.joint_names,
            lb=self.lb, ub=self.ub,
            proj_spec=proj_spec,
            group_name=self.group_name,
            state_dim=len(self.joint_names),
            max_iter=4000,
            step_size=0.01,
            edge_check_res=0.01,
            manifold_tol=1e-3,
            tangent_radius=0.2,
        )
        cplanner.set_constraints(combined)
        self.get_logger().info("[CBiRRT] constraints set (merged MoveIt Constraints).")


        # --- IK 시드 준비 (현재 자세 q_start에서 좌/우팔 시드 분리) ---
        nL = len(self.left_names); nR = len(self.right_names)
        if q_start.shape[0] < (nL + nR):
            q_pad = np.zeros(nL + nR, dtype=float); q_pad[:q_start.shape[0]] = q_start
            q_start = q_pad
        seed_left  = [float(q) for q in q_start[:nL]]
        seed_right = [float(q) for q in q_start[nL:nL+nR]]

        # ▶ 콜백에서 쓸 컨텍스트에 'constraints'도 꼭 넣기!
        self._abs_place_ctx = {
            "cplanner": cplanner,
            "constraints": combined,          # ★ 추가
            "q_start": q_start,
            "TWL_goal": TWL_goal,
            "TWR_goal": TWR_goal,
            "seed_left": seed_left,
            "seed_right": seed_right,
            "frame_id": cfg.get("frame_id","world"),
            "cfg": cfg,
        }

        # 왼팔 IK 비동기 시작
        self.get_logger().info("[CBiRRT] calling IK (left) ASYNC...")
        qL = pin.Quaternion(TWL_goal.rotation).coeffs()
        futL = _ik_request_async(
            group="left_arm",
            frame_id=self._abs_place_ctx["frame_id"],
            ik_link_name=self.ee_frame_l,
            pos=TWL_goal.translation.tolist(),
            quat=[float(qL[0]), float(qL[1]), float(qL[2]), float(qL[3])],
            seed_names=self.left_names,
            seed_values=seed_left,
            timeout=0.8,
            avoid_collisions=True,
        )

        # 현재 손의 pot frame에서 상대자세
        print("T_l_in_pot_meas:", T_l_in_pot_meas)
        print("T_r_in_pot_meas:", T_r_in_pot_meas)

        # 내가 정의한 목표랑 비교
        print("Target T_l_in_pot:", T_l_in_pot)
        print("Target T_r_in_pot:", T_r_in_pot)


        futL.add_done_callback(self._on_left_ik_done)
        return

    def _cbirrt_report(self, stage: str, **payload):
        """
        stage: 'start' | 'success' | 'fail'
        payload: 자유롭게 추가 (path_len, ms, reason 등)
        """
        msg = {
            "stage": stage,
            "ts": self.get_clock().now().nanoseconds,  # ns
            **payload
        }
        # 콘솔
        if stage == 'start':
            self.get_logger().info(f"[CBiRRT] start: {payload.get('what','')}")
        elif stage == 'success':
            self.get_logger().info(
                f"[CBiRRT] success: path_len={payload.get('path_len','?')}, "
                f"solve_ms={payload.get('solve_ms','?')}, note={payload.get('what','')}"
            )
        else:
            self.get_logger().warn(
                f"[CBiRRT] FAIL: reason={payload.get('reason','?')}, "
                f"solve_ms={payload.get('solve_ms','?')}, note={payload.get('what','')}"
            )

        # 토픽
        s = String()
        s.data = json.dumps(msg, ensure_ascii=False)
        self.cbirrt_status_pub.publish(s)

    def _on_left_ik_done(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"[CBiRRT] left IK future error: {e}")
            return

        if res is None:
            self.get_logger().error("[CBiRRT] left IK: no response")
            return
        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().warn(f"[CBiRRT] left IK failed (code={res.error_code.val})")
            return

        names = list(res.solution.joint_state.name)
        vals  = list(res.solution.joint_state.position)
        self._abs_place_ctx["ik_l"] = dict(zip(names, vals))

        # 오른팔 IK 비동기 요청
        self.get_logger().info("[CBiRRT] calling IK (right) ASYNC...")
        TWR_goal = self._abs_place_ctx["TWR_goal"]
        qR = pin.Quaternion(TWR_goal.rotation).coeffs()
        futR = _ik_request_async(
            group="right_arm",
            frame_id=self._abs_place_ctx["frame_id"],
            ik_link_name=self.ee_frame_r,
            pos=TWR_goal.translation.tolist(),
            quat=[float(qR[0]), float(qR[1]), float(qR[2]), float(qR[3])],
            seed_names=self.right_names,
            seed_values=self._abs_place_ctx["seed_right"],
            timeout=0.8,
            avoid_collisions=True,
        )
        futR.add_done_callback(self._on_right_ik_done)

    def _on_right_ik_done(self, future):
        try:
            res = future.result()
        except Exception as e:
            self.get_logger().error(f"[CBiRRT] right IK future error: {e}")
            return

        if res is None:
            self.get_logger().error("[CBiRRT] right IK: no response")
            return
        if res.error_code.val != res.error_code.SUCCESS:
            self.get_logger().warn(f"[CBiRRT] right IK failed (code={res.error_code.val})")
            return

        names = list(res.solution.joint_state.name)
        vals  = list(res.solution.joint_state.position)
        ik_r  = dict(zip(names, vals))

        # 컨텍스트 꺼내기
        ctx = getattr(self, "_abs_place_ctx", None)
        if not ctx or "ik_l" not in ctx:
            self.get_logger().error("[CBiRRT] missing context or left IK result")
            return

        ik_l       = ctx["ik_l"]
        q_start    = ctx["q_start"]
        cplanner   = ctx["cplanner"]
        cfg        = ctx["cfg"]
        constraints = ctx.get("constraints", None)   # ★ 병합된 MoveIt Constraints

        # q_goal 구성 (왼/오른팔 순서 주의)
        q_goal = q_start.copy()
        for i, nm in enumerate(self.left_names):
            if nm not in ik_l:
                self.get_logger().warn(f"[CBiRRT] left IK result missing joint {nm}")
                return
            q_goal[i] = float(ik_l[nm])
        for i, nm in enumerate(self.right_names):
            if nm not in ik_r:
                self.get_logger().warn(f"[CBiRRT] right IK result missing joint {nm}")
                return
            q_goal[7+i] = float(ik_r[nm])

        # ---- 계획 전에 유효성 체크 (조인트리밋/충돌/제약) ----
        if not self._check_valid(q_start, "q_start", constraints=None):
            self.get_logger().warn("[CBiRRT] start state invalid (collision/limits). Abort.")
            return

        # goal은 제약 포함해서 검사 (그대로 유지)
        if not self._check_valid(q_goal, "q_goal", constraints=constraints):
            self.get_logger().warn("[CBiRRT] goal state invalid under constraints. Abort.")
            return
        
        # 계획 실행
        cplanner.set_start(q_start)
        cplanner.set_goal(q_goal)
        self._cbirrt_report('start', what='abs_place(bimanual)',
                            target_pos=cfg["pos"], target_rpy_deg=cfg["rpy_deg"])
        t0 = time.time()
        ok, path = cplanner.solve(max_time=12.0)
        dt_ms = int((time.time() - t0)*1000)
        if not ok or path is None:
            self._cbirrt_report('fail', what='abs_place', reason='solve_failed', solve_ms=dt_ms)
            self.get_logger().warn("Absolute place planning failed.")
            return

        path = np.asarray(path)
        self._cbirrt_report('success', what='abs_place', path_len=len(path), solve_ms=dt_ms)

        # === 이하 그대로: 트젝 생성/퍼블리시 ===
        dof_each = 7
        Q_L = path[:, :dof_each]
        Q_R = path[:, dof_each:2*dof_each]

        qd_max_L  = np.array([1.5]*dof_each); qdd_max_L = np.array([3.0]*dof_each)
        qd_max_R  = np.array([1.5]*dof_each); qdd_max_R = np.array([3.0]*dof_each)

        outL = TrajectoryPlanner(
            Q_L, qd_max_L, qdd_max_L,
            ds=0.005, sdot_start=0.0, stop_window_s=0.05, alpha_floor=1.5,
            v_min_time=1e-4, sample_hz=10.0, max_points=100000
        )
        outR = TrajectoryPlanner(
            Q_R, qd_max_R, qdd_max_R,
            ds=0.005, sdot_start=0.0, stop_window_s=0.05, alpha_floor=1.5,
            v_min_time=1e-4, sample_hz=10.0, max_points=100000
        )

        trajL = TrajectoryPlanner.make_joint_trajectory_msg(
            joint_names=self.left_names,
            t0=0.0,
            t_samples=(outL["t_samples"] - outL["t_samples"][0]),
            Q_samples=outL["Q_samples"],
            Qd_samples=outL["Qd_samples"],
        )
        trajR = TrajectoryPlanner.make_joint_trajectory_msg(
            joint_names=self.right_names,
            t0=0.0,
            t_samples=(outR["t_samples"] - outR["t_samples"][0]),
            Q_samples=outR["Q_samples"],
            Qd_samples=outR["Qd_samples"],
        )

        goal_map = {}
        if trajL and len(trajL.points) > 0:
            for nm, q in zip(self.left_names, trajL.points[-1].positions):
                goal_map[nm] = float(q)
        if trajR and len(trajR.points) > 0:
            for nm, q in zip(self.right_names, trajR.points[-1].positions):
                goal_map[nm] = float(q)

        self._arm_arrival(goal_map, mode='open')
        self.get_logger().info('Arrival action set: OPEN when goal is reached.')

        now = self.get_clock().now().to_msg()
        if trajL is not None:
            trajL.header.frame_id = 'world'; trajL.header.stamp = now
            self.traj_pub_left.publish(trajL)
        if trajR is not None:
            trajR.header.frame_id = 'world'; trajR.header.stamp = now
            self.traj_pub_right.publish(trajR)

        self.get_logger().info(
            f"[OPEN-PHASE] Published: left({len(trajL.points) if trajL else 0}) right({len(trajR.points) if trajR else 0})"
        )

        # 동기화 트젝(100Hz)
        controller_hz = 100.0
        dt = 1.0/controller_hz
        splineL = Spline(outL["t_knots"], Q_L, outL["qdot_knots"])
        splineR = Spline(outR["t_knots"], Q_R, outR["qdot_knots"])
        T_L = float(outL["t_samples"][-1] - outL["t_samples"][0])
        T_R = float(outR["t_samples"][-1] - outR["t_samples"][0])
        T = max(T_L, T_R)
        t_common = np.arange(0.0, T + 0.5*dt, dt)

        QL_sync, QLd_sync, QR_sync, QRd_sync = [], [], [], []
        for t in t_common:
            qL, qLd = Spline.clamp_eval(splineL, outL["t_samples"][0] + t)
            qR, qRd = Spline.clamp_eval(splineR, outR["t_samples"][0] + t)
            QL_sync.append(qL); QLd_sync.append(qLd)
            QR_sync.append(qR); QRd_sync.append(qRd)
        QL_sync  = np.vstack(QL_sync);  QLd_sync = np.vstack(QLd_sync)
        QR_sync  = np.vstack(QR_sync);  QRd_sync = np.vstack(QRd_sync)
        Q_sync  = np.hstack([QL_sync, QR_sync])
        Qd_sync = np.hstack([QLd_sync, QRd_sync])

        traj_all = TrajectoryPlanner.make_joint_trajectory_msg(
            joint_names=(self.left_names + self.right_names),
            t0=0.0,
            t_samples=t_common,
            Q_samples=Q_sync,
            Qd_samples=Qd_sync
        )
        if traj_all is not None:
            traj_all.header.frame_id = 'world'
            traj_all.header.stamp = self.get_clock().now().to_msg()
            self.traj_pub.publish(traj_all)

        # (선택) 시각화와 그리퍼 오픈 타이머 – 기존 유지
        try:
            full_points = [self.q_to_point(q) for q in path]
            if len(full_points) >= 2:
                self.marker_pub.publish(self.make_polyline_marker(full_points, 200, 'birrt_path_abs_place', 'b', 0.01))
                path_msg = Path()
                path_msg.header.frame_id = 'world'
                path_msg.header.stamp = self.get_clock().now().to_msg()
                path_msg.poses = [self.q_to_posestamped(q) for q in path]
                self.path_pub_full.publish(path_msg)
        except Exception as e:
            self.get_logger().warn(f"viz skipped: {e}")

    def _concat_paths(self, paths):
        seq = []
        for i, p in enumerate(paths):
            if p is None or len(p) == 0:
                continue
            p = np.asarray(p, float)
            if i > 0 and np.allclose(seq[-1], p[0], atol=1e-9):
                p = p[1:]
            if not seq:
                seq = list(p)
            else:
                seq.extend(list(p))
        return np.asarray(seq, float) if seq else np.zeros((0, len(self.joint_names)))


    # ★ ADD: 세그먼트 경로를 중복 없이 이어붙이기
    def _cbirrt_sequence_and_publish(self):
        if hasattr(self, "_last_joint_state"):
            q_start = np.array(self._last_joint_state, float)
        else:
            q_start = np.asarray(self.goal_q, float)

        seg_paths = []
        cur_q = q_start.copy()

        for i, cfg in enumerate(self.after_grasp_targets):
            # cfg → pin.SE3 변환
            px, py, pz = cfg["pos"]
            r, p, y = [math.radians(d) for d in cfg["rpy_deg"]]
            qx, qy, qz, qw = quat_from_rpy(r, p, y)
            R = pin.Quaternion(np.array([qx, qy, qz, qw])).toRotationMatrix()
            T_pot_goal = pin.SE3(R, np.array([px, py, pz], dtype=float))

            tag = f"wp{i+1}"
            path, q_end = self._plan_cbirrt_segment(cur_q, T_pot_goal, tag)
            if path is None:
                self.get_logger().warn(f"[CBiRRT] waypoint {i+1} failed")
                return
            seg_paths.append(path)
            cur_q = q_end

        full_path = self._concat_paths(seg_paths)
        if len(full_path) < 2:
            self.get_logger().warn("Combined path too short."); return

        # (선택) 시각화
        try:
            pts = [self.q_to_point(q) for q in full_path]
            if len(pts) >= 2:
                self.marker_pub.publish(self.make_polyline_marker(pts, 310, 'cbirrt_seq', 'b', 0.012))
                msg = Path(); msg.header.frame_id='world'; msg.header.stamp=self.get_clock().now().to_msg()
                msg.poses = [self.q_to_posestamped(q) for q in full_path]
                self.path_pub_full.publish(msg)
        except Exception as e:
            self.get_logger().warn(f"viz skipped: {e}")

        # === 하나의 trajectory로 만들고 퍼블리시 ===
        dof_each = 7
        Q_L = full_path[:, :dof_each]
        Q_R = full_path[:, dof_each:2*dof_each]

        qd_max = np.array([1.5]*dof_each); qdd_max = np.array([3.0]*dof_each)
        tp_L = TrajectoryPlanner(
            ds=0.005,
            sdot_start=0.0,
            stop_window_s=0.05,
            alpha_floor=1.5,
            v_min_time=1e-4,
            sample_hz=10.0,
            max_points=100000,
        )

        outL = tp_L.plan(Q_L, qd_max, qdd_max)

        tp_R = TrajectoryPlanner(
            ds=0.005,
            sdot_start=0.0,
            stop_window_s=0.05,
            alpha_floor=1.5,
            v_min_time=1e-4,
            sample_hz=10.0,
            max_points=100000,
        )

        outR = tp_R.plan(Q_R, qd_max, qdd_max)


        # 좌/우 따로
        trajL = TrajectoryPlanner.make_joint_trajectory_msg(self.left_names, 0.0,
            (outL["t_samples"]-outL["t_samples"][0]), outL["Q_samples"], outL["Qd_samples"])
        trajR = TrajectoryPlanner.make_joint_trajectory_msg(self.right_names, 0.0,
            (outR["t_samples"]-outR["t_samples"][0]), outR["Q_samples"], outR["Qd_samples"])
        now = self.get_clock().now().to_msg()
        if trajL: trajL.header.frame_id='world'; trajL.header.stamp=now; self.traj_pub_left.publish(trajL)
        if trajR: trajR.header.frame_id='world'; trajR.header.stamp=now; self.traj_pub_right.publish(trajR)

        self.get_logger().info(
            f"[OPEN-PHASE] Sequence published: left({len(trajL.points) if trajL else 0}) right({len(trajR.points) if trajR else 0})"
        )


        # 동기화된 하나의 JointTrajectory (옵션)
        controller_hz = 100.0; dt = 1.0/controller_hz
        splineL = Spline(outL["t_knots"], Q_L, outL["qdot_knots"])
        splineR = Spline(outR["t_knots"], Q_R, outR["qdot_knots"])
        T_L = float(outL["t_samples"][-1] - outL["t_samples"][0])
        T_R = float(outR["t_samples"][-1] - outR["t_samples"][0])
        T = max(T_L, T_R)
        t_common = np.arange(0.0, T + 0.5*dt, dt)

        QL_sync, QLd_sync, QR_sync, QRd_sync = [], [], [], []
        for t in t_common:
            qL, qLd = Spline.clamp_eval(splineL, outL["t_samples"][0] + t)
            qR, qRd = Spline.clamp_eval(splineR, outR["t_samples"][0] + t)
            QL_sync.append(qL); QLd_sync.append(qLd)
            QR_sync.append(qR); QRd_sync.append(qRd)
        Q_sync  = np.hstack([np.vstack(QL_sync),  np.vstack(QR_sync)])
        Qd_sync = np.hstack([np.vstack(QLd_sync), np.vstack(QRd_sync)])

        traj_all = TrajectoryPlanner.make_joint_trajectory_msg(
            joint_names=(self.left_names + self.right_names),
            t0=0.0, t_samples=t_common, Q_samples=Q_sync, Qd_samples=Qd_sync
        )
        if traj_all is not None:
            traj_all.header.frame_id='world'; traj_all.header.stamp=self.get_clock().now().to_msg()
            self.traj_pub.publish(traj_all)

        # 도달 모니터링 세팅 (마지막 샘플 기준)
        goal_map = {}
        if trajL and trajL.points:
            for nm, q in zip(self.left_names, trajL.points[-1].positions): goal_map[nm]=float(q)
        if trajR and trajR.points:
            for nm, q in zip(self.right_names, trajR.points[-1].positions): goal_map[nm]=float(q)
        
        self._arm_arrival(goal_map, mode='open')
        self.get_logger().info('Arrival action set: OPEN when goal is reached.')



    def _plan_cbirrt_segment(self, q_start: np.ndarray, T_pot_goal: pin.SE3, tag: str):
        # projector 준비(파지 상대자세 측정 포함)
        if not hasattr(self, "_grasp_proj"):
            self._make_grasp_projector_once()

        # 목표 EE (pot 목표 * 파지 당시 상대자세 유지)
        TWL_goal = T_pot_goal * self._T_l_in_pot_meas
        TWR_goal = T_pot_goal * self._T_r_in_pot_meas

        def _to_ps(T: pin.SE3):
            ps = PoseStamped()
            ps.header.frame_id = 'world'
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = T.translation.tolist()
            q = pin.Quaternion(T.rotation).coeffs()
            ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = map(float, q)
            return ps

        obj_cons = ObjectConstraint()
        # 제약 생성
        cons_left  = obj_cons.place_constraints(self.ee_frame_l, _to_ps(TWL_goal), 80.0, 60.0, 5.0)
        cons_right = obj_cons.place_constraints(self.ee_frame_r, _to_ps(TWR_goal), 80.0, 60.0, 5.0)
        from moveit_msgs.msg import Constraints
        combined = Constraints()
        combined.name = f"bimanual_{tag}"
        combined.joint_constraints.extend(cons_left.joint_constraints)
        combined.joint_constraints.extend(cons_right.joint_constraints)
        combined.position_constraints.extend(cons_left.position_constraints)
        combined.position_constraints.extend(cons_right.position_constraints)
        combined.orientation_constraints.extend(cons_left.orientation_constraints)
        combined.orientation_constraints.extend(cons_right.orientation_constraints)

        # IK로 q_goal
        nL = len(self.left_names)
        seed_left  = [float(x) for x in q_start[:nL]]
        seed_right = [float(x) for x in q_start[nL:nL+len(self.right_names)]]

        qL = pin.Quaternion(TWL_goal.rotation).coeffs()
        ik_l = compute_ik_via_moveit(self,"left_arm","world", self.ee_frame_l,
            TWL_goal.translation.tolist(), [float(qL[0]),float(qL[1]),float(qL[2]),float(qL[3])],
            self.left_names, seed_left, timeout=0.8, attempts=12, avoid_collisions=True, wait_mode="poll")
        if ik_l is None: return None, None

        qR = pin.Quaternion(TWR_goal.rotation).coeffs()
        ik_r = compute_ik_via_moveit(self,"right_arm","world", self.ee_frame_r,
            TWR_goal.translation.tolist(), [float(qR[0]),float(qR[1]),float(qR[2]),float(qR[3])],
            self.right_names, seed_right, timeout=0.8, attempts=12, avoid_collisions=True, wait_mode="poll")
        if ik_r is None: return None, None

        q_goal = q_start.copy()
        for i,nm in enumerate(self.left_names):  q_goal[i]              = float(ik_l[nm])
        for i,nm in enumerate(self.right_names): q_goal[nL + i]         = float(ik_r[nm])

        # 계획기
        cplanner = TangentSpaceRRT(
            joint_names=self.joint_names,
            lb=self.lb, ub=self.ub,
            proj_spec=self._grasp_proj,
            group_name=self.group_name,
            state_dim=len(self.joint_names),
            max_iter=20000,
            step_size=0.03,
            edge_check_res=0.02,
            manifold_tol=5e-3,
            tangent_radius=0.4,
        )

        # 1) start는 제약 없이 (충돌/리밋만)
        if not self._check_valid(q_start, f"{tag}_start", constraints=None):
            self.get_logger().warn(f"[TS-RRT] {tag}: q_start invalid (collision/limits)")
            return None, None
        cplanner.set_start(q_start)

        # 2) place 제약 등록
        cplanner.set_constraints(combined)

        # 3) goal은 제약 포함해서 검사
        if not self._check_valid(q_goal, f"{tag}_goal", constraints=combined):
            self.get_logger().warn(f"[TS-RRT] {tag}: q_goal invalid under combined constraints")
            return None, None
        cplanner.set_goal(q_goal)

        # 4) 계획
        self.get_logger().info(f"[TS-RRT] {tag}: solving TS-RRT segment...")
        ok, path = cplanner.solve(max_time=15.0)
        if not ok or path is None:
            self.get_logger().warn(f"[TS-RRT] {tag}: TS-RRT solve failed (no path)")
            return None, None

        path = np.asarray(path, float)
        return path, path[-1]



    def _arm_arrival(self, goal_map: dict, mode: str = 'close'):
        self._arrival_mode = 'open' if mode == 'open' else 'close'
        self._arm_goal = goal_map
        self._awaiting_arrival = True
        self._arrive_count = 0
        self.get_logger().info(f"Arrival monitoring ON ({len(goal_map)} joints), mode={self._arrival_mode}")

        self._cancel_arrival_timeout()

        if self._arrival_mode == 'open':
            def _force_open_cb():
                if self._awaiting_arrival and self._arrival_mode == 'open':
                    self.get_logger().warn(f"Arrival timeout {self._arrival_timeout_s:.1f}s → forcing OPEN")
                    self._awaiting_arrival = False
                    self.open_grippers()
                self._cancel_arrival_timeout()

            self._arrival_deadline = self.get_clock().now().nanoseconds*1e-9 + self._arrival_timeout_s

            def _poll_timeout():
                now = self.get_clock().now().nanoseconds*1e-9
                if now >= self._arrival_deadline:
                    _force_open_cb()

            # ★ 콜백그룹 지정
            self._arrival_timeout_timer = self.create_timer(
                0.1, _poll_timeout, callback_group=self.cbgroup
            )
        def _stale_check():
            if self._awaiting_arrival:
                idle = time.time() - getattr(self, "_last_js_wall", 0.0)
                if idle > 1.0:
                    self.get_logger().warn(f"[ARRIVAL] /joint_states stale {idle:.1f}s while waiting (mode={self._arrival_mode})")
        self._stale_mon_timer = self.create_timer(0.5, _stale_check, callback_group=self.cbgroup)


    def _stop_all_grip_timers(self):
        for nm in ('_grip_hold_timer', '_force_open_timer', '_grip_ramp_timer'):
            t = getattr(self, nm, None)
            if t:
                try:
                    t.cancel()
                except Exception:
                    pass
                setattr(self, nm, None)

    def _cancel_arrival_timeout(self):
        for nm in ('_arrival_timeout_timer','_stale_mon_timer','_arrival_hb_timer'):
            t = getattr(self, nm, None)
            if t:
                try: t.cancel()
                except Exception: pass
                setattr(self, nm, None)


    def _grip_keepalive_tick(self):
        if getattr(self, '_grip_mode', None) != 'open':
            return

        names = [
            'gripper_r_joint1','gripper_r_joint2','gripper_r_joint3','gripper_r_joint4',
            'gripper_l_joint1','gripper_l_joint2','gripper_l_joint3','gripper_l_joint4'
        ]
        targets = [0.0]*8  # 완전 열림 유지

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = targets

        # 두 군데 + MuJoCo 브리지에도 발행
        self.desired_pub.publish(msg)

        self.grip_target_pub.publish(msg)
        if getattr(self, 'use_mujoco_grip', False) and hasattr(self, 'mj_grip_pub'):
            self.mj_grip_pub.publish(msg)

    def _pub_grip(self, names, positions):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = names
        msg.position = positions

        # 기존 퍼블리셔들
        self.desired_pub.publish(msg)
        self.grip_target_pub.publish(msg)

        # MuJoCo 브리지에도 보냄
        if getattr(self, 'use_mujoco_grip', False) and hasattr(self, 'mj_grip_pub'):
            self.mj_grip_pub.publish(msg)




def main():
    rclpy.init()
    node = PathNode()
    try:
        from rclpy.executors import MultiThreadedExecutor
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


                