from __future__ import annotations
import numpy as np
import random
import time
from typing import Callable, Optional

# q와 constraint를 받아 유효/무효를 돌려주는 콜백 타입
IsValidFn = Callable[[np.ndarray, Optional[object]], bool]

# joint limit 검사
def _joint_limit_ok(q: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> bool:
    q  = np.asarray(q,  float)
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)
    if q.shape != lb.shape or lb.shape != ub.shape:
        return False
    if (q < lb).any() or (q > ub).any():
        return False
    return True

# GetStateValidity 서비스로 충돌, 제약 포함 상태 유효성 확인
def _call_moveit_validity(node,
                          q: np.ndarray,
                          constraints: Optional[object] = None,
                          timeout: float = 0.25) -> bool:

    required = ("joint_names", "lb", "ub", "group_name", "state_valid_cli")
    if not all(hasattr(node, k) for k in required):
        return False

    if not _joint_limit_ok(q, node.lb, node.ub):
        return False

    cli = getattr(node, "state_valid_cli", None)
    if cli is None:
        return False

    try:
        from sensor_msgs.msg import JointState
        from moveit_msgs.msg import RobotState
        from moveit_msgs.srv import GetStateValidity

        js = JointState(name=list(node.joint_names),
                        position=np.asarray(q, float).tolist())
        rs = RobotState(joint_state=js)

        req = GetStateValidity.Request()
        req.robot_state = rs
        req.group_name = str(node.group_name)
        if constraints is not None:
            req.constraints = constraints

        fut = cli.call_async(req)

        # spin_until_future_complete 우선 시도
        try:
            import rclpy
            rclpy.spin_until_future_complete(node, fut, timeout_sec=float(timeout))
        except Exception:
            # 실패 시 폴링
            t0 = time.time()
            while not fut.done():
                if time.time() - t0 > timeout:
                    try:
                        node.get_logger().warn("path_simplify: validity timeout")
                    except Exception:
                        pass
                    return False
                time.sleep(0.001)

        if not fut.done():
            return False

        res = fut.result()
        return bool(res and res.valid)
    except Exception:
        return False

# Moveit 검사기를 간단히 호출할 수 있는 is_valid 콜백으로 래핑
def make_is_valid_from_node(node):
    def _cb(q: np.ndarray, constraints: Optional[object] = None) -> bool:
        return _call_moveit_validity(node, q, constraints, timeout=0.25)
    return _cb

# 직선 보간 중간 샘플들이 전부 유효한지 체크
def _edge_is_valid(q1: np.ndarray,
                   q2: np.ndarray,
                   is_valid: IsValidFn,
                   constraints: Optional[object] = None,
                   max_step: float = 0.012,
                   min_samples: int = 3) -> bool:
    q1 = np.asarray(q1, float); q2 = np.asarray(q2, float)
    seg_len = float(np.linalg.norm(q2 - q1)) + 1e-12
    n = max(min_samples, int(np.ceil(seg_len / max(1e-9, max_step))))
    for k in range(1, n):  # 1..n-1
        a = k / n
        q = (1.0 - a) * q1 + a * q2
        if not is_valid(q, constraints):
            return False
    return True

# 충돌 없이 건너뛰기 가능한 중간 점들을 삭제해 경로 점 수를 줄임
def reduce_vertices(path: np.ndarray,
                    is_valid: IsValidFn,
                    constraints: Optional[object] = None,
                    max_step: float = 0.012,
                    passes: int = 3) -> np.ndarray:
    if path is None or len(path) < 3:
        return np.asarray(path, float)
    P = np.asarray(path, float)
    for _ in range(max(1, passes)):
        keep = [0]
        i = 0
        while i < len(P) - 2:
            if _edge_is_valid(P[i], P[i+2], is_valid, constraints, max_step=max_step):
                j = i + 2
                while j + 1 < len(P) and _edge_is_valid(P[i], P[j+1], is_valid, constraints, max_step=max_step):
                    j += 1
                keep.append(j)
                i = j
            else:
                keep.append(i + 1)
                i += 1
        if keep[-1] != len(P) - 1:
            keep.append(len(P) - 1)

        new_keep = []
        seen = set()
        for idx in keep:
            if idx not in seen:
                new_keep.append(idx)
                seen.add(idx)
        P = P[np.asarray(new_keep, dtype=int)]
    return P

# 경로 길이 + 곡률(라플라시안 제곱) 가중합으로 비용 계산
def _path_cost(P: np.ndarray, lam_len: float = 1.0, lam_curv: float = 4.0) -> float:
    P = np.asarray(P, float)
    if len(P) < 2:
        return 0.0
    L = float(np.sum(np.linalg.norm(P[1:] - P[:-1], axis=1)))
    if len(P) < 3:
        return lam_len * L
    lap = P[:-2] - 2.0 * P[1:-1] + P[2:]
    C = float(np.sum(np.sum(lap * lap, axis=1)))
    return lam_len * L + lam_curv * C

# 무작위 구간을 직선으로 대체해 비용이 줄면 채택하는 쇼트컷
def shortcut_path_curvaware(path: np.ndarray,
                            is_valid: IsValidFn,
                            constraints: Optional[object] = None,
                            max_step: float = 0.012,
                            attempts: int = 900,
                            lam_len: float = 1.0,
                            lam_curv: float = 4.0,
                            rng: Optional[random.Random] = None) -> np.ndarray:
    if path is None or len(path) < 3:
        return np.asarray(path, float)
    P = np.asarray(path, float).copy()
    rnd = rng or random
    base_cost = _path_cost(P, lam_len, lam_curv)

    for _ in range(max(1, attempts)):
        if len(P) < 3:
            break
        i = rnd.randint(0, len(P) - 3)
        j = rnd.randint(i + 2, len(P) - 1)
        if _edge_is_valid(P[i], P[j], is_valid, constraints,
                          max_step=max_step, min_samples=3):
            cand = np.vstack([P[:i+1], P[j:]])
            new_cost = _path_cost(cand, lam_len, lam_curv)
            if new_cost + 1e-12 < base_cost:
                P = cand
                base_cost = new_cost
    return P

# 중간 점을 이웃 평균 쪽으로 이동하되 충돌 없는 경우에만 스무딩 반영
def laplacian_smooth(path: np.ndarray,
                     is_valid: IsValidFn,
                     constraints: Optional[object] = None,
                     step: float = 0.35,
                     iters: int = 25,
                     max_step: float = 0.012) -> np.ndarray:
    if path is None or len(path) < 3:
        return np.asarray(path, float)
    P = np.asarray(path, float).copy()
    N = len(P)
    for _ in range(max(1, iters)):
        changed = False
        for i in range(1, N - 1):
            p_old = P[i].copy()
            p_new = p_old + step * (P[i-1] - 2.0 * p_old + P[i+1])
            ok = (_edge_is_valid(P[i-1], p_new, is_valid, constraints,
                                 max_step=max_step, min_samples=3)
                  and
                  _edge_is_valid(p_new, P[i+1], is_valid, constraints,
                                 max_step=max_step, min_samples=3))
            if ok:
                P[i] = p_new
                changed = True
        if not changed:
            break
    return P

# 인접 점 간 거리가 max_step 이하가 되도록 중간 점을 삽입
def densify_by_maxstep(path: np.ndarray, max_step: float = 0.012) -> np.ndarray:
    if path is None or len(path) < 2:
        return np.asarray(path, float)
    P = np.asarray(path, float)
    out = [P[0]]
    for i in range(len(P) - 1):
        a, b = P[i], P[i+1]
        seg = float(np.linalg.norm(b - a))
        n = max(1, int(np.ceil(seg / max(1e-9, max_step))))
        for k in range(1, n + 1):
            out.append((1 - k/n) * a + (k/n) * b)
    return np.asarray(out, float)

# OMPL PathSimplifier(점 감소→쇼트컷→B-스플라인)로 MoveIt 유효성 기반 경로 단순화.
def simplify_path_with_ompl(node,
                            path: np.ndarray,
                            constraints: Optional[object] = None,
                            *,
                            max_step: float = 0.012,
                            reduce_vertices: bool = True,
                            do_shortcut: bool = True,
                            do_bspline: bool = True,
                            shortcut_max_time: float = 2.0,
                            bspline_max_time: float = 1.5,
                            rng_seed: Optional[int] = None
                            ) -> np.ndarray:

    if path is None or len(path) < 2:
        return np.asarray(path, float)

    try:
        import ompl.base as ob
        import ompl.geometric as og
    except Exception:
        # OMPL 바인딩이 없으면 None 반환해서 폴백 사용
        return None

    P = np.asarray(path, float)
    dof = int(P.shape[1])

    # 1) 상태공간 & 경계
    space = ob.RealVectorStateSpace(dof)
    bounds = ob.RealVectorBounds(dof)
    for i in range(dof):
        bounds.setLow(i, float(getattr(node, "lb")[i]))
        bounds.setHigh(i, float(getattr(node, "ub")[i]))
    space.setBounds(bounds)

    # 2) SpaceInformation
    si = ob.SpaceInformation(space)

    # validity checker: MoveIt 서비스 재사용
    is_valid = make_is_valid_from_node(node)
    class _VC(ob.StateValidityChecker):
        def __init__(self, si):
            super().__init__(si)
        def isValid(self, state: ob.State) -> bool:
            q = np.array([state[i] for i in range(dof)], dtype=float)
            return is_valid(q, constraints)

    si.setStateValidityChecker(_VC(si))

    # 분해능 설정: edge 샘플 간격 비슷하게 (절대/상대 혼용)
    # OMPL은 fraction으로 많이 쓰므로 conservative 하게 설정
    # 각 Joint step을 유사하게 맞추기 위해 resolution도 조정
    si.setStateValidityCheckingResolution(max(1e-5, max_step))  # absolute step (for R^n)
    space.setup()
    si.setup()

    # 3) 기존 path -> PathGeometric
    pg = og.PathGeometric(si)
    for q in P:
        s = ob.State(space)
        for i in range(dof):
            s[i] = float(q[i])
        pg.append(s)

    # 4) Simplifier
    if rng_seed is not None:
        ob.RandomNumbers().setSeed(rng_seed)

    simplifier = og.PathSimplifier(si)

    # (a) 꼭짓점 감소
    if reduce_vertices:
        # reduceVertices: 인접한 점을 삭제해도 유효하면 제거
        simplifier.reduceVertices(pg)

    # (b) 쇼트컷
    if do_shortcut:
        # shortcutPath(path, max_time, range_ratio=0.0)
        # range_ratio=0이면 내부 기본값 사용 (적응형)
        simplifier.shortcutPath(pg, shortcut_max_time)

    # (c) B-스플라인 스무딩
    if do_bspline:
        # smoothBSpline(path, max_time, min_change)
        simplifier.smoothBSpline(pg, bspline_max_time)

    # 5) 결과 -> numpy
    out = []
    for i in range(pg.getStateCount()):
        st = pg.getState(i)
        out.append([st[j] for j in range(dof)])
    return np.asarray(out, float)

# 가능하면 OMPL로, 실패 시 자체(리덕션→쇼트컷→스무딩) 파이프라인으로 경로 단순화하는 엔트리 함수.
def simplify_path(path: np.ndarray,
                  is_valid: Optional[IsValidFn] = None,
                  constraints: Optional[object] = None,
                  *,
                  node: Optional[object] = None,
                  max_step: float = 0.02,
                  red_passes: int = 3,
                  shortcut_attempts: int = 900,
                  lam_len: float = 1.0,
                  lam_curv: float = 4.0,
                  smooth_iters: int = 25,
                  smooth_step: float = 0.35,
                  do_densify: bool = False,
                  rng: Optional[random.Random] = None) -> np.ndarray:

    # OMPL 경로 단순화 시도 (node가 있고 필수 필드가 있을 때)
    if node is not None and all(hasattr(node, k) for k in ("lb", "ub", "joint_names", "group_name", "state_valid_cli")):
        ompl_out = simplify_path_with_ompl(
            node=node,
            path=path,
            constraints=constraints,
            max_step=max_step,
            reduce_vertices=True,
            do_shortcut=True,
            do_bspline=True,
            shortcut_max_time=0.45,
            bspline_max_time=0.25,
            rng_seed=None
        )
        if isinstance(ompl_out, np.ndarray) and len(ompl_out) >= 2:
            if do_densify:
                ompl_out = densify_by_maxstep(ompl_out, max_step=max_step)
            return ompl_out

    # ---- 폴백: 기존 파이프라인 ----
    if is_valid is None:
        # node로부터 생성 가능하면 생성
        if node is not None:
            is_valid = make_is_valid_from_node(node)
        else:
            raise ValueError("simplify_path: OMPL을 쓰려면 node가 필요하고, 폴백을 쓰려면 is_valid 콜백이 필요합니다.")

    if path is None or len(path) < 2:
        return np.asarray(path, float)

    P = np.asarray(path, float)
    P = reduce_vertices(P, is_valid, constraints, max_step=max_step, passes=red_passes)
    P = shortcut_path_curvaware(P, is_valid, constraints, max_step=max_step,
                                attempts=shortcut_attempts,
                                lam_len=lam_len, lam_curv=lam_curv, rng=rng)
    P = laplacian_smooth(P, is_valid, constraints, step=smooth_step,
                         iters=smooth_iters, max_step=max_step)
    if do_densify:
        P = densify_by_maxstep(P, max_step=max_step)
    return P
