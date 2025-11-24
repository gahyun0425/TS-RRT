# projection
# Newton_Raphson method
# q가 유효한지 판단해주는 함수
# edge 유효성 검사
# 손잡이를 양팔로 잡았다는 rigid 제약

import numpy as np
import pinocchio as pin

from typing import Tuple, Callable, List, Optional
from moveit_msgs.msg import Constraints
from dataclasses import dataclass
from new_bimanual_pkg.constraint.constraint import is_state_valid
from new_bimanual_pkg.planning.linear_interpolation import linear_edge_samples

@dataclass
class ConstraintProjec:
    # h(q) 제약식
    function: Callable[[np.ndarray], np.ndarray]

    # J(q) 자코비안
    jacobian: Callable[[np.ndarray], np.ndarray]
    
    # 허용 오차
    tol: float = 1e-6

    # 반복 횟수
    max_iters: int = 50

# Newton-Raphson
def projection_newton(q_in: np.ndarray,
                      proj: ConstraintProjec,
                      lb: np.ndarray,
                      ub: np.ndarray) -> Tuple[np.ndarray, bool]:
    
    q = np.asarray(q_in, float).copy()
    lb = np.asarray(lb, float)
    ub = np.asarray(ub, float)

    for _ in range(max(1, proj.max_iters)):
        # h(q) 계산. q가 제약에서 얼마나 벗어났는지 (오차)를 계산하는 단계
        h = np.atleast_1d(proj.function(q)).astype(float)

        # 제약 위반 정도 -> 오차가 충분히 작으면 성공 종료
        if np.linalg.norm(h) <= proj.tol:
            return np.clip(q, lb, ub), True
        
        # J(q) 계산. 제약식이 q에 어떻게 민감한지(기울기)를 가져오는 단계.
        J = np.asarray(proj.jacobian(q), float)
        if J.ndim != 2:
            raise ValueError("jacobian(q) must be 2-D (m x N)")
        
        # 선형 대수 관련 에러가 발생하면 except로
        try:
            # pseudo-inverse 사용
            J_pinv = np.linalg.pinv(J)
            dq = - J_pinv @ h

        # 선형대수 에러 발생 시 q와 false 반환
        except np.linalg.LinAlgError:
            return q, False
        
        # 각 관절값 클리핑
        q = np.clip(q + dq, lb, ub)

    # 루프가 끝난 후 최종 판정 True or False
    # 마지막 q에서 제약 위반량 h(q)를 배열 형태로 뽑음
    h_final = np.atleast_1d(proj.function(q)).astype(float)
    # 최종 q, 성공 여부를 한 번에 돌려주는 라인
    return np.clip(q, lb, ub), (np.linalg.norm(h_final) <= proj.tol)

# q 유효성 검사 함수
def validity(joint_names: List[str],
             lb: np.ndarray,
             ub: np.ndarray,
             group_name: str,
             constraints: Optional[Constraints], # constraint 일 때만 적용 가능
             timeout: float = 2.0) -> Callable[[np.ndarray], bool]:
    
    def _valid(q: np.ndarray) -> bool:
        return is_state_valid(q, joint_names, lb, ub, group_name, timeout, constraints)
    return _valid

# edge 유효성 검사: 선형보간 + projection + collision detection
def check_edge(q_from: np.ndarray,
               q_to: np.ndarray,
               validity: Callable[[np.ndarray], bool],
               step: float = 0.05,
               projector: Optional[ConstraintProjec] = None,
               lb: Optional[np.ndarray] = None,
               ub: Optional[np.ndarray] = None
               ) -> Tuple[bool, List[np.ndarray]]:
    
    q_from = np.asarray(q_from, float)
    q_to = np.asarray(q_to, float)

    # 시작점 포함해서 경로 저장
    path: List[np.ndarray] = [q_from.copy()]

   # q_from과 q_to가 사실상 같으면: q_to만 검사하고 종료
    if float(np.linalg.norm(q_to - q_from)) <= 1e-9:
        ok = validity(q_to)
        if ok:
            path.append(q_to.copy())
        return ok, path

    # linear_edge_samples가 qa 제외, qb 포함 샘플을 넘겨줌
    for q_lin in linear_edge_samples(q_from, q_to, res=step):
        q_lin = np.asarray(q_lin, float)

        # projection이 주어졌을 때
        if projector is not None:
            if lb is None or ub is None:
                raise ValueError("lb/ub 비어있음")
            
            # newton method 사용 -> true, false 반환
            q_proj, ok = projection_newton(q_lin, projector, lb, ub)
            print("iter proj:", np.linalg.norm(projector.function(q_proj)), "ok?", ok)

            # projection에 실패했다면
            if not ok:
                # false 반환
                return False, path
            
            # projection에 성공 시 실제 유효성 검사 q = q_proj
            q_chk = q_proj
        else:
            # projector를 사용하지 않은 경우 선형 보간한 값 그대로 사용
            q_chk = q_lin

        # projection 된 q가 제약에 만족하는지 확인
        if not validity(q_chk):
            return False, path
        
        # 만족 시 path에 추가 후 계속 진행
        path.append(q_chk)

    return True, path

# pot 손잡이를 양팔로 잡았다는 rigid 제약 projector 생성
def pot_grasp_projector(model: pin.Model,
                        data: pin.Data,
                        ee_frame_l: str,
                        ee_frame_r: str,
                        T_pot0: pin.SE3,          # (미사용: 인터페이스 유지)
                        T_l_in_pot: pin.SE3,
                        T_r_in_pot: pin.SE3,
                        tol: float = 1e-1,
                        max_iters=50,
                        damping=1e-6
                        ) -> ConstraintProjec:
    
    fidL = model.getFrameId(ee_frame_l)
    fidR = model.getFrameId(ee_frame_r)

    def h(q: np.ndarray):
        # joint q에 대해
        # FK 계산
        pin.forwardKinematics(model, data, q)
        # frame pose들 oMf 업데이트
        pin.updateFramePlacements(model, data)

        T_wL = data.oMf[fidL] # world 기준 왼손 pose (SE3)
        T_wR = data.oMf[fidR] # world 기준 오른손 pose

        # 왼손이 pot을 잡고 있다고 가정 했을 때 pot 위치 추정
        T_pot_est = T_wL * T_l_in_pot.inverse()
        # 그때 오른손 pose 추정
        T_wR_expected = T_pot_est * T_r_in_pot

        # 실제 오른손 위치 - 기대 오른손 위치 -> q에서 양손이 pot을 같이 잡고 있는지를 확인하는 위치 오차
        err_pos = T_wR.translation - T_wR_expected.translation
        # 오차를 10개 스케일링해서 반환 -> h(q)는 현재ㅐ q가 제약에서 얼마나 벗어났는지를 나타내는 residual -> 빠르게 projectioon 하기 위해
        return 10.0 * err_pos
    
    def J(q: np.ndarray):
        # 모든 frame jacobian 계산
        pin.computeJointJacobians(model, data, q)
        pin.updateFramePlacements(model, data)

        T_wL = data.oMf[fidL]
        T_wR = data.oMf[fidR]
        T_pot_est     = T_wL * T_l_in_pot.inverse()
        T_wR_expected = T_pot_est * T_r_in_pot

        # world 기준 왼손 frame jacobian
        JL_w = pin.computeFrameJacobian(model, data, q, fidL, pin.WORLD)
        # world 기준 오른손 frame jacobian
        JR_w = pin.computeFrameJacobian(model, data, q, fidR, pin.WORLD) 

        # 왼손의 움직임이 오른손 pse에 어떻게 영향을 미치는지 계산
        # Ad는 SE3의 adjoint (action matrix)
        Ad = (T_wR_expected * T_wL.inverse()).toActionMatrix()
        # 왼손의 움직임에 따라 오른손 pose 예측
        Jexp_from_L = Ad @ JL_w

        # 위치 row만 추출
        # 실제 오른손 위치 Jacobian- 기대 오른손 위치 Jacobian
        #(q) = pos_R - pos_R_expected 의 Jacobian
        J_rigid_pos = JR_w[:3, :] - Jexp_from_L[:3, :]
        return J_rigid_pos
    
    return ConstraintProjec(
        function=h,
        jacobian=J,
        tol=1.0,        
        max_iters=2000,  
    )