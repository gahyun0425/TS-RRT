# constraint bidirectional rrt

import numpy as np
import time
from typing import Optional

from new_bimanual_pkg.constraint.projection import validity, check_edge, projection_newton, ConstraintProjec
from new_bimanual_pkg.constraint.constraint import is_state_valid

class ConstraintBiRRT:
    def __init__(self,
                 joint_names,           
                 lb, ub,                
                 group_name="manipulator",
                 state_dim=None,        
                 max_iter=4000,
                 step_size=0.1,
                 edge_check_res=0.05):
        
        # params
        self.joint_names = list(joint_names)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.group_name = group_name
        self.state_dim = int(state_dim if state_dim is not None else len(self.joint_names))

        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.edge_check_res = float(edge_check_res)

        # trees
        self.start_tree = None
        self.goal_tree  = None    

        # constraint env
        self._constraints = None               # moveit_msgs/Constraints (phase별 교체)
        self._validity_cb = None               # build_validity_cb 로 생성
        self._projector: Optional[ConstraintProjec] = None  # 제약 다양체 투영 스펙

    def set_constraints(self, constraints):
        self._constraints = constraints
        self._validity_cb = validity(
            joint_names=self.joint_names,
            lb=self.lb, ub=self.ub,
            group_name=self.group_name,
            constraints=constraints,
            timeout=2.0
        )

    def set_projector(self, proj_spec: Optional[ConstraintProjec]):
        self._projector = proj_spec

    def _new_tree(self, q_root: np.ndarray):
        q_root = np.asarray(q_root, dtype=float)
        return [{'q': q_root, 'parent': None}]
    
    def is_valid(self, q: np.ndarray) -> bool:
        return is_state_valid(q, self.joint_names, self.lb, self.ub,
                            group_name=self.group_name, timeout=2.0,
                            constraints=None)
    
    def set_start(self, q_start: np.ndarray):
        q = np.asarray(q_start, float)
        if self._projector is not None:
            q, ok = projection_newton(q, self._projector, self.lb, self.ub)
            if not ok:
                raise RuntimeError("start root projection failed")
        
        if not self.is_valid(q):
            raise RuntimeError("start root is in collision/limits")
        self.start_tree = self._new_tree(q)
    
    def set_goal(self, q_goal: np.ndarray):
        q = np.asarray(q_goal, float)
        if self._projector is not None:
            q, ok = projection_newton(q, self._projector, self.lb, self.ub)
            if not ok:
                raise RuntimeError("goal root projection failed")
        if not self.is_valid(q):
            raise RuntimeError("goal root is in collision/limits")
        self.goal_tree = self._new_tree(q)

    def sample_random_config(self) -> np.ndarray:
        for _ in range(50):
            q = np.random.uniform(self.lb, self.ub, size=self.state_dim)
            if self._projector is not None:
                q, ok = projection_newton(q, self._projector, self.lb, self.ub)
                if not ok:
                    print("[CBiRRT] WARN: projection failed in sample_random_config")
                    continue
            q = np.clip(q, self.lb, self.ub)
            if self.is_valid(q):
                return q
        # fallback: 그냥 반환
        return np.clip(np.random.uniform(self.lb, self.ub, size=self.state_dim), self.lb, self.ub)
    
    def nearest_idx(self, tree, q: np.ndarray) -> int:
        dists = [np.linalg.norm(n['q'] - q) for n in tree]
        return int(np.argmin(dists))
    
    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        d = q_to - q_from
        L = np.linalg.norm(d)
        if L == 0.0:
            return q_from.copy()
        step = min(self.step_size, L)
        return q_from + d * (step / L)

    def edge_is_valid(self, qa: np.ndarray, qb: np.ndarray):
        if self._validity_cb is None:
            return True
        ok = check_edge(
            q_from=qa, q_to=qb,
            validity=self._validity_cb,
            step=self.edge_check_res,
            projector=self._projector,
            lb=self.lb, ub=self.ub
        )
        return ok
    
    def neighbors(self, tree, q_new: np.ndarray, radius: float):
        return [i for i, n in enumerate(tree) if np.linalg.norm(n['q'] - q_new) <= radius]

    def add_node(self, tree, q: np.ndarray, parent_idx: int) -> int:
        tree.append({'q': q, 'parent': parent_idx})
        return len(tree) - 1

    def path_from(self, tree, idx:int):
        path = []
        while idx is not None:
            path.append(tree[idx]['q'])
            idx = tree[idx]['parent']
        return path[::-1]
    
    def extend(self, tree, q_target):
        i_near = self.nearest_idx(tree, q_target)
        q_near = tree[i_near]['q']
        q_new  = self.steer(q_near, q_target)

        if self._projector is not None:
            q_new, ok = projection_newton(q_new, self._projector, self.lb, self.ub)
            if not ok:
                print("[CBiRRT] WARN: projection failed in extend()")
                return None

        # 에지 유효성 검사
        if not self.edge_is_valid(q_near, q_new):
            return None

        # 노드 추가 + 인덱스 반환
        return self.add_node(tree, q_new, i_near)
    
    def try_connect(self, tree_a, tree_b, idx_new_in_a):
        q_target = tree_a[idx_new_in_a]['q']
        i_near_b = self.nearest_idx(tree_b, q_target)
        q_curr   = tree_b[i_near_b]['q']
        parent   = i_near_b

        while True:
            q_next = self.steer(q_curr, q_target)

            if self._projector is not None:
                q_next, ok = projection_newton(q_next, self._projector, self.lb, self.ub)
                if not ok:
                    print("[CBiRRT] WARN: projection failed in try_connect()")
                    return None, None

            if not self.edge_is_valid(q_curr, q_next):
                return None, None

            # tree_b에 연속적으로 붙이기
            idx_new_b = self.add_node(tree_b, q_next, parent)

            # 도달 (스텝이 더 이상 줄어들지 않거나 충분히 가까움)
            if np.linalg.norm(q_next - q_target) <= self.edge_check_res:
                return idx_new_in_a, idx_new_b

            # 다음 스텝
            parent = idx_new_b
            q_curr = q_next

    def solve(self, max_time: float | None = None):
        if self.start_tree is None or self.goal_tree is None:
            raise RuntimeError("start/goal이 설정되지 않았습니다. set_start(), set_goal() 먼저 호출하세요.")

        t0 = time.time()

        for it in range(self.max_iter):

            if max_time is not None and (time.time() - t0) > max_time:
                break

            # goal bias
            if np.random.random() < 0.9:
                mu = self.goal_tree[0]['q'] # 정규 분포의 평균 -> goal 근처에서 샘플을 뽑겠다
                sig = 0.05 # 정규 분포의 표준 편차
                q_rand = np.clip(np.random.normal(mu, sig, size=self.state_dim), self.lb, self.ub)
            else:
                q_rand = self.sample_random_config()

            if not self.is_valid(q_rand):
                continue

            idx_new_a = self.extend(self.start_tree, q_rand)

            if idx_new_a is None:
                self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
                continue

            idx_a, idx_b = self.try_connect(self.start_tree, self.goal_tree, idx_new_a)
            if idx_a is not None and idx_b is not None:
                path_a = self.path_from(self.start_tree, idx_a)
                path_b = self.path_from(self.goal_tree, idx_b)
                full = path_a + path_b[::-1]
                return True, np.vstack(full)

            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree

        return False, None
    
    def set_grasp_projector(self, proj_spec: ConstraintProjec):
        self._projector = proj_spec




