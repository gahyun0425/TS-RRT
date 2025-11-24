# Bidirectional RRT code

import numpy as np
import time

from new_bimanual_pkg.constraint.constraint import is_state_valid
from new_bimanual_pkg.planning.linear_interpolation import linear_edge_samples

class BiRRT:
    def __init__(self,
                 joint_names,         
                 lb, ub,              
                 group_name="manipulator",
                 state_dim=None,        # 관절 공간 차원 -> RRT가 탐색하는 q의 길이 -> joint_names
                 max_iter=2000,
                 step_size=0.1,
                 edge_check_res=0.05):
        
        # parameter
        self.joint_names = list(joint_names)
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        self.group_name = group_name
        # 따로 state_dim에 대한 입력이 없다면 joint_name으로 입력됨
        self.state_dim = int(state_dim if state_dim is not None else len(self.joint_names))

        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.edge_check_res = float(edge_check_res)

        # tree를 담아두는 변수
        self.start_tree = None
        self.goal_tree = None

    # tree의 맨 처음 기준이 되는 노드 = root node
    def _new_tree(self, q_root: np.ndarray):
        q_root = np.asarray(q_root, dtype=float) # 루트 상태를 float 배열로
        return [{'q': q_root, 'parent': None}] # 루트 노드: 부모 없음, 비용 0
    
    # start tree 초기화
    def set_start(self, q_start: np.ndarray):
        self.start_tree = self._new_tree(q_start)

    # goal tree 초기화
    def set_goal(self, q_goal: np.ndarray):
        self.goal_tree = self._new_tree(q_goal)

    # collision detectioin
    def is_valid(self, q: np.ndarray) -> bool:
        return is_state_valid(q, self.joint_names, self.lb, self.ub, self.group_name)

    def sample_random_config(self) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub, size=self.state_dim)
    
    def nearest_idx(self, tree, q: np.ndarray) ->  int:
        dists = [np.linalg.norm(n['q'] - q) for n in tree] # 각 노드까지의 거리
        return int(np.argmin(dists)) # 가장 가까운 인덱스 반환

    def steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        d = q_to - q_from # 진행 방향 벡터
        L = np.linalg.norm(d) # 거리
        if L == 0.0: # 동일 지점이면,
            return q_from.copy() # 그대로 반환
        step = min(self.step_size, L) # 최대 스텝 제한
        return q_from + d * (step / L) # 한 스텝 전진한 점 
    
    #edge linear interpolation 진행 및 edge 샘플링
    def edge_is_valid(self, qa: np.ndarray, qb: np.ndarray) -> bool:
        if np.linalg.norm(qb - qa) == 0.0:
            return self.is_valid(qb)

        for qi in linear_edge_samples(qa, qb, self.edge_check_res):
            if not self.is_valid(qi):
                return False
        return True
    
    def neighbors(self, tree, q_new: np.ndarray, radius: float):
        # 반경 내에 있는 모든 노드 인덱스를 반환 (간단 고정 반경)
        return [i for i, n in enumerate(tree) if np.linalg.norm(n['q'] - q_new) <= radius]

    def add_node(self, tree, q: np.ndarray, parent_idx: int) -> int:
        # 트리에 새 노드 추가 후 인데스 반환
        tree.append({'q': q, 'parent': parent_idx})
        return len(tree) -1
    
    def path_from(self, tree, idx:int):
        # 루트까지 부모를 따라가며 경로 복원
        path = []
        while idx is not None:
            path.append(tree[idx]['q'])
            idx = tree[idx]['parent']
        return path[::-1] # 시작-> 현재 순서로 뒤집어 반환
    
    # 한 트리 RRT 확장
    def extend(self, tree, q_target: np.ndarray):
        """최근접에서 q_target 방향으로 한 스텝 전진. 성공 시 새 노드 인덱스 반환, 실패 시 None."""
        i_near = self.nearest_idx(tree, q_target)
        q_near = tree[i_near]['q']
        q_new  = self.steer(q_near, q_target)
        if not self.edge_is_valid(q_near, q_new):
            return None
        return self.add_node(tree, q_new, i_near)

    
    def try_connect(self, tree_a, tree_b, idx_new_in_a):
        """B 트리가 A의 새 노드까지 닿을 때까지(막힐 때까지) 여러 스텝 전진."""
        q_target = tree_a[idx_new_in_a]['q']
        i_near_b = self.nearest_idx(tree_b, q_target)
        q_curr   = tree_b[i_near_b]['q']
        parent   = i_near_b
        while True:
            q_next = self.steer(q_curr, q_target)
            if not self.edge_is_valid(q_curr, q_next):
                return None, None  # 더 못감 (충돌)
            parent = self.add_node(tree_b, q_next, parent)
            q_curr = q_next
            if np.linalg.norm(q_curr - q_target) <= self.step_size:
                if self.edge_is_valid(q_curr, q_target):
                    return idx_new_in_a, parent  # 연결 성공


    def solve(self, max_time: float | None = None):
        if self.start_tree is None or self.goal_tree is None:
            raise RuntimeError("start/goal이 설정되지 않았습니다. set_start(), set_goal() 먼저 호출하세요.")

        for _ in range(self.max_iter):

            q_rand = self.sample_random_config()
            if not self.is_valid(q_rand):
                continue

            # A 트리 확장
            idx_new_a = self.extend(self.start_tree, q_rand)
            if idx_new_a is None:
                # 실패했어도 트리 스왑해서 교대 확장
                self.start_tree, self.goal_tree = self.goal_tree, self.start_tree
                continue

            # B 트리와 연결 시도
            idx_a, idx_b = self.try_connect(self.start_tree, self.goal_tree, idx_new_a)
            if idx_a is not None and idx_b is not None:
                path_a = self.path_from(self.start_tree, idx_a)
                path_b = self.path_from(self.goal_tree, idx_b)
                full = path_a + path_b[::-1]
                return True, np.vstack(full)

            # 번갈아 확장
            self.start_tree, self.goal_tree = self.goal_tree, self.start_tree

        return False, None