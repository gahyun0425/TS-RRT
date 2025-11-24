# TS-RRT

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from new_bimanual_pkg.constraint.projection import ConstraintProjec, projection_newton
from new_bimanual_pkg.constraint.constraint import is_state_valid

def compute_projection_matrix(J: np.ndarray) -> np.ndarray:
    """
    P(q) = I - J^† J
    J: (m, n)
    """
    J = np.asarray(J, float)
    n = J.shape[1]
    J_pinv = np.linalg.pinv(J)        # (n, m)
    P = np.eye(n) - J_pinv @ J        # (n, n)
    return P


def gram_schmidt_from_P(
    P: np.ndarray,
    q_root: np.ndarray,
    parent_root: Optional[np.ndarray],
    rank_tol: float = 1e-8,
) -> np.ndarray:
    """
    P가 정의하는 tangent subspace에서 정규직교 basis {d_i}를 만든다.
    """
    n = P.shape[0]
    basis: List[np.ndarray] = []

    # 1) d1: parent 방향
    if parent_root is not None:
        d1 = q_root - parent_root
        d1 = P @ d1
        if np.linalg.norm(d1) > rank_tol:
            d1 = d1 / np.linalg.norm(d1)
            basis.append(d1)

    # 2) 나머지 방향들
    for _ in range(n * 2):
        if len(basis) >= n:
            break

        v = np.random.randn(n)
        v = P @ v

        # 기존 basis에 대해 직교화
        for b in basis:
            v -= np.dot(v, b) * b

        norm = np.linalg.norm(v)
        if norm < rank_tol:
            continue

        v /= norm
        basis.append(v)

    if not basis:
        # P가 거의 0이면 tangent 공간이 없는 경우
        return np.zeros((0, n), float)

    return np.stack(basis, axis=0)   # (k, n)


def default_tangent_bounds(num_dirs: int, radius: float) -> np.ndarray:
    """
    각 tangent 방향별 bound s_i 설정 (초기버전: 전부 동일 radius).
    """
    return np.full(num_dirs, radius, dtype=float)

@dataclass
class TangentPlane:
    root_q: np.ndarray
    basis: np.ndarray
    bounds: np.ndarray
    parent_root_q: Optional[np.ndarray] = None
    node_indices: List[int] = field(default_factory=list)
    tree_id: int = 0 
        
# tanget 공간에서 샘플링
def sample_on_tangent_plane(plane: TangentPlane) -> np.ndarray:
    k, n = plane.basis.shape
    alphas = np.empty(k, dtype=float)
    for i in range(k):
        low, high = -plane.bounds[i], plane.bounds[i]
        if i == 0:
            # backtracking 방지: 0 ~ s_1
            low = 0.0
        alphas[i] = np.random.uniform(low, high)

    # basis: (k, n), alphas: (k,) → (n,)
    delta_q = alphas @ plane.basis
    return plane.root_q + delta_q

# TS-RRT 노드/트리 구조
class TSNode:
    __slots__ = ("q", "parent", "plane_id", "tree_id")

    def __init__(self, q: np.ndarray, parent: Optional[int],
                 plane_id: int, tree_id: int):
        self.q = np.asarray(q, float)
        self.parent = parent
        self.plane_id = plane_id
        self.tree_id = tree_id

# TS-RRT 본체
class TangentSpaceRRT:
    def __init__(self,
                 joint_names,
                 lb, ub,
                 proj_spec: ConstraintProjec,
                 group_name: str = "manipulator",
                 state_dim: Optional[int] = None,
                 max_iter: int = 4000,
                 step_size: float = 0.1,
                 edge_check_res: float = 0.05,
                 manifold_tol: float = 1e-3,  # EM (논문에서 threshold)
                 tangent_radius: float = 0.2):

        self.joint_names = list(joint_names)
        self.lb = np.asarray(lb, float)
        self.ub = np.asarray(ub, float)
        self.group_name = group_name
        self.state_dim = int(state_dim if state_dim is not None else len(self.joint_names))

        self.max_iter = int(max_iter)
        self.step_size = float(step_size)
        self.edge_check_res = float(edge_check_res)
        self.manifold_tol = float(manifold_tol)
        self.tangent_radius = float(tangent_radius)
        self._constraints = None 

        # equality constraint projector (TS-RRT의 핵심)
        self.proj_spec = proj_spec

        # trees
        self.start_tree: List[TSNode] = []
        self.goal_tree: List[TSNode] = []

        # tangent planes
        self.tangent_planes: List[TangentPlane] = []

    def set_constraints(self, constraints):
        """MoveIt Constraints를 저장 (CBiRRT와 동일한 인터페이스)."""
        self._constraints = constraints

    def is_valid(self, q: np.ndarray) -> bool:
        """ inequality + (선택적) MoveIt Constraints 검사. """
        return is_state_valid(
            q, self.joint_names, self.lb, self.ub,
            group_name=self.group_name,
            timeout=2.0,
            constraints=self._constraints 
        )

    def add_plane(self, plane: TangentPlane) -> int:
        self.tangent_planes.append(plane)
        return len(self.tangent_planes) - 1
    
    def _select_tangent_plane_idx(self) -> int:
        """
        SelectTangentPlane(): 노드 수가 적은 tangent plane을 조금 더 자주 선택.
        """
        if not self.tangent_planes:
            raise RuntimeError("No tangent planes")

        weights = []
        for plane in self.tangent_planes:
            # node 수가 적을수록 weight 크게
            w = 1.0 / (len(plane.node_indices) + 1.0)
            weights.append(w)

        weights = np.asarray(weights, dtype=float)
        weights /= weights.sum()
        return int(np.random.choice(len(self.tangent_planes), p=weights))

    
    def _create_tangent_plane(self,
                            q: np.ndarray,
                            tree_id: int,
                            parent_root_q: Optional[np.ndarray] = None,
                            project: bool = True) -> Optional[int]:
        """
        논문 Table II: CreateTangentPlane(q)를 구현한 부분.
        - q는 manifold 근처의 config
        - projection_newton으로 manifold 위로 보정
        - J, P로부터 tangent basis 생성
        - bounds는 일단 고정 radius 사용
        """

        # 1) Projection(q)
        if project:
            q_proj, ok = projection_newton(q, self.proj_spec, self.lb, self.ub)
            if not ok:
                # projection 실패 → 이 tangent plane 안 쓰고 넘어간다
                # print / logger 쓰고 싶으면 여기서 로그 찍어도 됨
                return None
        else:
            q_proj = np.asarray(q, float)

        # 2) J, P 계산
        J = np.asarray(self.proj_spec.jacobian(q_proj), float)
        P = compute_projection_matrix(J)

        # 3) Gram-Schmidt로 basis
        basis = gram_schmidt_from_P(P, q_proj, parent_root_q)
        k = basis.shape[0]

        # basis가 하나도 안 나오면 plane 만들 이유가 없다
        if k == 0:
            return None

        bounds = default_tangent_bounds(k, self.tangent_radius)

        plane = TangentPlane(
            root_q=q_proj,
            basis=basis,
            bounds=bounds,
            parent_root_q=parent_root_q,
            tree_id=tree_id
        )
        return self.add_plane(plane)

    
    # Start / Goal 설정
    def set_start(self, q_start: np.ndarray):
        q_start = np.asarray(q_start, float)

        # ✅ equality projection 안 함: 이미 manifold 위라고 가정
        if not self.is_valid(q_start):
            raise RuntimeError("start config invalid (collision/limits)")

        # tangent plane은 q_start 주변에서 만들되, 여기서도 projection은 스킵
        plane_id = self._create_tangent_plane(
            q_start,
            tree_id=0,
            parent_root_q=None,
            project=False,
        )
        if plane_id is None:
            raise RuntimeError("failed to create start tangent plane")

        node = TSNode(q=q_start, parent=None, plane_id=plane_id, tree_id=0)
        self.start_tree = [node]
        self.tangent_planes[plane_id].node_indices.append(0)


    # Tangent plane 선택 / nearest / extend
    # Tangent plane 선택 -> 노드가 적을수록 더 자주 뽑히는 가중치로 구현 & 나중에 curvatue weight를 곱해도 됨. -> 아직은 반영X
    # 양방향 path planning이기 때문에 Tangent plane 선택 과정이 더욱 필요
    def set_goal(self, q_goal: np.ndarray):
        q_goal = np.asarray(q_goal, float)
        qf, ok = projection_newton(q_goal, self.proj_spec, self.lb, self.ub)
        if not ok:
            raise RuntimeError("goal projection failed")
        if not self.is_valid(qf):
            raise RuntimeError("goal config invalid (collision/limits)")

        plane_id = self._create_tangent_plane(qf, tree_id=1, parent_root_q=None)
        if plane_id is None:
            raise RuntimeError("failed to create goal tangent plane")

        node = TSNode(q=qf, parent=None, plane_id=plane_id, tree_id=1)
        self.goal_tree = [node]
        self.tangent_planes[plane_id].node_indices.append(0)

    
    # plane 내부에서 nearest 노드 -> q_near 선택 과정
    def _nearest_in_plane(self, tree: List[TSNode],
                          plane: TangentPlane,
                          q_rand: np.ndarray) -> int:
        """
        해당 tangent plane에 속한 노드들 중 q_rand와 가장 가까운 노드 index.
        """
        assert plane.node_indices, "plane has no nodes"
        dists = []
        for idx in plane.node_indices:
            q = tree[idx].q
            dists.append(np.linalg.norm(q - q_rand))
        best_local_idx = int(np.argmin(dists))
        return plane.node_indices[best_local_idx]
    
    # Extend(q_near, q_rand) Tangent plane 위에서 RRT step
    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        d = q_to - q_from
        L = np.linalg.norm(d)
        if L == 0.0:
            return q_from.copy()
        step = min(self.step_size, L)
        return q_from + d * (step / L)
    
    def _extend_on_plane(self,
                        tree: List[TSNode],
                        plane_id: int,
                        q_rand: np.ndarray) -> Optional[int]:
        """
        Extend(q_near, q_rand) on tangent plane k.
        - tangent plane 내부에서 한 스텝 나가고
        - inequality constraint 만족하면 node 추가
        - equality는 'lazy' → manifold에서의 거리만 threshold 체크
        """
        plane = self.tangent_planes[plane_id]
        if plane.tree_id == 0:
            base_tree = self.start_tree
        else:
            base_tree = self.goal_tree
        assert base_tree is tree

        idx_near = self._nearest_in_plane(tree, plane, q_rand)
        q_near = tree[idx_near].q
        q_new = self._steer(q_near, q_rand)

        # inequality constraint 체크
        if not self.is_valid(q_new):
            return None

        # equality constraint 거리 체크 (h(q) norm)
        e = np.atleast_1d(self.proj_spec.function(q_new)).astype(float)
        if np.linalg.norm(e) > self.manifold_tol:
            # threshold 넘으면 "새 tangent plane 후보"로 사용 (논문 line 14–16)
            plane_id_new = self._create_tangent_plane(
                q_new,
                tree_id=plane.tree_id,
                parent_root_q=plane.root_q
            )
            # ★ projection 실패 / basis 없음 등으로 plane 못 만들면 그냥 이 step 버림
            if plane_id_new is None:
                return None

            plane_new = self.tangent_planes[plane_id_new]
            q_root_new = plane_new.root_q

            # node 추가는 projection된 root를 새 node로 쓰는 방식
            node_idx = len(tree)
            node = TSNode(q=q_root_new,
                          parent=idx_near,
                          plane_id=plane_id_new,
                          tree_id=plane.tree_id)
            tree.append(node)
            plane_new.node_indices.append(node_idx)
            return node_idx

        # 아직 manifold 근처: 그냥 q_new로 새로운 node 추가
        node_idx = len(tree)
        node = TSNode(q=q_new,
                      parent=idx_near,
                      plane_id=plane_id,
                      tree_id=plane.tree_id)
        tree.append(node)
        plane.node_indices.append(node_idx)
        return node_idx



    
    # 두 tree 연결 (Connect + ExtractPath + LazyProjection)
    def _edge_is_valid_ambient(self, qa: np.ndarray, qb: np.ndarray) -> bool:
        """
        ambient 공간에서 선형 보간하며 inequality constraint만 체크.
        TS-RRT에는 equality projection을 'lazy'하게 하기 위해 여기서는 projector 사용X.
        """
        dist = float(np.linalg.norm(qb - qa))
        if dist < 1e-9:
            return self.is_valid(qb)

        num = max(2, int(dist / self.edge_check_res))
        for i in range(1, num + 1):
            s = i / num
            q = (1.0 - s) * qa + s * qb
            if not self.is_valid(q):
                return False
        return True
    
    def _try_connect_to_other_tree(self,
                                   newly_extended_q: np.ndarray,
                                   src_tree: List[TSNode],
                                   dst_tree: List[TSNode]) -> Optional[tuple]:
        """
        Connect(qnew): 새로 확장된 q_new에서 반대편 tree와 연결 시도.
        """
        # 반대 tree에서 nearest node 찾기
        dists = [np.linalg.norm(n.q - newly_extended_q) for n in dst_tree]
        idx_near = int(np.argmin(dists))
        q_near = dst_tree[idx_near].q

        # 간단한 RRT-Connect style: 직선으로 여러 스텝 가면서 validity 체크
        q_curr = q_near
        parent = idx_near
        while True:
            q_next = self._steer(q_curr, newly_extended_q)

            if not self._edge_is_valid_ambient(q_curr, q_next):
                return None

            # dst_tree에 새 노드 붙이기
            idx_new = len(dst_tree)
            node = TSNode(q=q_next, parent=parent,
                          plane_id=dst_tree[parent].plane_id,
                          tree_id=dst_tree[parent].tree_id)
            dst_tree.append(node)
            parent = idx_new
            q_curr = q_next

            if np.linalg.norm(q_next - newly_extended_q) <= self.edge_check_res:
                # 연결 성공
                return (idx_new, )
            
    # Path 추출하여 전체 projection
    def _extract_path(self,
                    start_tree: List[TSNode],
                    goal_tree: List[TSNode],
                    idx_in_start: int,
                    idx_in_goal: int) -> np.ndarray:
        """
        start_tree[idx_in_start]와 goal_tree[idx_in_goal]가 만났다고 가정하고
        full path를 반환.
        """
        path_a = []
        i = idx_in_start
        while i is not None:
            node = start_tree[i]
            path_a.append(node.q)
            i = node.parent
        path_a = path_a[::-1]

        path_b = []
        j = idx_in_goal
        while j is not None:
            node = goal_tree[j]
            path_b.append(node.q)
            j = node.parent

        full = path_a + path_b  # goal tree는 이미 방향이 맞다고 가정
        return np.vstack(full)  # (N, dof)

    def _lazy_projection_path(self, path: np.ndarray) -> np.ndarray:
        """
        LazyProjection(Path): path 상의 노드들을 manifold로 투영.
        """
        proj_pts = []
        for q in path:
            q_proj, ok = projection_newton(q, self.proj_spec, self.lb, self.ub)
            if not ok:
                raise RuntimeError("Lazy projection failed for some node")
            proj_pts.append(q_proj)
        return np.vstack(proj_pts)

    def solve(self, max_time: Optional[float] = None):
        """
        TS-RRT(q_init, q_final) 메인 루프.
        - set_start(), set_goal() 먼저 호출되어 있어야 함.
        """
        import time
        if not self.start_tree or not self.goal_tree:
            raise RuntimeError("set_start(), set_goal() 먼저 호출하세요.")

        t0 = time.time()

        tree_a = self.start_tree
        tree_b = self.goal_tree

        for it in range(self.max_iter):
            if max_time is not None and (time.time() - t0) > max_time:
                break

            # 6) SelectTangentPlane()
            k = self._select_tangent_plane_idx()
            plane = self.tangent_planes[k]

            # 해당 tangent plane이 start_tree/goal_tree 중 어느쪽인지에 따라 트리 선택
            tree = tree_a if plane.tree_id == 0 else tree_b

            # 7) RandomSampleOnTangentPlane
            q_rand = sample_on_tangent_plane(plane)

            # 8–9) NearestNode + Extend
            idx_new = self._extend_on_plane(tree, k, q_rand)
            if idx_new is None:
                continue

            q_new = tree[idx_new].q

            # 10) inequality constraint는 _extend_on_plane에서 이미 검사했으므로 생략

            # 11) Connect(qnew) 시도
            #    - plane.tree_id = 0이면 tree는 start_tree, 반대는 goal_tree
            if plane.tree_id == 0:
                src_tree, dst_tree = self.start_tree, self.goal_tree
                idx_src_new = idx_new
            else:
                src_tree, dst_tree = self.goal_tree, self.start_tree
                idx_src_new = idx_new

            connect_ret = self._try_connect_to_other_tree(
                newly_extended_q=q_new,
                src_tree=src_tree,
                dst_tree=dst_tree
            )

            if connect_ret is not None:
                idx_in_dst = connect_ret[0]
                # start / goal 인덱스로 다시 정리
                if plane.tree_id == 0:
                    # start_tree에 새 node, goal_tree에 idx_in_dst
                    path_ambient = self._extract_path(
                        start_tree=self.start_tree,
                        goal_tree=self.goal_tree,
                        idx_in_start=idx_src_new,
                        idx_in_goal=idx_in_dst
                    )
                else:
                    # goal_tree에 새 node, start_tree에 idx_in_dst
                    path_ambient = self._extract_path(
                        start_tree=self.start_tree,
                        goal_tree=self.goal_tree,
                        idx_in_start=idx_in_dst,
                        idx_in_goal=idx_src_new
                    )
                # 13) LazyProjection(Path)
                path_proj = self._lazy_projection_path(path_ambient)
                return True, path_proj

            # 15–16) q_new > EM 일 때 새로운 tangent plane 생성은
            #         _extend_on_plane 안에서 처리하고 있음.

            # 17) Tree.AddNode(qnew), Tree.AddEdge(qnear, qnew)
            #     이것도 _extend_on_plane 안에서 처리

        return False, None


