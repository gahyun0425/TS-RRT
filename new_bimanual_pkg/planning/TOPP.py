# TOPP-RA 
from __future__ import annotations

import math
import numpy as np

from typing import Dict, Tuple

# 0 나누기/정수화 문제 피하기 위한 아주 작은 값 (epsilon)
EPS = 1e-12

class PathTOPP:
    def __init__(
        self,
        ds: float = 1e-3, # 샘플 간격 -> 구간을 얼마나 잘게 쪼갤지
        sdot_start: float = 0.0, # 시작 s' -> 시작할 때 path 진행 속도 (보통은 0)
        stop_window_s: float = 0.05, # 끝(1.0)에 가까운 구간에서 속도를 -으로 줄이기 위한 구간 길이
        alpha_floor: float = 0.2, # 중간에서 너무 느려지지 않도록 vmax의 일정 비율 이하로 떨어지지 않게 만드는 바닥 비율
        v_min_time: float = 1e-4, # 시간 적분 시 평균 속도 하한 (총 시간 폭주 방지)
    ):
        
        self.ds = ds
        self.sdot_start = sdot_start
        self.stop_window_s = stop_window_s
        self.alpha_floor = alpha_floor
        self.v_min_time = v_min_time

    # 특정 구간에서의 방향 dq/ds에 대해, 각 joint velocity limit으로부터 허용 가능한 최대 s'을 계산하는 정적 메서드
    @staticmethod
    def _speed_cap(dqds_k: np.ndarray, qd_max: np.ndarray) -> float:
        # 조인트 개수
        D = dqds_k.shape[0]
        # 각 joint i에 대해 계산 -> dqds가 거의 0이면 (그 joint는 그 구간에서 거의 안움직이면) 제한 리스트에 넣지 않음
        caps = [
            qd_max[i] / abs(dqds_k[i])
            for i in range(D)
            if abs(dqds_k[i]) > 1e-12
        ]
        # 모든 joint의 제약 중 가장 작은 값이 실제 s' 상한 caps가 비어있으면 사실상 제한이 없므로 큰 값 반환
        return min(caps) if caps else 1e6
    
    # 가속도 제한으로부터 허용 가능한 s''의 최소/최대값 계산하는 정적 메서드
    @staticmethod
    def _acc_bounds(dqds_k: np.ndarray, qdd_max: np.ndarray) -> Tuple[float, float]:
        # 처음에는 넓은 범위에서 시작
        lo, hi = -1e9, 1e9
        D = dqds_k.shape[0] # 조인트 개수
        for i in range(D):
            a = abs(dqds_k[i])
            if a <= 1e-12:
                continue
            b = qdd_max[i] / a
            lo = max(lo, -b)
            hi = min(hi, b)
        return lo, hi # 최종적으로 허용 가능한 s'' 범위 반환
    
    # 실제 TOPP를 수행하는 main
    # input: waypoint, 속도 상한, 가속도 상한
    # output: 각 knot에서의 t, q'등 정보를 dict로 반환
    def compute(
        self,
        Q: np.ndarray,
        qd_max: np.ndarray,
        qdd_max: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        
        Q = np.asarray(Q, float)
        qd_max = np.asarray(qd_max, float)
        qdd_max = np.asarray(qdd_max, float)

        N, D = Q.shape
        if N < 2:
            raise ValueError("웨이포인트는 최소 2개 이상 필요")

        # 각 segment 길이 계산
        seg_len = np.linalg.norm(np.diff(Q, axis=0), axis=1)
        s = np.zeros(N) # 누적 거리 배열 s 초기화
        s[1:] = np.cumsum(seg_len) # segment 길이를 누적해서 각 waypoint까지의 총 거리 계산 
        L = s[-1] # 전체 경로 길이 L

        if L <= EPS:
            # 모든 점이 같다 → 시간 0, 속도 0
            t_knots = np.zeros(N)
            qdot_knots = np.zeros_like(Q)
            return dict(
                s_grid=np.array([0.0, 1.0]),
                v=np.zeros(2),
                t_grid=t_knots.copy(),
                seg_id=np.array([0, 0]),
                s_knots=np.linspace(0, 1, N),
                t_knots=t_knots,
                qdot_knots=qdot_knots,
            )

        s_knots = s / L
        seg_s = np.maximum(np.diff(s_knots), EPS)
        dqds = [(Q[k + 1] - Q[k]) / seg_s[k] for k in range(N - 1)]

        # --- s-grid 생성 ---
        s_grid = [s_knots[0]]
        seg_id = [0]
        for k in range(N - 1):
            s0, s1 = s_knots[k], s_knots[k + 1]
            n = max(1, int(math.ceil((s1 - s0) / self.ds)))
            for j in range(1, n + 1):
                s_val = min(s0 + j * (s1 - s0) / n, s1)
                s_grid.append(s_val)
                seg_id.append(k)
        s_grid = np.array(s_grid, float)
        seg_id = np.array(seg_id, int)
        M = len(s_grid)

        # --- forward pass ---
        v = np.zeros(M, float)
        v[0] = float(self.sdot_start)

        for k in range(M - 1):
            dqds_k = dqds[seg_id[k]]
            vmax = self._speed_cap(dqds_k, qd_max)
            v[k] = min(v[k], vmax)

            lo, hi = self._acc_bounds(dqds_k, qdd_max)
            a = max(0.0, hi)  # 가속만
            ds_k = s_grid[k + 1] - s_grid[k]
            v[k + 1] = math.sqrt(max(0.0, v[k] * v[k] + 2 * a * ds_k))

        # --- backward pass ---
        for k in range(M - 2, -1, -1):
            dqds_k = dqds[seg_id[k + 1]]
            vmax = self._speed_cap(dqds_k, qd_max)
            v[k + 1] = min(v[k + 1], vmax)

            lo, hi = self._acc_bounds(dqds_k, qdd_max)
            a = min(0.0, lo)  # 감속만
            ds_k = s_grid[k + 1] - s_grid[k]
            v_prev = math.sqrt(max(0.0, v[k + 1] * v[k + 1] - 2 * abs(a) * ds_k))
            v[k] = min(v[k], v_prev)

        # --- 중앙부 바닥 속도 (alpha_floor) ---
        if self.alpha_floor > 0.0:
            vmax_grid = np.array(
                [self._speed_cap(dqds[seg_id[i]], qd_max) for i in range(M)],
                float,
            )
            sw = max(0.0, min(1.0, self.stop_window_s))
            s_start = max(0.0, 1.0 - sw) if sw > 0.0 else 1.0
            for k in range(M - 1):  # 마지막은 감속용으로 남겨둠
                if s_grid[k] < s_start:
                    v[k] = min(
                        max(v[k], self.alpha_floor * vmax_grid[k]),
                        vmax_grid[k],
                    )

        # --- 끝 윈도우에서 0으로 수렴 ---
        sw = max(0.0, min(1.0, self.stop_window_s))
        if sw > 0.0:
            s_start = max(0.0, 1.0 - sw)
            vmax_grid = np.array(
                [self._speed_cap(dqds[seg_id[i]], qd_max) for i in range(M)],
                float,
            )
            idx_start = int(np.searchsorted(s_grid, s_start))
            idx_start = min(max(idx_start, 0), M - 2)
            v_ref = max(v[idx_start], 1e-9)

            for k in range(idx_start, M):
                s_cur = s_grid[k]
                if s_cur >= s_start:
                    tau = 0.5 * (
                        1.0
                        + math.cos(
                            math.pi * (s_cur - s_start) / max(sw, EPS)
                        )
                    )
                else:
                    tau = 1.0
                tau = max(0.0, min(1.0, tau))
                v[k] = min(v[k], tau * v_ref, vmax_grid[k])

            v[-1] = 0.0

        # --- 시간 적분 (s → t) ---
        t_grid = np.zeros_like(s_grid)
        V_MIN_TIME = float(max(self.v_min_time, 1e-8))
        for k in range(1, M):
            ds_k = s_grid[k] - s_grid[k - 1]
            vavg = 0.5 * (v[k] + v[k - 1])
            if not np.isfinite(vavg):
                vavg = 0.0
            vavg = max(vavg, V_MIN_TIME)
            t_grid[k] = t_grid[k - 1] + ds_k / vavg

        # --- 웨이포인트 시각/속도 ---
        sdot_knots = np.interp(s_knots, s_grid, v)
        qdot_knots = np.zeros_like(Q)
        for k in range(N - 1):
            qdot_knots[k] = dqds[k] * sdot_knots[k]
        qdot_knots[-1] = np.zeros(D)  # 종단 속도 0

        t_knots = np.interp(s_knots, s_grid, t_grid)

        return dict(
            s_grid=s_grid,
            v=v,
            t_grid=t_grid,
            seg_id=seg_id,
            s_knots=s_knots,
            t_knots=t_knots,
            qdot_knots=qdot_knots,
        )