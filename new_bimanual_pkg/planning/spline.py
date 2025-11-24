# spline interpolation

from __future__ import annotations
import math
from typing import Tuple
import numpy as np

EPS = 1e-12


class Spline:

    def __init__(self, t_knots: np.ndarray, Q: np.ndarray, V: np.ndarray):
        """
        t_knots: (N,) 시간 매듭
        Q: (N,D) 위치
        V: (N,D) 속도
        """
        self.t = np.asarray(t_knots, float)
        self.Q = np.asarray(Q, float)
        self.V = np.asarray(V, float)

        assert self.t.ndim == 1
        assert self.Q.ndim == 2
        assert self.V.shape == self.Q.shape

        self.N, self.D = self.Q.shape
        if self.N < 2:
            raise ValueError("스플라인 매듭은 최소 2개 이상 필요")

    def _seg_idx(self, tt: float) -> int:
        # 주어진 시간 tt가 포함되는 구간 인덱스 k (t[k] <= tt <= t[k+1])
        if tt <= self.t[0]:
            return 0
        if tt >= self.t[-1]:
            return self.N - 2
        return int(np.searchsorted(self.t, tt) - 1)

    @staticmethod
    def _basis(u: float):
        # Hermite basis 값
        u2 = u * u
        u3 = u2 * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        return h00, h10, h01, h11

    @staticmethod
    def _basis_d(u: float):
        # Hermite basis 도함수
        h00p = 6 * u * u - 6 * u
        h10p = 3 * u * u - 4 * u + 1
        h01p = -6 * u * u + 6 * u
        h11p = 3 * u * u - 2 * u
        return h00p, h10p, h01p, h11p

    def eval(self, tt: float) -> np.ndarray:
        """q(t) 위치"""
        k = self._seg_idx(tt)
        t0, t1 = self.t[k], self.t[k + 1]
        h = max(t1 - t0, EPS)
        u = (tt - t0) / h

        h00, h10, h01, h11 = self._basis(u)
        q0, v0 = self.Q[k], self.V[k]
        q1, v1 = self.Q[k + 1], self.V[k + 1]

        return h00 * q0 + h10 * h * v0 + h01 * q1 + h11 * h * v1

    def eval_d(self, tt: float) -> np.ndarray:
        """q̇(t) 속도"""
        k = self._seg_idx(tt)
        t0, t1 = self.t[k], self.t[k + 1]
        h = max(t1 - t0, EPS)
        u = (tt - t0) / h

        h00p, h10p, h01p, h11p = self._basis_d(u)
        q0, v0 = self.Q[k], self.V[k]
        q1, v1 = self.Q[k + 1], self.V[k + 1]

        dq_du = h00p * q0 + h10p * h * v0 + h01p * q1 + h11p * h * v1
        return dq_du / h

    def sample_uniform(
        self,
        sample_hz: float = 50.0,
        max_points: int = 100_000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        t0, t1 = float(self.t[0]), float(self.t[-1])
        T_total = t1 - t0

        if (not np.isfinite(T_total)) or (T_total <= 0.0):
            # 비정상: 매듭만 반환
            t_samples = self.t.copy()
            Qs = np.vstack([self.eval(tt) for tt in t_samples])
            Qds = np.vstack([self.eval_d(tt) for tt in t_samples])
            return t_samples, Qs, Qds

        M = int(max(2, math.ceil(T_total * float(sample_hz))))
        if M > int(max_points):
            M = int(max_points)

        t_samples = np.linspace(t0, t1, M)
        Qs = np.vstack([self.eval(tt) for tt in t_samples])
        Qds = np.vstack([self.eval_d(tt) for tt in t_samples])

        return t_samples, Qs, Qds

    def clamp_eval(self, t: float):

        t0, t1 = float(self.t[0]), float(self.t[-1])

        if t <= t0:
            q = self.eval(t0)
            qd = np.zeros_like(q)
            return q, qd

        if t >= t1:
            q = self.eval(t1)
            qd = np.zeros_like(q)
            return q, qd

        return self.eval(t), self.eval_d(t)
