# TOPP-RA & Cubic Spline Interpolation
from __future__ import annotations
import numpy as np

from typing import Dict, Sequence
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

from new_bimanual_pkg.planning.TOPP import PathTOPP
from new_bimanual_pkg.planning.spline import Spline

class TrajectoryPlanner:
    def __init__(
        self,
        *,
        # TOPP 설정
        ds: float = 1e-3,
        sdot_start: float = 0.0,
        stop_window_s: float = 0.05,
        alpha_floor: float = 0.2,
        v_min_time: float = 1e-4,
        # 샘플링 설정
        sample_hz: float = 50.0,
        max_points: int = 100_000,
    ):
        self.topp = PathTOPP(
            ds=ds,
            sdot_start=sdot_start,
            stop_window_s=stop_window_s,
            alpha_floor=alpha_floor,
            v_min_time=v_min_time,
        )
        self.sample_hz = sample_hz
        self.max_points = max_points

    def plan(
        self,
        Q: np.ndarray,
        qd_max: np.ndarray,
        qdd_max: np.ndarray,
    ) -> Dict[str, np.ndarray]:

        # 1) TOPP: t_knots, qdot_knots 등 계산
        topp_res = self.topp.compute(Q, qd_max, qdd_max)

        # 2) spline
        spline = Spline(
            topp_res["t_knots"],
            Q,
            topp_res["qdot_knots"],
        )

        # 3) 균일 시간 샘플링
        t_samples, Q_samples, Qd_samples = spline.sample_uniform(
            sample_hz=self.sample_hz,
            max_points=self.max_points,
        )

        # 필요한 값들 패키징해서 반환
        return dict(
            t_knots=topp_res["t_knots"],
            qdot_knots=topp_res["qdot_knots"],
            t_samples=t_samples,
            Q_samples=Q_samples,
            Qd_samples=Qd_samples,
            s_grid=topp_res["s_grid"],
            v=topp_res["v"],
            t_grid=topp_res["t_grid"],
        )

    @staticmethod
    def make_joint_trajectory_msg(
        joint_names: Sequence[str],
        t0: float,
        t_samples: np.ndarray,
        Q_samples: np.ndarray,
        Qd_samples: np.ndarray | None = None,
        hold_sec: float = 1.0,
    ) -> JointTrajectory:

        traj = JointTrajectory()
        traj.joint_names = list(joint_names)

        # 샘플들을 point로 넣기
        for i, tt in enumerate(t_samples):
            pt = JointTrajectoryPoint()
            pt.positions = Q_samples[i].tolist()

            if Qd_samples is not None:
                pt.velocities = Qd_samples[i].tolist()

            tau = max(0.0, float(tt - t0))
            sec = int(tau)
            nsec = int((tau - sec) * 1e9)
            pt.time_from_start = Duration(sec=sec, nanosec=nsec)

            traj.points.append(pt)

        # 마지막 점 보정 + hold 포인트 추가
        if len(traj.points) > 0:
            # 마지막 속도 0
            traj.points[-1].velocities = [0.0] * len(joint_names)

            # hold 포인트
            last = traj.points[-1]
            hold = JointTrajectoryPoint()
            hold.positions = list(last.positions)
            hold.velocities = [0.0] * len(joint_names)

            last_tau = (
                last.time_from_start.sec
                + last.time_from_start.nanosec * 1e-9
            )
            tau = last_tau + float(hold_sec)
            sec = int(tau)
            nsec = int((tau - sec) * 1e9)
            hold.time_from_start = Duration(sec=sec, nanosec=nsec)

            traj.points.append(hold)

        return traj