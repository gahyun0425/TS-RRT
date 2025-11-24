# joint limit

import numpy as np
import yaml, os

from urdf_parser_py.urdf import URDF

def load_joint_limits(urdf_path: str, joint_limits_yaml: str, joint_names: list):
    # URDF에서 설정한 joint_limit 값으로 lb, ub 설정

    # URDF 파싱
    with open(urdf_path, 'r') as f:
        urdf_xml = f.read()
    robot = URDF.from_xml_string(urdf_xml)

    # URDF에서 읽은 limit 설정
    limits = {}
    for j in robot.joints:
        if j.type == 'continuous':
            limits[j.name] = (-np.pi, np.pi)
        elif j.type in ('revolute', 'prismatic'):
            limits[j.name] = (float(j.limit.lower), float(j.limit.upper))

    if joint_limits_yaml and os.path.exists(joint_limits_yaml):
        with open(joint_limits_yaml, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        jlim = cfg.get('joint_limits', {})
        for name, v in jlim.items():
            has = v.get('has_position_limits', False)
            if has:
                limits[name] = (float(v['min_position']), float(v['max_position']))
            else:
                # has_position_limits: false 인 경우 연속 관절처럼 처리
                limits.setdefault(name, (-np.pi, np.pi))

    # 순서 정렬해서 배열 생성 (없으면 연속 관절처럼 fallback)
    lb = []
    ub = []
    for nm in joint_names:
        lo, hi = limits.get(nm, (-np.pi, np.pi))
        lb.append(lo)
        ub.append(hi)
    return np.array(lb, dtype=float), np.array(ub, dtype=float)
