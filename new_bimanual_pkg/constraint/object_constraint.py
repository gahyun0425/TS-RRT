# Object Constraint 설정
# position constraint, orientation constraint, path constraint

import math

from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, BoundingVolume
from shape_msgs.msg import SolidPrimitive
from typing import Optional, Tuple, List

class ObjectConstraint:
    
    # robot position에 대한 constraint 설정
    def position_constraint(self, link_name: str, 
                            target_pose_world, 
                            eps_xyz =(0.08, 0.08, 0.07)) -> PositionConstraint:
        ex, ey, ez = eps_xyz

        prim = SolidPrimitive() # 단순 3D 물체를 표현하는 ROS 메세지
        prim.type = SolidPrimitive.BOX # 물체가 BOX라는 것을 표현
        prim.dimensions = [2*ex, 2*ey, 2*ez] # BOX 전체 길이

        bv = BoundingVolume() # 허용/제약 영역임을 표시하는 3D 공간 덩어리를 표시하는 메세지
        bv.primitives.append(prim) # bv.primitives 리스트에 prim 추가
        bv.primitive_poses.append(target_pose_world.pose) # 박스 중심/자세 = 목표 포즈

        pc = PositionConstraint()
        pc.header.frame_id = target_pose_world.header.frame_id  # ← 목표 포즈 프레임 그대로
        pc.link_name = link_name
        pc.constraint_region = bv
        pc.weight = 1.0
        return pc
    
    # robot orientation에 대한 constraint 설정
    def orientation_constraint(self, link_name: str,
                               frame_id: str = "world",
                               tol_xyz_deg: Tuple[float,float,float] = (180.0,180.0,180.0),
                               quat: Tuple[float,float,float,float] = (0.0,0.0,0.0,1.0)) -> OrientationConstraint:
        
        oc = OrientationConstraint()
        oc.header.frame_id = frame_id
        oc.link_name = link_name
        oc.orientation.x, oc.orientation.y, oc.orientation.z, oc.orientation.w = quat
        oc.absolute_x_axis_tolerance = math.radians(tol_xyz_deg[0])
        oc.absolute_y_axis_tolerance = math.radians(tol_xyz_deg[1])
        oc.absolute_z_axis_tolerance = math.radians(tol_xyz_deg[2])
        oc.weight = 1.0
        return oc
    
    # robot path에 대한 constraint 설정 (position + orientation)
    def path_constraints(self,
                     pcs: List[PositionConstraint],
                     ocs: Optional[List[OrientationConstraint]] = None,
                     name: str = "path_constraints") -> Constraints:

        cons = Constraints()
        cons.name = name
        cons.position_constraints = pcs
        if ocs:
            cons.orientation_constraints = ocs
        return cons
    
    def place_constraints(self, ee_link, place_pose_world,
                            eps_xy_mm=50.0, eps_z_mm=60.0,
                            level_deg=5.0):
        # 위치 박스 크기 (± -> BOX 길이 = 2*eps)
        eps = (eps_xy_mm/1000.0, eps_xy_mm/1000.0, eps_z_mm/1000.0)

        # PositionConstraint: 목표 포즈를 중심으로 한 BOX
        pc = self.position_constraint(ee_link, place_pose_world, eps_xyz=eps)

        # ★ OrientationConstraint: 목표 포즈의 쿼터니언을 그대로 사용
        ori = place_pose_world.pose.orientation
        oc = self.orientation_constraint(
            ee_link,
            frame_id=place_pose_world.header.frame_id,
            tol_xyz_deg=(level_deg, level_deg, 10.0),
            quat=(ori.x, ori.y, ori.z, ori.w),
        )

        return self.path_constraints(pcs=[pc], ocs=[oc], name="place_phase")

        





        