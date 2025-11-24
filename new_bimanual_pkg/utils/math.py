# RPY -> quaternion 

import math

def quat_from_rpy(roll, pitch, yaw):
    """RPY(rad) -> quaternion [x,y,z,w] (world/arm_base_link 기준)"""
    cr = math.cos(roll*0.5);  sr = math.sin(roll*0.5)
    cp = math.cos(pitch*0.5); sp = math.sin(pitch*0.5)
    cy = math.cos(yaw*0.5);   sy = math.sin(yaw*0.5)
    qw = cr*cp*cy + sr*sp*sy
    qx = sr*cp*cy - cr*sp*sy
    qy = cr*sp*cy + sr*cp*sy
    qz = cr*cp*sy - sr*sp*cy
    return [qx, qy, qz, qw]