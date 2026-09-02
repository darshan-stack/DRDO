"""Small ROS-independent SE(3) helpers used by the ROS adapter and tests."""
from __future__ import annotations
import numpy as np

def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    if len(points) == 0: return points.copy()
    hom = np.concatenate([points, np.ones((len(points), 1), dtype=np.float32)], axis=1)
    return (hom @ matrix.T)[:, :3].astype(np.float32)

def quaternion_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = x*x+y*y+z*z+w*w
    if n < 1e-12: return np.eye(4)
    s = 2.0/n
    xx, yy, zz = x*x*s, y*y*s, z*z*s
    xy, xz, yz = x*y*s, x*z*s, y*z*s
    wx, wy, wz = w*x*s, w*y*s, w*z*s
    out = np.eye(4)
    out[:3,:3] = [[1-yy-zz, xy-wz, xz+wy], [xy+wz, 1-xx-zz, yz-wx], [xz-wy, yz+wx, 1-xx-yy]]
    return out

def transform_from_ros_transform(t) -> np.ndarray:
    out = quaternion_to_matrix(float(t.rotation.x), float(t.rotation.y), float(t.rotation.z), float(t.rotation.w))
    out[:3, 3] = [float(t.translation.x), float(t.translation.y), float(t.translation.z)]
    return out
