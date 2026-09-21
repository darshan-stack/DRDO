import numpy as np

from ffem.perception.motion import VoxelMotionDetector


def test_ego_motion_compensation_suppresses_static_scene_motion():
    detector = VoxelMotionDetector(voxel_size=0.5, threshold=0.2)
    previous = np.array(
        [[2.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, -1.0, 0.0]],
        dtype=np.float32,
    )
    detector.detect(previous)

    # Platform moved +1 m in x; static world returns therefore appear -1 m in
    # the new sensor frame. The relative transform maps previous-frame points
    # into the current sensor frame.
    current = previous - np.array([1.0, 0.0, 0.0], dtype=np.float32)
    relative = np.eye(4, dtype=np.float64)
    relative[0, 3] = -1.0

    residual = detector.detect(current, ego_transform=relative)
    assert np.count_nonzero(residual) == 0


def test_uncompensated_motion_is_detected():
    detector = VoxelMotionDetector(voxel_size=0.5, threshold=0.2)
    previous = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    detector.detect(previous)
    current = np.array([[2.35, 0.0, 0.0]], dtype=np.float32)
    residual = detector.detect(current)
    assert residual.tolist() == [1.0]
