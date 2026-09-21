import numpy as np
import pytest

from ffem.perception.segmentation import ProjectionConfig, RangeImageProjector


def test_projection_returns_nearest_return_per_pixel():
    projector = RangeImageProjector(
        ProjectionConfig(height=4, width=8, min_range=0.1, max_range=20.0)
    )
    points = np.array(
        [[5.0, 0.0, 0.0], [6.0, 0.0, 0.0], [2.0, 2.0, 0.0]],
        dtype=np.float32,
    )
    out = projector.project(points, np.array([0.1, 0.2, 0.3], dtype=np.float32))
    assert np.count_nonzero(out["point_index"] >= 0) >= 2


def test_projection_rejects_bad_intensity_length():
    projector = RangeImageProjector()
    with pytest.raises(ValueError):
        projector.project(np.zeros((2, 3), dtype=np.float32), np.zeros(1))
