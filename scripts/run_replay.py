#!/usr/bin/env python3
"""Run the dependency-light FFEM demo replay.

This first milestone intentionally uses deterministic mock observations. Replace
these adapters with real LiDAR, semantic, and MOS backends in later stages.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

try:
    import rerun as rr
except ImportError:  # pragma: no cover
    rr = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', type=int, default=100)
    parser.add_argument('--recording', default='outputs/ffem_demo.rrd')
    parser.add_argument('--no-rerun', action='store_true')
    args = parser.parse_args()

    use_rr = rr is not None and not args.no_rerun
    if use_rr:
        rr.init('ffem-lidar-mapping', spawn=True)
        rr.set_time('frame', sequence=0)

    rng = np.random.default_rng(7)
    for frame_idx in range(args.frames):
        theta = rng.uniform(-np.pi, np.pi, 1500)
        radius = rng.uniform(2.0, 45.0, 1500)
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        z = 0.04 * np.sin(x / 4.0) + 0.03 * np.cos(y / 3.0)
        points = np.column_stack((x, y, z))
        moving = ((x - 8.0 - 0.12 * frame_idx) ** 2 + (y - 2.0) ** 2) < 3.0
        z[moving] += 1.0
        colours = np.zeros((len(points), 3), dtype=np.uint8)
        colours[:] = [90, 140, 220]
        colours[moving] = [230, 70, 60]

        if use_rr:
            rr.set_time('frame', sequence=frame_idx)
            rr.log('world/lidar/raw', rr.Points3D(points, colors=colours))
            rr.log('world/dynamics/moving_points', rr.Points3D(points[moving], colors=[240, 60, 50]))
            rr.log('world/robot/trajectory', rr.LineStrips3D([[[0, 0, 0], [0.12 * frame_idx, 0, 0]]]))
            rr.log('metrics/latency/pipeline_ms', rr.Scalars([3.0 + 0.02 * int(moving.sum())]))
            rr.log('metrics/memory/active_cells', rr.Scalars([2500 + 4 * int(moving.sum())]))

    if use_rr:
        Path(args.recording).parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.recording)
        print(f'Saved Rerun recording to {args.recording}')
    else:
        print(f'Completed {args.frames} frames without Rerun logging')


if __name__ == '__main__':
    main()
