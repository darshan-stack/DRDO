#!/usr/bin/env python3
"""Run the end-to-end FFEM synthetic real-time demo."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ffem.pipeline import FFEMPipeline, FFEMConfig
from ffem.perception.segmentation import NumpyFallbackSegmenter, TorchRangeSegmenter, ProjectionConfig

try:
    import rerun as rr
except ImportError:
    rr = None


def log_frame(result: dict, pipeline: FFEMPipeline, frame: int) -> None:
    if rr is None: return
    points, moving, probs, stats = result['points'], result['moving'], result['semantic_probs'], result['stats']
    map_points, map_colors, levels = pipeline.mapping.arrays()
    rr.set_time('frame', sequence=frame)
    colors = np.zeros((len(points), 3), dtype=np.uint8); colors[:] = [80, 140, 220]; colors[moving] = [235, 65, 55]
    rr.log('world/lidar/raw', rr.Points3D(points, colors=colors))
    rr.log('world/dynamics/moving_points', rr.Points3D(points[moving], colors=[245, 60, 50]))
    rr.log('world/map/elevation', rr.Points3D(map_points, colors=map_colors))
    if len(map_points):
        rr.log('world/map/adaptive_cells', rr.Points3D(map_points, radii=0.04 + 0.03 * levels, colors=map_colors))
    if pipeline.mapping.events:
        ev = pipeline.mapping.events[-min(20, len(pipeline.mapping.events)):]
        event_pts = np.array([[e['cell'][0], e['cell'][1], 0.04] for e in ev], dtype=np.float32)
        rr.log('world/adaptation/refinement_events', rr.Points3D(event_pts, colors=[245, 190, 40], radii=0.08))
    rr.log('world/robot/trajectory', rr.LineStrips3D([[[0.12 * frame, 0, 0], [0.12 * frame + 0.1, 0, 0]]]))
    rr.log('metrics/latency/total_ms', rr.Scalars([stats['total_ms']]))
    rr.log('metrics/latency/map_ms', rr.Scalars([stats['map_ms']]))
    rr.log('metrics/memory/active_cells', rr.Scalars([stats['active_cells']]))
    rr.log('metrics/topology_changes', rr.Scalars([stats['topology_changes']]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=300)
    ap.add_argument('--recording', default='outputs/ffem_demo.rrd')
    ap.add_argument('--no-rerun', action='store_true')
    ap.add_argument('--seed', type=int, default=7)
    ap.add_argument('--backend', choices=['fallback','torch_range'], default='fallback')
    ap.add_argument('--checkpoint', default='')
    args = ap.parse_args()
    use_rr = rr is not None and not args.no_rerun
    if use_rr: rr.init('ffem-lidar-mapping', spawn=True); rr.set_time('frame', sequence=0)
    cfg = FFEMConfig()
    if args.backend == 'torch_range':
        if not args.checkpoint: raise SystemExit('--checkpoint is required with --backend torch_range')
        segmenter = TorchRangeSegmenter(args.checkpoint, ProjectionConfig(height=32, width=1024, max_range=80.0), num_classes=cfg.num_classes)
    else:
        segmenter = NumpyFallbackSegmenter(cfg.num_classes)
    pipeline = FFEMPipeline(cfg, seed=args.seed, segmenter=segmenter)
    for frame in range(args.frames):
        result = pipeline.step(frame)
        log_frame(result, pipeline, frame)
        if frame % 25 == 0: print(f"frame={frame:04d} total_ms={result['stats']['total_ms']:.2f} cells={result['stats']['active_cells']} changes={result['stats']['topology_changes']}")
    if use_rr:
        Path(args.recording).parent.mkdir(parents=True, exist_ok=True); rr.save(args.recording); print(f'Saved Rerun recording to {args.recording}')
    else: print(f'Completed {args.frames} frames without Rerun logging')

if __name__ == '__main__': main()
