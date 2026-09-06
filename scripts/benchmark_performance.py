#!/usr/bin/env python3
"""Benchmark FFEM processing latency on captured LiDAR frames.

Uses the exact frames produced by evaluate_memory_savings.py. Reports mean,
median, p95 latency, estimated FPS, and map-update latency so performance claims
are backed by repeatable measurements rather than a single frame.
"""
from __future__ import annotations
import argparse, json, statistics, time
from pathlib import Path
import numpy as np

from ffem.perception.factory import build_segmenter
from ffem.pipeline import FFEMConfig, FFEMPipeline


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--frames-file',default='outputs/memory_experiment_frames.npz')
    ap.add_argument('--checkpoint',default='models/checkpoints/semanticposs_range_model.pt')
    ap.add_argument('--frames',type=int,default=100)
    ap.add_argument('--max-points',type=int,default=12000)
    ap.add_argument('--warmup',type=int,default=5)
    ap.add_argument('--output',default='outputs/performance_benchmark.json')
    args=ap.parse_args()

    data=np.load(args.frames_file,allow_pickle=True)
    points=list(data['points']); intensities=list(data['intensity'])
    count=min(args.frames,len(points))
    segmenter,selected=build_segmenter('torch_range',args.checkpoint,7)
    pipeline=FFEMPipeline(config=FFEMConfig(max_active_cells=20000),segmenter=segmenter)

    timings=[]; map_timings=[]; point_counts=[]
    for i in range(min(args.warmup,count)):
        p=np.asarray(points[i],dtype=np.float32)
        it=np.asarray(intensities[i],dtype=np.float32)
        if len(p)>args.max_points:
            idx=np.linspace(0,len(p)-1,args.max_points,dtype=int); p=p[idx]; it=it[idx]
        pipeline.process_points(p,it,frame=i)

    for i in range(count):
        p=np.asarray(points[i],dtype=np.float32); it=np.asarray(intensities[i],dtype=np.float32)
        if len(p)>args.max_points:
            idx=np.linspace(0,len(p)-1,args.max_points,dtype=int); p=p[idx]; it=it[idx]
        t=time.perf_counter(); out=pipeline.process_points(p,it,frame=i+1); elapsed=(time.perf_counter()-t)*1000
        timings.append(elapsed); map_timings.append(float(out['stats']['map_ms'])); point_counts.append(len(p))

    mean_ms=statistics.fmean(timings); median_ms=statistics.median(timings); p95_ms=float(np.percentile(timings,95)); fps=1000/max(mean_ms,1e-9)
    report={
        'frames':count,'warmup_frames':min(args.warmup,count),'checkpoint':selected,
        'max_points_per_frame':args.max_points,'mean_points':statistics.fmean(point_counts),
        'mean_total_ms':mean_ms,'median_total_ms':median_ms,'p95_total_ms':p95_ms,
        'mean_map_ms':statistics.fmean(map_timings),'p95_map_ms':float(np.percentile(map_timings,95)),
        'estimated_fps':fps,
        'fps_definition':'1000 / mean end-to-end processing milliseconds',
        'note':'Benchmark uses the exact captured frames from the controlled memory experiment; FPS is measured on this workstation and is not a certified real-time guarantee.'
    }
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ('frames','mean_points','mean_total_ms','median_total_ms','p95_total_ms','mean_map_ms','estimated_fps')},indent=2))
    print(f'saved {out}')

if __name__=='__main__': main()
