#!/usr/bin/env python3
"""Benchmark FFEM replay and emit JSON metrics for baseline comparison."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from ffem.pipeline import FFEMPipeline, FFEMConfig
from ffem.perception.factory import build_segmenter
from ffem.evaluation.metrics import runtime_summary

def run(mode, frames, backend, checkpoint):
    cfg = FFEMConfig()
    if mode == "uniform_fine":
        cfg.base_cell_size = 0.05
        cfg.finest_cell_size = 0.05
        cfg.max_level = 0
        cfg.max_active_cells = 200000
    elif mode == "uniform_coarse":
        cfg.base_cell_size = 0.50
        cfg.finest_cell_size = 0.50
        cfg.max_level = 0

    seg, _ = build_segmenter(backend, checkpoint, 7)
    pipe = FFEMPipeline(cfg, segmenter=seg)

    # Uniform baselines must actually use a uniform radial policy; changing
    # FFEMConfig alone is insufficient because the production map owns the
    # radial policy independently.
    if mode != "ffem":
        from ffem.mapping.radial_resolution import RadialResolutionPolicy, ResolutionBand
        size = 0.05 if mode == "uniform_fine" else 0.50
        pipe.mapping.radial = RadialResolutionPolicy(
            (ResolutionBand(100.0, size, "uniform"),)
        )
        pipe.mapping._limits = np.asarray([100.0], dtype=np.float32)
        pipe.mapping._sizes = np.asarray([size], dtype=np.float32)

    for f in range(frames):
        pipe.step(f)
    out=runtime_summary(pipe.history); out['mode']=mode; out['mean_points']=float(np.mean([h['points'] for h in pipe.history])); out['mean_moving_points']=float(np.mean([h['moving_points'] for h in pipe.history])); return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--frames',type=int,default=100); ap.add_argument('--backend',choices=['auto','torch_range','fallback'],default='auto'); ap.add_argument('--checkpoint',default=''); ap.add_argument('--output',default='outputs/replay_benchmark.json'); args=ap.parse_args()
    results=[run(m,args.frames,args.backend,args.checkpoint) for m in ('uniform_fine','uniform_coarse','ffem')]
    report={'frames':args.frames,'results':results}; Path(args.output).parent.mkdir(parents=True,exist_ok=True); Path(args.output).write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
