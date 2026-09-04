"""Checkpoint discovery and perception-backend construction."""
from __future__ import annotations
import os
from pathlib import Path
from .segmentation import NumpyFallbackSegmenter, TorchRangeSegmenter, ProjectionConfig

def discover_checkpoint(explicit: str = '', search_root: str = 'models/checkpoints') -> str | None:
    if explicit:
        p=Path(explicit).expanduser()
        if not p.exists(): raise FileNotFoundError(f'Checkpoint does not exist: {p}')
        return str(p)
    env=os.environ.get('FFEM_CHECKPOINT','')
    if env and Path(env).expanduser().exists(): return str(Path(env).expanduser())
    candidates=sorted(Path(search_root).glob('*.pt')) if Path(search_root).exists() else []
    return str(candidates[0]) if candidates else None

def build_segmenter(backend: str='auto', checkpoint: str='', num_classes: int=7, height: int=32, width: int=1024, max_range: float=80.0):
    found=discover_checkpoint(checkpoint)
    selected=backend
    if backend=='auto': selected='torch_range' if found else 'fallback'
    if selected=='torch_range':
        if not found: raise FileNotFoundError('No .pt checkpoint found. Put it in models/checkpoints/ or set FFEM_CHECKPOINT.')
        return TorchRangeSegmenter(found,ProjectionConfig(height=height,width=width,max_range=max_range),num_classes=num_classes), found
    if selected=='fallback': return NumpyFallbackSegmenter(num_classes), None
    raise ValueError(f'Unknown perception backend: {backend}')
