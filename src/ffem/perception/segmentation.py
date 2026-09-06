"""LiDAR semantic segmentation interfaces and range-image backend."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import numpy as np

@dataclass
class ProjectionConfig:
    height: int = 32
    width: int = 1024
    min_range: float = 1.0
    max_range: float = 80.0
    fov_up_deg: float = 10.0
    fov_down_deg: float = -30.0

class RangeImageProjector:
    def __init__(self, config: ProjectionConfig | None=None):
        self.cfg=config or ProjectionConfig()
    def project(self, points: np.ndarray, intensity: np.ndarray | None=None) -> dict[str,np.ndarray]:
        p=np.asarray(points,dtype=np.float32).reshape(-1,3); n=len(p)
        inten=np.zeros(n,dtype=np.float32) if intensity is None else np.asarray(intensity,dtype=np.float32)
        depth=np.linalg.norm(p,axis=1); yaw=np.arctan2(p[:,1],p[:,0]); pitch=np.arcsin(np.clip(p[:,2]/np.maximum(depth,1e-6),-1,1))
        col=((yaw+np.pi)/(2*np.pi)*self.cfg.width).astype(np.int32)%self.cfg.width
        vfov=np.deg2rad(self.cfg.fov_up_deg-self.cfg.fov_down_deg)
        valid=(depth>=self.cfg.min_range)&(depth<=self.cfg.max_range)&np.isfinite(p).all(axis=1)
        idx=np.flatnonzero(valid)
        ri=np.full((self.cfg.height,self.cfg.width),-1,dtype=np.int64)
        dimg=np.zeros((self.cfg.height,self.cfg.width),dtype=np.float32)
        iimg=np.zeros_like(dimg)
        if len(idx):
            row=np.clip(((np.deg2rad(self.cfg.fov_up_deg)-pitch[idx])/vfov*self.cfg.height).astype(np.int32),0,self.cfg.height-1)
            colv=col[idx]; flat=row*self.cfg.width+colv
            order=np.lexsort((depth[idx],flat)); sorted_flat=flat[order]
            first=np.empty(len(order),dtype=bool); first[0]=True; first[1:]=sorted_flat[1:]!=sorted_flat[:-1]
            chosen=idx[order[first]]; chosen_flat=sorted_flat[first]
            rr=chosen_flat//self.cfg.width; cc=chosen_flat%self.cfg.width
            ri[rr,cc]=chosen; dimg[rr,cc]=depth[chosen]; iimg[rr,cc]=inten[chosen]
        return {'depth':dimg,'intensity':iimg,'point_index':ri,'valid':valid}

class SemanticSegmenter:
    num_classes:int
    def predict(self,points:np.ndarray,intensity:np.ndarray|None=None): raise NotImplementedError

class NumpyFallbackSegmenter(SemanticSegmenter):
    """Non-neural fallback for smoke tests only; never use for final results."""
    def __init__(self,num_classes=7): self.num_classes=num_classes
    def predict(self,points,intensity=None):
        p=np.asarray(points); inten=np.zeros(len(p)) if intensity is None else np.asarray(intensity)
        labels=np.zeros(len(p),dtype=np.int64); labels[p[:,2]>.25]=2; labels[inten>.72]=1; labels[p[:,2]>.8]=4
        probs=np.full((len(p),self.num_classes),.02/max(self.num_classes-1,1),dtype=np.float32); probs[np.arange(len(p)),labels]=.88
        return labels,probs

class TorchRangeSegmenter(SemanticSegmenter):
    """Point-wise range-image model adapter. Requires torch and a trained checkpoint."""
    def __init__(self,checkpoint:str,projection:ProjectionConfig|None=None,num_classes:int=7,device:str='auto'):
        try:
            import torch; import torch.nn as nn
        except ImportError as exc:
            raise RuntimeError('Install torch to use TorchRangeSegmenter.') from exc
        self.torch=torch; self.projector=RangeImageProjector(projection); self.num_classes=num_classes
        self.device='cuda' if device=='auto' and torch.cuda.is_available() else device if device!='auto' else 'cpu'
        if self.device=='cpu':
            try: torch.set_num_threads(int(os.environ.get('FFEM_TORCH_THREADS','4')))
            except Exception: pass
        self.model=nn.Sequential(nn.Conv2d(2,32,3,padding=1),nn.ReLU(),nn.Conv2d(32,64,3,padding=1),nn.ReLU(),nn.Conv2d(64,num_classes,1)).to(self.device)
        try: state=torch.load(Path(checkpoint),map_location='cpu',weights_only=True)
        except TypeError: state=torch.load(Path(checkpoint),map_location='cpu')
        if isinstance(state,dict) and 'classes' in state and int(state['classes'])!=num_classes: raise ValueError(f'Checkpoint has {state["classes"]} classes but FFEM expects {num_classes}')
        if isinstance(state,dict) and 'class_names' in state and len(state['class_names'])!=num_classes: raise ValueError('Checkpoint class_names length does not match model output')
        weights=state.get('model',state) if isinstance(state,dict) else state
        self.model.load_state_dict(weights); self.model.eval()
    def predict(self,points:np.ndarray,intensity:np.ndarray|None=None):
        t=self.torch; img=self.projector.project(points,intensity)
        x=np.stack((img['depth']/self.projector.cfg.max_range,img['intensity']),axis=0)[None]
        xt=t.from_numpy(x).float().to(self.device)
        with t.inference_mode(): logits=self.model(xt)[0].cpu().numpy()
        pix=np.argmax(logits,axis=0); probs=np.exp(logits-logits.max(0,keepdims=True)); probs/=probs.sum(0,keepdims=True)+1e-8
        ri=img['point_index']; labels=np.zeros(len(points),dtype=np.int64); out=np.zeros((len(points),self.num_classes),dtype=np.float32); rows,cols=np.where(ri>=0); orig=ri[rows,cols]
        labels[orig]=pix[rows,cols]; out[orig]=probs[:,rows,cols].T
        return labels,out
