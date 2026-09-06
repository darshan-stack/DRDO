"""Scan-to-scan motion residuals and lightweight cluster tracking."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class Track:
    track_id: int
    center: np.ndarray
    velocity: np.ndarray
    size: np.ndarray
    age: int = 1
    missed: int = 0

class VoxelMotionDetector:
    """Fast voxel residual detector using sorted integer voxel keys."""
    def __init__(self, voxel_size: float=0.35, threshold: float=0.45):
        self.voxel_size=voxel_size
        self.threshold=threshold
        self.previous: dict[tuple[int,int,int],np.ndarray]={}
        self.previous_keys=np.empty((0,),dtype=[('x','i8'),('y','i8'),('z','i8')])
        self.previous_points=np.empty((0,3),dtype=np.float32)

    def _structured_keys(self,points:np.ndarray):
        q=np.floor(np.asarray(points,dtype=np.float32)/self.voxel_size).astype(np.int64)
        out=np.empty(len(q),dtype=[('x','i8'),('y','i8'),('z','i8')])
        out['x']=q[:,0]; out['y']=q[:,1]; out['z']=q[:,2]
        return out

    def detect(self,points:np.ndarray,ego_transform:np.ndarray|None=None)->np.ndarray:
        p=np.asarray(points,dtype=np.float32).reshape(-1,3)
        keys=self._structured_keys(p)
        out=np.zeros(len(p),dtype=np.float32)
        if len(self.previous_points):
            order=np.argsort(self.previous_keys,order=('x','y','z'))
            prev_keys=self.previous_keys[order]
            loc=np.searchsorted(prev_keys,keys)
            valid=loc<len(prev_keys)
            if valid.any():
                valid_idx=np.flatnonzero(valid)
                eq=prev_keys[loc[valid_idx]]==keys[valid_idx]
                valid_idx=valid_idx[eq]
                if len(valid_idx):
                    old=self.previous_points[order[loc[valid_idx]]]
                    out[valid_idx]=(np.linalg.norm(p[valid_idx]-old,axis=1)>self.threshold).astype(np.float32)
        self.previous_keys=keys
        self.previous_points=p.copy()
        return out

    def _key(self,x): return tuple(np.floor(np.asarray(x)/self.voxel_size).astype(int))
    @staticmethod
    def _transform(previous,points,matrix): return points

class CentroidTracker:
    def __init__(self,max_distance:float=3.0,max_missed:int=5): self.max_distance=max_distance; self.max_missed=max_missed; self.tracks={}; self.next_id=1
    def update(self,points:np.ndarray,motion:np.ndarray)->list[Track]:
        p=np.asarray(points); active=p[np.asarray(motion)>0.5]; clusters=self._clusters(active); used=set()
        for center,size in clusters:
            best=None; dist=self.max_distance
            for tid,t in self.tracks.items():
                d=float(np.linalg.norm(center-t.center))
                if d<dist and tid not in used: best,dist=tid,d
            if best is None:
                self.tracks[self.next_id]=Track(self.next_id,center,np.zeros(3),size); used.add(self.next_id); self.next_id+=1
            else:
                t=self.tracks[best]; t.velocity=center-t.center; t.center=center; t.size=size; t.age+=1; t.missed=0; used.add(best)
        for tid in list(self.tracks):
            if tid not in used: self.tracks[tid].missed+=1
            if self.tracks[tid].missed>self.max_missed: del self.tracks[tid]
        return list(self.tracks.values())
    @staticmethod
    def _clusters(points):
        if len(points)==0:return []
        bins={}
        for p in points: bins.setdefault((int(np.floor(p[0]/2)),int(np.floor(p[1]/2))),[]).append(p)
        return [(np.mean(v,axis=0),np.ptp(v,axis=0)+0.1) for v in bins.values() if len(v)>=3]
