"""Adaptive FFEM mapping and perception pipeline."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import time
import numpy as np
from ffem.perception.motion import VoxelMotionDetector, CentroidTracker
from ffem.mapping.radial_resolution import RadialResolutionPolicy

@dataclass
class FFEMConfig:
    base_cell_size: float = 1.0
    finest_cell_size: float = 0.25
    max_level: int = 2
    refine_threshold: float = 0.60
    merge_threshold: float = 0.25
    dwell_frames: int = 5
    max_active_cells: int = 20000
    max_topology_changes: int = 32
    predictive_dilation_frames: int = 2
    semantic_weight: float = 0.30
    motion_weight: float = 0.30
    traversability_weight: float = 0.20
    geometry_weight: float = 0.15
    range_weight: float = 0.05
    num_classes: int = 7

@dataclass
class Cell:
    key: tuple[int,int,int]
    level: int = 0
    count: int = 0
    elevation: float = 0.0
    variance: float = 0.0
    semantic_probs: np.ndarray = field(default_factory=lambda: np.ones(7)/7)
    motion_probability: float = 0.0
    traversability: float = 0.0
    attention: float = 0.0
    quiet_frames: int = 0
    size_m: float = 1.0
    slices: list[dict[str,float]] = field(default_factory=list)
    @property
    def size(self) -> float:
        return self.size_m/(2**self.level)

class SyntheticLidar:
    def __init__(self, seed=7): self.rng=np.random.default_rng(seed)
    def frame(self,index,n=2200):
        theta=self.rng.uniform(-np.pi,np.pi,n); radius=self.rng.uniform(2,45,n); x,y=radius*np.cos(theta),radius*np.sin(theta)
        z=.10*np.sin(x/4)+.07*np.cos(y/3); cx,cy=8+.18*index,2+.05*np.sin(index/5); moving=((x-cx)**2+(y-cy)**2)<3.5; z[moving]+=1
        intensity=np.clip(.4+.3*np.sin(x)+.2*self.rng.normal(size=n),0,1)
        return np.column_stack((x,y,z)),intensity,moving

class MockPerception:
    def __init__(self,num_classes=4): self.num_classes=num_classes
    def infer(self,points,moving,intensity):
        labels=np.zeros(len(points),dtype=np.int64); labels[(points[:,2]>.25)&~moving]=1; labels[(intensity>.72)&~moving]=2; labels[moving]=3
        probs=np.full((len(points),self.num_classes),.04/max(1,self.num_classes-1),dtype=np.float32); probs[np.arange(len(points)),labels]=.96
        return probs,moving.astype(np.float32)

class AdaptiveElevationMap:
    def __init__(self,config):
        self.cfg=config; self.cells={}; self.radial=RadialResolutionPolicy(); self.events=[]
        self._limits=np.asarray([b.max_radius_m for b in self.radial.bands],dtype=np.float32)
        self._sizes=np.asarray([b.cell_size_m for b in self.radial.bands],dtype=np.float32)
    def _key(self,x,y,level=0):
        r=float(np.hypot(x,y)); b=int(np.clip(np.searchsorted(self._limits,r,side='left'),0,len(self._limits)-1)); s=self._sizes[b]/(2**level)
        return b,int(np.floor(x/s)),int(np.floor(y/s))
    def _cell(self,key,level=0):
        if key not in self.cells:
            self.cells[key]=Cell(key=key,level=level,size_m=float(self._sizes[key[0]]),semantic_probs=np.ones(self.cfg.num_classes)/self.cfg.num_classes)
        return self.cells[key]
    def _aggregate(self,inv,w,g): return np.bincount(inv,weights=np.asarray(w,dtype=np.float64),minlength=g)
    def update(self,points,semantic_probs,motion,frame):
        t0=time.perf_counter(); p=np.asarray(points,dtype=np.float32).reshape(-1,3); n=len(p)
        if n==0: return {'map_ms':(time.perf_counter()-t0)*1000,'active_cells':len(self.cells),'topology_changes':0}
        r=np.hypot(p[:,0],p[:,1]); bands=np.searchsorted(self._limits,r,side='left').astype(np.int16); bands=np.clip(bands,0,len(self._sizes)-1); sizes=self._sizes[bands]
        kd=np.empty(n,dtype=[('b','i2'),('x','i8'),('y','i8')]); kd['b']=bands; kd['x']=np.floor(p[:,0]/sizes); kd['y']=np.floor(p[:,1]/sizes)
        uk,inv=np.unique(kd,return_inverse=True); g=len(uk); counts=np.bincount(inv,minlength=g).astype(np.float64); z=p[:,2].astype(np.float64)
        sumz=self._aggregate(inv,z,g); meanz=sumz/np.maximum(counts,1); varz=self._aggregate(inv,(z-meanz[inv])**2,g)/np.maximum(counts,1); mm=self._aggregate(inv,np.asarray(motion).reshape(-1),g)/np.maximum(counts,1)
        probs=np.asarray(semantic_probs,dtype=np.float32)
        if probs.ndim!=2 or probs.shape[0]!=n: raise ValueError('semantic_probs must have shape [N, num_classes]')
        means=np.empty((g,self.cfg.num_classes),dtype=np.float64)
        for j in range(self.cfg.num_classes): means[:,j]=self._aggregate(inv,probs[:,j],g)/np.maximum(counts,1)
        for gi in range(g):
            key=(int(uk['b'][gi]),int(uk['x'][gi]),int(uk['y'][gi])); c=self._cell(key); old=c.elevation
            c.count+=int(counts[gi]); c.elevation=float(meanz[gi]); c.variance=float(varz[gi]); c.semantic_probs=.85*c.semantic_probs+.15*means[gi]; c.semantic_probs/=max(float(c.semantic_probs.sum()),1e-12); c.motion_probability=float(.8*c.motion_probability+.2*mm[gi])
            c.traversability=float(np.clip(3*np.sqrt(c.variance+1e-6)+abs(c.elevation-old),0,1)); ent=float(-np.sum(c.semantic_probs*np.log(c.semantic_probs+1e-8))/np.log(self.cfg.num_classes))
            c.attention=float(np.clip(self.cfg.semantic_weight*ent+self.cfg.motion_weight*c.motion_probability+self.cfg.traversability_weight*c.traversability+self.cfg.geometry_weight*min(1,4*c.variance)+self.cfg.range_weight*min(1,np.hypot(key[1]*c.size,key[2]*c.size)/50),0,1))
            if counts[gi]>=8 and c.variance>.04 and not c.slices:
                zz=z[inv==gi]; mid=float(np.mean(zz)); c.slices=[{'height':float(np.min(zz)),'support':float(np.sum(zz<mid)/len(zz))},{'height':float(np.max(zz)),'support':float(np.sum(zz>=mid)/len(zz))}]
        cells=list(self.cells.values()); changes=0
        refine=[c for c in cells if c.level<self.cfg.max_level and c.attention>=self.cfg.refine_threshold]
        refine.sort(key=lambda c:c.attention,reverse=True)
        for c in refine[:self.cfg.max_topology_changes]:
            old=c.level; c.level+=1; c.quiet_frames=0; self.events.append({'frame':frame,'cell':c.key,'old_level':old,'new_level':c.level,'reason':'attention','score':c.attention}); changes+=1
        if changes<self.cfg.max_topology_changes:
            merge=[]
            for c in cells:
                if c.attention<self.cfg.merge_threshold and c.level>0:
                    c.quiet_frames+=1
                    if c.quiet_frames>=self.cfg.dwell_frames: merge.append(c)
            merge.sort(key=lambda c:c.attention)
            for c in merge[:self.cfg.max_topology_changes-changes]:
                old=c.level; c.level-=1; c.quiet_frames=0; self.events.append({'frame':frame,'cell':c.key,'old_level':old,'new_level':c.level,'reason':'hysteresis','score':c.attention}); changes+=1
        if len(self.cells)>self.cfg.max_active_cells:
            for key in list(self.cells)[:len(self.cells)-self.cfg.max_active_cells]: del self.cells[key]
        return {'map_ms':(time.perf_counter()-t0)*1000,'active_cells':len(self.cells),'topology_changes':changes}
    def arrays(self):
        if not self.cells: return np.empty((0,3)),np.empty((0,3),dtype=np.uint8),np.empty((0,))
        cells=tuple(self.cells.values()); n=len(cells); pts=np.empty((n,3),dtype=np.float32); colors=np.empty((n,3),dtype=np.uint8); levels=np.empty(n,dtype=np.int8)
        pal=np.array([[90,90,90],[70,140,220],[70,210,100],[180,120,60],[230,70,60],[220,80,180],[245,190,40]],dtype=np.uint8)
        for i,c in enumerate(cells): s=c.size; pts[i]=[(c.key[1]+.5)*s,(c.key[2]+.5)*s,c.elevation]; colors[i]=pal[int(np.argmax(c.semantic_probs))]; levels[i]=c.level
        return pts,colors,levels
    def diagnostics(self):
        """Return per-cell arrays for RViz/dashboard diagnostic layers."""
        if not self.cells:
            empty3=np.empty((0,3),dtype=np.float32); empty=np.empty((0,),dtype=np.float32); empty_i=np.empty((0,),dtype=np.int32)
            return empty3, empty, empty, empty, empty_i
        cells=tuple(self.cells.values()); n=len(cells)
        pts=np.empty((n,3),dtype=np.float32); traversability=np.empty(n,dtype=np.float32)
        uncertainty=np.empty(n,dtype=np.float32); attention=np.empty(n,dtype=np.float32); levels=np.empty(n,dtype=np.int32)
        for i,c in enumerate(cells):
            s=c.size; pts[i]=[(c.key[1]+0.5)*s,(c.key[2]+0.5)*s,c.elevation]
            probs=np.asarray(c.semantic_probs,dtype=np.float64); probs=probs/max(float(probs.sum()),1e-12)
            uncertainty[i]=float(np.clip(-np.sum(probs*np.log(probs+1e-8))/np.log(self.cfg.num_classes),0,1))
            traversability[i]=float(c.traversability); attention[i]=float(c.attention); levels[i]=int(c.level)
        return pts,traversability,uncertainty,attention,levels

class FFEMPipeline:
    def __init__(self,config=None,seed=7,segmenter=None,motion_detector=None,tracker=None):
        self.config=config or FFEMConfig(); self.sensor=SyntheticLidar(seed); self.perception=MockPerception(self.config.num_classes); self.segmenter=segmenter; self.motion_detector=motion_detector or VoxelMotionDetector(); self.tracker=tracker or CentroidTracker(); self.mapping=AdaptiveElevationMap(self.config); self.history=[]
    def process_points(self,points,intensity=None,motion=None,frame=0):
        t0=time.perf_counter(); points=np.asarray(points,dtype=np.float32).reshape(-1,3); intensity=np.zeros(len(points),dtype=np.float32) if intensity is None else np.asarray(intensity,dtype=np.float32); motion=self.motion_detector.detect(points) if motion is None else np.asarray(motion,dtype=np.float32)
        if self.segmenter is not None: _,probs=self.segmenter.predict(points,intensity); inferred_motion=motion
        else: probs,inferred_motion=self.perception.infer(points,motion>.5,intensity)
        motion=np.maximum(motion,inferred_motion); stats=self.mapping.update(points,probs,motion,frame); tracks=self.tracker.update(points,motion); stats.update({'frame':frame,'total_ms':(time.perf_counter()-t0)*1000,'points':len(points),'moving_points':int((motion>.5).sum()),'tracks':len(tracks)}); self.history.append(stats)
        return {'points':points,'intensity':intensity,'moving':motion>.5,'motion_probability':motion,'semantic_probs':probs,'tracks':tracks,'stats':stats}
    def step(self,frame):
        p,i,m=self.sensor.frame(frame); return self.process_points(p,i,m.astype(np.float32),frame)
