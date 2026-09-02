"""SemanticKITTI binary scan and label adapter."""
from __future__ import annotations
from pathlib import Path
import numpy as np

DEFAULT_LABEL_MAP={0:0,1:1,10:2,11:2,13:2,15:2,16:2,18:2,20:2,30:3,31:3,32:3,40:1,44:1,48:3,49:3,50:3,51:3,52:3,60:1,70:1,71:1,72:1,80:4,81:4,99:6}
class SemanticKITTIDataset:
    def __init__(self, root: str, sequences: list[str], label_map=None):
        self.root=Path(root); self.sequences=[str(s).zfill(2) for s in sequences]; self.label_map=label_map or DEFAULT_LABEL_MAP; self.samples=[]
        for seq in self.sequences:
            vel=self.root/'sequences'/seq/'velodyne'; lab=self.root/'sequences'/seq/'labels'
            for scan in sorted(vel.glob('*.bin')):
                label=lab/(scan.stem+'.label'); self.samples.append((scan,label if label.exists() else None))
        if not self.samples: raise FileNotFoundError(f'No SemanticKITTI scans found under {self.root}')
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        scan,label=self.samples[index]; points=np.fromfile(scan,dtype=np.float32).reshape(-1,4); y=None
        if label is not None:
            raw=(np.fromfile(label,dtype=np.uint32)&0xFFFF).astype(np.int32); y=np.array([self.label_map.get(int(v),0) for v in raw],dtype=np.int64)
            if len(y)!=len(points): raise ValueError(f'Label/point mismatch for {scan}: {len(y)} vs {len(points)}')
        return points[:,:3],points[:,3],y,scan
    def split_by_sequence(self, val_sequences=('08',)):
        val=set(str(s).zfill(2) for s in val_sequences); train=[]; valid=[]
        for i,(p,l) in enumerate(self.samples): (valid if p.parts[-3] in val else train).append(i)
        return train,valid

def validate_dataset(root: str, sequences: list[str]) -> dict:
    ds=SemanticKITTIDataset(root,sequences); labeled=sum(l is not None for _,l in ds.samples); points=0
    for i in range(min(len(ds),100)): points += len(ds[i][0])
    return {'scans':len(ds),'labeled_scans':labeled,'sampled_points':points,'label_coverage':labeled/max(1,len(ds))}
