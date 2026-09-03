"""SemanticPOSS loader and label remapping for FFEM.

SemanticPOSS uses KITTI-style float32 x,y,z,remission scans and uint32 labels;
the lower 16 bits are semantic IDs and upper 16 bits are instance IDs.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np

RAW_LABEL_NAMES={0:'unlabeled',4:'1 person',5:'2+ person',6:'rider',7:'car',8:'trunk',9:'plants',10:'traffic sign 1',11:'traffic sign 2',12:'traffic sign 3',13:'pole',14:'trashcan',15:'building',16:'cone/stone',17:'fence',21:'bike',22:'ground'}
# FFEM compact classes: 0 unknown, 1 ground, 2 vegetation, 3 structure,
# 4 vehicle/bike, 5 person/rider, 6 other obstacle.
DEFAULT_LABEL_MAP={0:0,1:0,2:0,3:0,4:5,5:5,6:5,7:4,8:6,9:2,10:6,11:6,12:6,13:6,14:6,15:3,16:6,17:6,18:0,19:0,20:0,21:4,22:1}
CLASS_NAMES=('unknown','ground','vegetation','structure','vehicle','person','obstacle')

class SemanticPOSSDataset:
    def __init__(self, root: str, sequences: list[str] | None=None, label_map=None, require_labels=True):
        self.root=Path(root).expanduser(); self.label_map=label_map or DEFAULT_LABEL_MAP
        available=sorted(p.name for p in (self.root/'sequences').glob('*') if p.is_dir()) if (self.root/'sequences').exists() else []
        self.sequences=[str(s).zfill(2) for s in sequences] if sequences else available
        self.samples=[]
        for seq in self.sequences:
            vel=self.root/'sequences'/seq/'velodyne'; lab=self.root/'sequences'/seq/'labels'
            if not vel.exists(): continue
            for scan in sorted(vel.glob('*.bin')):
                label=lab/(scan.stem+'.label')
                if require_labels and not label.exists(): continue
                self.samples.append((scan,label if label.exists() else None,seq))
        if not self.samples: raise FileNotFoundError(f'No SemanticPOSS labeled scans found under {self.root}')
    @property
    def available_sequences(self): return sorted(set(s for _,_,s in self.samples))
    def __len__(self): return len(self.samples)
    def __getitem__(self,index):
        scan,label,seq=self.samples[index]; points=np.fromfile(scan,dtype=np.float32)
        if points.size%4: raise ValueError(f'Invalid point record length in {scan}')
        points=points.reshape(-1,4); y=None
        if label is not None:
            raw=(np.fromfile(label,dtype=np.uint32)&0xffff).astype(np.int32)
            if len(raw)!=len(points): raise ValueError(f'Label/point mismatch for {scan}: {len(raw)} vs {len(points)}')
            y=np.array([self.label_map.get(int(v),0) for v in raw],dtype=np.int64)
        return points[:,:3],points[:,3],y,scan

def validate_dataset(root: str, sequences: list[str] | None=None, sample_scans=100):
    ds=SemanticPOSSDataset(root,sequences); points=0; classes=np.zeros(len(CLASS_NAMES),dtype=np.int64)
    for i in range(min(len(ds),sample_scans)):
        p,_,y,_=ds[i]; points+=len(p)
        if y is not None: classes+=np.bincount(y,minlength=len(CLASS_NAMES))
    return {'scans':len(ds),'sequences':ds.available_sequences,'sampled_points':int(points),'class_counts':classes.tolist(),'class_names':list(CLASS_NAMES),'label_coverage':1.0}

def write_sequence_splits(root: str, output_dir: str='configs'):
    root=Path(root); seqs=sorted(p.name for p in (root/'sequences').glob('*') if p.is_dir())
    if len(seqs)<3: raise ValueError('Need at least three SemanticPOSS sequences for train/val/test splits')
    n=len(seqs); a=max(1,int(.7*n)); b=max(a+1,int(.85*n)); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    for name,items in [('semanticposs_train.txt',seqs[:a]),('semanticposs_val.txt',seqs[a:b]),('semanticposs_test.txt',seqs[b:])]: (out/name).write_text('\n'.join(items)+'\n')
    return {'train':seqs[:a],'val':seqs[a:b],'test':seqs[b:]}
