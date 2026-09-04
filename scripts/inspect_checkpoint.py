#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('checkpoint'); args=ap.parse_args()
    try:
        import torch
    except ImportError as exc:
        raise SystemExit('PyTorch is required for checkpoint inspection') from exc
    path=Path(args.checkpoint)
    try:
        obj=torch.load(path,map_location='cpu',weights_only=True)
    except TypeError:
        obj=torch.load(path,map_location='cpu')
    print('type:', type(obj).__name__)
    if isinstance(obj,dict):
        print('keys:', sorted(obj.keys()))
        for k,v in obj.items():
            if hasattr(v,'shape'):
                print(f'{k}: shape={tuple(v.shape)} dtype={getattr(v,"dtype",None)}')
            elif isinstance(v,dict):
                print(f'{k}: {len(v)} entries')
            else:
                print(f'{k}: {v!r}')
        state=obj.get('model') or obj.get('state_dict')
        if isinstance(state,dict):
            print('state_dict tensors:', len(state))
            for k,v in list(state.items())[:20]: print(' ',k,tuple(v.shape))
    elif hasattr(obj,'state_dict'):
        print('module state tensors:',len(obj.state_dict()))
    else:
        print('unsupported checkpoint object for automatic loading')
if __name__=='__main__': main()
