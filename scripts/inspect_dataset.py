#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from ffem.io.semantic_poss import validate_dataset, write_sequence_splits

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data-root',required=True); ap.add_argument('--write-splits',action='store_true'); ap.add_argument('--output-dir',default='configs'); args=ap.parse_args()
    report=validate_dataset(args.data_root)
    print(json.dumps(report,indent=2))
    if args.write_splits: print(json.dumps(write_sequence_splits(args.data_root,args.output_dir),indent=2))
if __name__=='__main__': main()
