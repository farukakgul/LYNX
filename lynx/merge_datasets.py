#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Merge multiple shard NPZ datasets into a single NPZ")
    ap.add_argument("--inputs", type=Path, nargs="+", required=True, help="List of *.npz shards to merge")
    ap.add_argument("--out", type=Path, required=True, help="Output npz path")
    return ap.parse_args()


def main():
    args = parse_args()
    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    for p in args.inputs:
        d = np.load(p)
        Xs.append(d["X"].astype(np.float32))
        ys.append(d["y"].astype(np.int64))
        print(f"Loaded {p}: X={Xs[-1].shape}, y={ys[-1].shape}")
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, X=X, y=y)
    print(f"Saved merged dataset: {args.out} (X={X.shape}, y={y.shape})")


if __name__ == "__main__":
    main()

