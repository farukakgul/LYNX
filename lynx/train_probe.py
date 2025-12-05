#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# Ensure repo root on sys.path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lynx.utils import save_args_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Train a two-layer MLP probe for early-exit correctness detection")
    ap.add_argument("--train-npz", type=Path, required=True)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/probe"))
    ap.add_argument("--mlp-hidden", type=str, default="256,64", help="Comma-separated hidden sizes, e.g. 256,64")
    ap.add_argument("--mlp-max-iter", type=int, default=500)
    return ap.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save CLI configuration alongside probe artifacts.
    try:
        save_args_json(args, args.out_dir / "train_probe_config.json")
    except Exception:
        pass
    print(f"[probe] Loading dataset: {args.train_npz}")
    data = np.load(args.train_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    print(f"[probe] Dataset: X={X.shape}, y={y.shape}; pos={int((y==1).sum())}, neg={int((y==0).sum())}")
    Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=args.val_split, random_state=args.seed, stratify=y)
    # Choose MLP hidden sizes.
    # Keep the original default (256,64) for most models, but bump capacity slightly
    # for large feature dims such as Qwen3-8B (e.g., global_math_qwen3_8b_* NPZs).
    hidden_spec = args.mlp_hidden.strip()
    if hidden_spec == "256,64":
        npz_path_lower = str(args.train_npz).lower()
        # Heuristic: any NPZ path mentioning qwen3-8b or qwen3_8b uses a larger MLP.
        if "qwen3-8b" in npz_path_lower or "qwen3_8b" in npz_path_lower:
            hidden = (256, 64)
        else:
            hidden = (256, 64)
    else:
        hidden = tuple(int(x) for x in hidden_spec.split(",") if x.strip()) or (256, 64)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MLPClassifier(hidden_layer_sizes=hidden, activation="relu", solver="adam",
                               random_state=args.seed, max_iter=args.mlp_max_iter, early_stopping=True,
                               n_iter_no_change=10, validation_fraction=max(0.1, args.val_split))),
    ])
    # Class-imbalance aware training via sample weights
    n_pos = max(1, int((ytr == 1).sum()))
    n_neg = max(1, int((ytr == 0).sum()))
    n_all = len(ytr)
    w_pos = n_all / (2.0 * n_pos)
    w_neg = n_all / (2.0 * n_neg)
    sample_weight = np.where(ytr == 1, w_pos, w_neg).astype(np.float32)
    t0 = time.time(); pipe.fit(Xtr, ytr, clf__sample_weight=sample_weight); t1 = time.time()
    p = pipe.predict_proba(Xval)[:, 1]
    yhat = (p >= 0.5).astype(np.int64)
    acc = accuracy_score(yval, yhat)
    try:
        auc = roc_auc_score(yval, p)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(yval, yhat, labels=[0,1])
    print(f"[probe] val: acc={acc:.3f}, auc={auc:.3f}, cm={cm.tolist()}")
    print(f"[probe] report:\n{classification_report(yval, yhat, digits=3)}")

    out_pkl = args.out_dir / "probe.pkl"
    with out_pkl.open("wb") as fh:
        pickle.dump(pipe, fh)
    print(f"Saved: {out_pkl}")

    metrics = {
        "train_npz": str(args.train_npz),
        "val_split": float(args.val_split),
        "hidden": list(hidden),
        "max_iter": int(args.mlp_max_iter),
        "val_acc": float(acc),
        "val_auc": float(auc),
        "cm": cm.tolist(),
        "train_seconds": float(t1 - t0),
        "n_train": int(len(ytr)),
        "n_val": int(len(yval)),
        "train_pos": int(n_pos),
        "train_neg": int(n_neg),
        "w_pos": float(w_pos),
        "w_neg": float(w_neg),
    }
    (args.out_dir / "probe_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
