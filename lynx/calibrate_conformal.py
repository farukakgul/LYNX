#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

# Ensure repo root on sys.path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lynx.utils import conformal_quantile, save_args_json


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Calibrate split conformal thresholds for binary probe at 1-delta coverage")
    ap.add_argument("--probe-path", type=Path, required=True)
    ap.add_argument("--calib-npz", type=Path, required=True)
    ap.add_argument("--delta", type=float, default=0.1, help="Miscoverage level; 0.1 -> 90% coverage")
    ap.add_argument("--out", type=Path, default=Path("outputs/conformal_calib.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    with args.probe_path.open("rb") as fh:
        probe = pickle.load(fh)
    data = np.load(args.calib_npz)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.int64)

    proba_pos = probe.predict_proba(X)[:, 1]
    proba_neg = 1.0 - proba_pos

    # Nonconformity scores s_c(x) = 1 - p(c|x)
    s_pos = 1.0 - proba_pos[y == 1]
    s_neg = 1.0 - proba_neg[y == 0]

    q_pos = conformal_quantile(s_pos, args.delta) if len(s_pos) else float("inf")
    q_neg = conformal_quantile(s_neg, args.delta) if len(s_neg) else float("inf")

    out = {
        "delta": float(args.delta),
        "q_pos": float(q_pos),
        "q_neg": float(q_neg),
        "n_calib": int(len(y)),
        "n_pos": int((y == 1).sum()),
        "n_neg": int((y == 0).sum()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Saved: {args.out}")

    # Save CLI configuration next to the calibration file.
    try:
        cfg_path = args.out.parent / "calibrate_conformal_config.json"
        save_args_json(args, cfg_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
