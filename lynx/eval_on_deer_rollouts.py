#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

# Ensure LYNX repo root is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lynx.probing_utils import (
    load_llm,
    forward_hidden_and_logits,
    find_pattern_positions_by_tokens,
)
from lynx.utils import (
    build_event_features,
    make_exit_prompt_prefix,
    generate_from_prefix,
    predictive_set_binary,
    is_correct_answer_with_policy,
    get_exit_cue,
    save_args_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Probe+conformal early exit on DEER rollouts (hmm/wait/alternatively events), DEER-format export")
    ap.add_argument(
        "--deer-rollout-path",
        type=Path,
        required=True,
        help="JSONL produced by a DEER-style generator (question/generated_responses/gold_answer)",
    )
    ap.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="HF model id (same one used to generate rollouts)",
    )
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")

    ap.add_argument("--probe-path", type=Path, required=True)
    ap.add_argument("--conformal-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)

    ap.add_argument("--max-examples", type=int, default=None, help="Process only first N rows of rollout")
    ap.add_argument("--exit-max-new-tokens", type=int, default=64)
    ap.add_argument("--exit-temperature", type=float, default=0.0)
    ap.add_argument("--exit-top-p", type=float, default=1.0)
    ap.add_argument("--exit-cue-variant", choices=["paper", "answer_tag"], default="paper")
    # For in-script reporting; when using DEER's check.py you usually want --eval-policy deer
    ap.add_argument("--eval-policy", choices=["deer", "strict", "zst"], default="deer")
    ap.add_argument("--verbose", action="store_true")
    return ap.parse_args()


def load_deer_rollouts(path: Path, limit: int | None = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def build_deer_prompt(tok, question: str) -> str:
    """
    Rebuild a DEER-style prompt:
    system: "Please reason step by step, and put your final answer within \\boxed{}."
    user:   <question>
    """
    sys_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

    # dtype fallback
    if args.dtype == "bfloat16" and torch.cuda.is_available():
        try:
            if not torch.cuda.is_bf16_supported():
                print("[warn] bf16 not supported; using float16")
                args.dtype = "float16"
        except Exception:
            args.dtype = "float16"

    # Save config next to outputs (best-effort)
    try:
        save_args_json(args, args.out_dir / "eval_on_deer_rollouts_config.json")
    except Exception:
        pass

    # load model/tokenizer
    model, tok = load_llm(
        args.model_id,
        device_map=args.device_map,
        torch_dtype=args.dtype,
    )
    llm_device = str(next(model.parameters()).device)
    print(f"[info] model on {llm_device}")

    # load probe + conformal
    import pickle

    with args.probe_path.open("rb") as fh:
        probe = pickle.load(fh)
    conf = json.loads(args.conformal_path.read_text())
    q_pos = float(conf.get("q_pos", float("inf")))
    q_neg = float(conf.get("q_neg", float("inf")))

    # load DEER rollouts
    rows = load_deer_rollouts(args.deer_rollout_path, args.max_examples)
    print(f"[info] loaded {len(rows)} rows from {args.deer_rollout_path}")

    # Decide feature layers based on model depth / id (mirror early_exit_eval.py):
    # - QwQ-32B (64 layers): (16,32,48,last)
    # - Qwen3-8B (36 layers): (12,18,24,last)
    # - Nemotron 8B (32 layers): (8,16,24,last)
    # - otherwise: (9,16,last).
    try:
        n_layers = int(getattr(model.config, "num_hidden_layers", 0))
    except Exception:
        n_layers = 0
    model_id_lower = str(args.model_id).lower()
    if "qwq-32b" in model_id_lower or n_layers == 64:
        feature_layers = (16, 32, 48)
    elif "qwen3-8b" in model_id_lower or n_layers == 36:
        feature_layers = (12, 18, 24)
    elif "nemotron" in model_id_lower or n_layers == 32:
        feature_layers = (8, 16, 24)
    else:
        feature_layers = (9, 16)

    # metrics
    rollouts_attempted = 0
    baseline_correct = 0
    baseline_tokens = 0
    early_correct = 0
    early_tokens = 0
    exited_count = 0

    accepted_correct = 0
    accepted_tokens = 0
    accepted_baseline_correct = 0
    help_count = 0
    harm_count = 0
    unchanged_correct = 0
    unchanged_incorrect = 0

    meta_rows: List[Dict[str, Any]] = []
    early_text_map: Dict[int, str] = {}

    t0 = time.time()

    for ridx, row in enumerate(tqdm(rows, desc="probe-on-deer (hmm/wait/alternatively)")):
        question = row.get("question") or row.get("problem") or ""
        gold = row.get("gold_answer") or row.get("answer") or ""
        gen_list = row.get("generated_responses") or row.get("generated_response") or []
        if isinstance(gen_list, list):
            base_text = gen_list[0] if gen_list else ""
        else:
            base_text = str(gen_list)

        # 1) rebuild DEER prompt
        prompt_text = build_deer_prompt(tok, question)

        # 2) re-encode prompt + assistant text
        prompt_ids = tok(prompt_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        full_text = prompt_text + base_text
        seq_ids = tok(full_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
        enc_len = len(prompt_ids)
        new_len = len(seq_ids) - enc_len

        # 3) baseline correctness for reporting
        base_label = is_correct_answer_with_policy(base_text, gold, policy=args.eval_policy)
        baseline_correct += base_label
        baseline_tokens += max(0, new_len)
        rollouts_attempted += 1

        # 4) forward to get hidden states
        _, hidden_states = forward_hidden_and_logits(model, seq_ids)

        # 5) find "hmm"/"wait"/"alternatively" events in the assistant part
        assistant_ids = seq_ids[enc_len:]
        pos_map = find_pattern_positions_by_tokens(assistant_ids, tok, patterns=("hmm", "wait", "alternatively"))
        events = sorted(
            [(p, "hmm") for p in pos_map.get("hmm", [])]
            + [(p, "wait") for p in pos_map.get("wait", [])]
            + [(p, "alternatively") for p in pos_map.get("alternatively", [])],
            key=lambda t: t[0],
        )

        exit_done = False

        for ev_rank, (pos_rel, ev_tag) in enumerate(events, start=1):
            fv = build_event_features(
                hidden_states=hidden_states,
                enc_len=enc_len,
                pos_rel=pos_rel,
                layers=feature_layers,
            )
            p_pos = float(probe.predict_proba(fv[None, :])[:, 1][0])
            S = predictive_set_binary(p_pos, q_pos, q_neg)
            set_size = len(S)
            confident_correct = (S == [1])

            meta_rows.append(
                {
                    "row_index": ridx,
                    "event_rank": ev_rank,
                    "pos_rel": pos_rel,
                    "event_tag": ev_tag,
                    "p_pos": p_pos,
                    "predictive_set": S,
                    "set_size": set_size,
                    "exited": False,
                }
            )

            if confident_correct and not exit_done:
                prefix = make_exit_prompt_prefix(
                    tok,
                    seq_ids,
                    enc_len,
                    pos_rel,
                    variant=args.exit_cue_variant,
                )

                final_text, added_tokens = generate_from_prefix(
                    model,
                    tok,
                    prefix,
                    max_new_tokens=args.exit_max_new_tokens,
                    temperature=args.exit_temperature,
                    top_p=args.exit_top_p,
                    do_sample=False,
                )

                early_label = is_correct_answer_with_policy(final_text, gold, policy=args.eval_policy)

                cue_str = get_exit_cue(args.exit_cue_variant)
                cue_ids = tok(cue_str, add_special_tokens=False)["input_ids"]
                cue_len = len(cue_ids)
                used_tokens = (pos_rel - 1) + cue_len + added_tokens

                early_tokens += used_tokens
                early_correct += early_label
                exited_count += 1

                accepted_correct += early_label
                accepted_tokens += used_tokens
                accepted_baseline_correct += base_label

                if base_label == 0 and early_label == 1:
                    help_count += 1
                elif base_label == 1 and early_label == 0:
                    harm_count += 1
                elif base_label == 1 and early_label == 1:
                    unchanged_correct += 1
                else:
                    unchanged_incorrect += 1

                meta_rows[-1].update(
                    {
                        "exited": True,
                        "early_label": early_label,
                        "baseline_label": int(base_label),
                        "used_tokens": used_tokens,
                        "added_tokens": added_tokens,
                        "cue_len": cue_len,
                    }
                )

                early_text_map[ridx] = final_text

                exit_done = True
                if args.verbose:
                    print(f"[exit] row={ridx}, ev_rank={ev_rank}, pos_rel={pos_rel}, p_pos={p_pos:.4f}")
                break

        if not exit_done:
            early_tokens += max(0, new_len)
            early_correct += base_label
            if args.verbose:
                print(f"[no-exit] row={ridx} baseline used")

    t1 = time.time()

    baseline_acc = baseline_correct / max(1, rollouts_attempted)
    early_acc = early_correct / max(1, rollouts_attempted)
    avg_base_tokens = baseline_tokens / max(1, rollouts_attempted)
    avg_early_tokens = early_tokens / max(1, rollouts_attempted)
    savings = avg_base_tokens - avg_early_tokens
    savings_rate = (savings / avg_base_tokens) if avg_base_tokens > 0 else 0.0
    accepted_rate = exited_count / max(1, rollouts_attempted)
    accepted_acc = accepted_correct / max(1, exited_count) if exited_count > 0 else 0.0
    accepted_baseline_acc = accepted_baseline_correct / max(1, exited_count) if exited_count > 0 else 0.0
    avg_accepted_tokens = accepted_tokens / max(1, exited_count) if exited_count > 0 else 0.0

    preds_path = args.out_dir / "early_exit_predictions.jsonl"
    with preds_path.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row) + "\n")

    metrics = {
        "deer_rollout_path": str(args.deer_rollout_path),
        "model_id": args.model_id,
        "probe_path": str(args.probe_path),
        "conformal_path": str(args.conformal_path),
        "rollouts_attempted": rollouts_attempted,
        "baseline": {
            "accuracy": baseline_acc,
            "avg_tokens": avg_base_tokens,
            "total_tokens": baseline_tokens,
        },
        "early_exit": {
            "accuracy": early_acc,
            "avg_tokens": avg_early_tokens,
            "total_tokens": early_tokens,
            "exit_rate": accepted_rate,
            "avg_savings": savings,
            "savings_rate": savings_rate,
        },
        "accepted_subset": {
            "rate": accepted_rate,
            "accuracy": accepted_acc,
            "baseline_accuracy": accepted_baseline_acc,
            "avg_tokens": avg_accepted_tokens,
            "total_tokens": accepted_tokens,
            "help": help_count,
            "harm": harm_count,
            "unchanged_correct": unchanged_correct,
            "unchanged_incorrect": unchanged_incorrect,
        },
        "wall_time_sec": t1 - t0,
    }
    (args.out_dir / "early_exit_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    deer_fmt_path = args.out_dir / "our_early_exit_deer_format.jsonl"
    with deer_fmt_path.open("w", encoding="utf-8") as f_deer:
        for ridx, row in enumerate(rows):
            question = row.get("question") or row.get("problem") or ""
            gold = row.get("gold_answer") or row.get("answer") or ""
            base_gen_list = row.get("generated_responses") or row.get("generated_response") or []
            if isinstance(base_gen_list, list):
                base_ans = base_gen_list[0] if base_gen_list else ""
            else:
                base_ans = str(base_gen_list)
            early_ans = early_text_map.get(ridx, base_ans)
            rec = {
                "question": question,
                "generated_responses": [early_ans],
                "gold_answer": gold,
                "too_long": 0,
                "thinking_steps": 0,
                "rep_end": 0,
                "high_prob": 0,
                "regular_end": 1,
            }
            f_deer.write(json.dumps(rec) + "\n")

    print(f"[done] wrote {preds_path}")
    print(f"[done] wrote {args.out_dir / 'early_exit_metrics.json'}")
    print(f"[done] wrote DEER-format to {deer_fmt_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

