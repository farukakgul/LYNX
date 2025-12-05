#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import os
import subprocess

# Ensure repo root is on sys.path when running as a script
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lynx.probing_utils import (
    GenConfig,
    find_pattern_positions_by_tokens,
    forward_hidden_and_logits,
    is_correct_final,
    load_llm,
)
from lynx.utils import (
    build_event_features,
    make_exit_prompt_prefix,
    generate_from_prefix,
    is_correct_answer_with_policy,
    save_args_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Build a global early-exit probe dataset from Hendrycks MATH subsets (EleutherAI/hendrycks_math)")
    ap.add_argument("--dataset-id", default="EleutherAI/hendrycks_math")
    ap.add_argument(
        "--subsets",
        type=str,
        default="algebra,counting_and_probability,intermediate_algebra,number_theory,prealgebra",
        help="Comma-separated Hendrycks MATH subset names",
    )
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")

    ap.add_argument("--split", choices=["train", "calib"], required=True)
    ap.add_argument("--limit-per-subset", type=int, default=-1, help="Max problems per subset (<=0 means all)")
    ap.add_argument(
        "--subset-offset",
        type=int,
        default=0,
        help="Deterministic offset into each subset's RNG permutation before taking --limit-per-subset items. Use same --seed across runs and vary this offset to avoid overlap (e.g., train offset=0, calib offset=train_limit).",
    )
    ap.add_argument("--rollouts", type=int, default=1)
    ap.add_argument("--events-per-rollout", type=int, default=80)
    ap.add_argument("--skip-first-hmmwait", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    # Generation params
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)

    # Exit generation (short)
    ap.add_argument("--exit-max-new-tokens", type=int, default=96)
    ap.add_argument("--exit-temperature", type=float, default=0.0, help="Temperature for exit completion (0=greedy)")
    ap.add_argument("--exit-top-p", type=float, default=1.0)
    ap.add_argument("--exit-cue-variant", choices=["paper", "answer_tag"], default="paper")

    ap.add_argument("--out-dir", type=Path, default=Path("outputs/artifacts"))
    # Optional sharding across multiple GPUs
    ap.add_argument("--total-shards", type=int, default=1, help="Split selected items into N shards")
    ap.add_argument("--shard-rank", type=int, default=0, help="This worker's shard rank [0..N-1]")
    ap.add_argument("--spawn-workers", action="store_true", help="If set with --total-shards>1, spawn N workers locally")
    ap.add_argument("--debug-save-first", type=int, default=0, help="Save first N forced-exit examples to debug.jsonl")
    # Prompt style toggle for baseline generations used to extract events/features
    ap.add_argument(
        "--prompt-style", choices=["ours", "zst"], default="ours",
        help=(
            "Prompting style for baseline generation. 'zst' uses a single user message with "
            "'Please reason step by step, and put your final answer within \\boxed{}'."
        ),
    )
    return ap.parse_args()


def _extract_math_gold(solution: str) -> str:
    import re
    if not isinstance(solution, str):
        return ""
    m = re.search(r"####\s*([^\n]+)", solution)
    ans = (m.group(1) if m else solution).strip().rstrip(". )]")
    return ans


def _normalize_latex_answer(ans: str) -> str:
    """Lightweight LaTeX simplifier for gold answers.
    - Converts \frac{a}{b} -> a/b
    - Strips common LaTeX spacing/control tokens and $ delimiters
    - Trims trailing punctuation and spaces
    """
    import re
    if not isinstance(ans, str):
        return ""
    s = ans.strip()
    s = s.strip("$")
    # \frac or \dfrac
    s = re.sub(r"\\d?frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}", r"\1/\2", s)
    # Remove common formatting tokens
    s = re.sub(r"\\left|\\right|\\,|\\;|\\:|\\!|\\\s+", "", s)
    s = s.strip().rstrip(". )]")
    return s


def _extract_gold_from_solution(sol: str) -> str:
    """Robustly extract a final answer from Hendrycks-style solutions.
    Priority:
      1) last \boxed{...}
      2) '#### <answer>' marker
      3) last number-like token
    Returns a short, normalized string (e.g., '49', '3/4').
    """
    import re
    if not isinstance(sol, str):
        return ""
    s = sol.strip()
    # 1) last boxed
    boxed = re.findall(r"\\boxed\{([^}]*)\}", s)
    if boxed:
        return _normalize_latex_answer(boxed[-1])
    # 2) '#### answer'
    m = re.search(r"####\s*([^\n]+)", s)
    if m:
        return _normalize_latex_answer(m.group(1))
    # 3) last number-like token (including fractions/decimals)
    nums = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", s)
    if nums:
        return _normalize_latex_answer(nums[-1])
    return ""


def build_item_list(dataset_id: str, subsets: List[str], limit_per_subset: int, rng: np.random.Generator, subset_offset: int) -> List[Tuple[str, int]]:
    try:
        import datasets as hf_datasets  # type: ignore
    except Exception as e:
        raise RuntimeError("Please install 'datasets' to load EleutherAI/hendrycks_math.") from e

    items: List[Tuple[str, int]] = []
    for name in subsets:
        ds = hf_datasets.load_dataset(dataset_id, name, split="train")
        n = len(ds)
        # Deterministic permutation per subset under the provided RNG state
        perm = rng.permutation(n).tolist()
        start = max(0, int(subset_offset or 0))
        if limit_per_subset and limit_per_subset > 0:
            end = min(n, start + int(limit_per_subset))
        else:
            end = n
        if start >= n:
            sel = []
        else:
            sel = perm[start:end]
        for i in sel:
            items.append((name, int(i)))
    # Shuffle globally for fair sharding
    rng.shuffle(items)
    return items


def main():
    # Avoid importing torchvision through Transformers (text-only pipeline)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    # Force math attention kernels to avoid unavailable fused kernels on new GPUs.
    try:
        from torch.backends.cuda import sdp_kernel as _sdp_kernel
        _sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save CLI configuration alongside outputs for reproducibility.
    try:
        save_args_json(args, args.out_dir / f"build_global_math_{args.split}_config.json")
    except Exception:
        pass
    rng = np.random.default_rng(args.seed)

    # Guard against unsupported bf16 on older GPUs: transparently fall back to float16
    if args.dtype == "bfloat16" and torch.cuda.is_available():
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        if not bf16_ok:
            print("[warn] GPU does not support bfloat16; falling back to float16 for model load.")
            args.dtype = "float16"

    subsets = [s.strip() for s in args.subsets.split(",") if s.strip()]
    items_all = build_item_list(args.dataset_id, subsets, args.limit_per_subset, rng, args.subset_offset)

    # Optional local spawning for parallel shards
    if args.total_shards > 1 and args.spawn_workers:
        procs = []
        script = Path(__file__).resolve()
        for r in range(args.total_shards):
            child_env = os.environ.copy()
            parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
            if parent_visible:
                toks = [t.strip() for t in parent_visible.split(",") if t.strip()]
                gpu_token = toks[r % len(toks)] if toks else str(r)
                child_env["CUDA_VISIBLE_DEVICES"] = gpu_token
            else:
                child_env["CUDA_VISIBLE_DEVICES"] = str(r)
            py_exe = sys.executable or os.environ.get("PYTHON") or os.environ.get("PYTHON3") or "python"
            cmd = [
                py_exe,
                str(script),
                "--dataset-id", args.dataset_id,
                "--subsets", args.subsets,
                "--model-id", args.model_id,
                "--dtype", args.dtype,
                "--device-map", "cuda:0",
                "--split", args.split,
                "--limit-per-subset", str(args.limit_per_subset),
                "--rollouts", str(args.rollouts),
                "--events-per-rollout", str(args.events_per_rollout),
                "--skip-first-hmmwait", str(args.skip_first_hmmwait),
                "--seed", str(args.seed),
                "--subset-offset", str(args.subset_offset),
                "--max-new-tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature),
                "--top-p", str(args.top_p),
                "--exit-max-new-tokens", str(args.exit_max_new_tokens),
                "--exit-temperature", str(args.exit_temperature),
                "--exit-top-p", str(args.exit_top_p),
                "--exit-cue-variant", args.exit_cue_variant,
                "--out-dir", str(args.out_dir),
                "--total-shards", str(args.total_shards),
                "--shard-rank", str(r),
                "--prompt-style", args.prompt_style,
            ]
            print(f"[spawn] shard {r}/{args.total_shards}: {' '.join(cmd)}  (CUDA_VISIBLE_DEVICES={child_env.get('CUDA_VISIBLE_DEVICES')})")
            procs.append(subprocess.Popen(cmd, env=child_env))
        rc = 0
        for p in procs:
            rc |= p.wait()
        raise SystemExit(rc)

    # Shard selection without spawning
    if args.total_shards > 1:
        flat = list(range(len(items_all)))
        sel_idx = [k for j, k in enumerate(flat) if j % args.total_shards == args.shard_rank]
        items = [items_all[k] for k in sel_idx]
        print(f"[{args.split}] items shard {args.shard_rank}/{args.total_shards}: {len(items)}")
    else:
        items = items_all
        print(f"[{args.split}] items: {len(items)}")

    # Load models
    model, tok = load_llm(args.model_id, device_map=args.device_map, torch_dtype=args.dtype)
    gcfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(float(args.temperature) > 0.0),
    )

    feats: List[np.ndarray] = []
    labels: List[int] = []
    metas: List[dict] = []

    # Decide feature layers based on model depth / id.
    # For specific families we tap a few intermediate layers + last:
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

    # Lazy import to avoid circulars
    from datasets import load_dataset as hf_load_dataset  # type: ignore
    from lynx.probing_utils import generate_once

    # Group items by subset for efficient loading
    by_subset: dict[str, List[int]] = {}
    for name, idx in items:
        by_subset.setdefault(name, []).append(idx)

    for name in subsets:
        sel = by_subset.get(name, [])
        if not sel:
            continue
        ds = hf_load_dataset(args.dataset_id, name, split="train")
        for idx in tqdm(sel, desc=f"subset={name}"):
            row = ds[int(idx)]
            problem = row.get("problem") or row.get("question") or ""
            sol_text = str(row.get("solution", ""))
            ans_field = row.get("answer")
            gold_raw = ans_field if isinstance(ans_field, str) and ans_field.strip() else _extract_gold_from_solution(sol_text)
            problem = str(problem)
            gold = _normalize_latex_answer(str(gold_raw))

            for r in range(args.rollouts):
                try:
                    if args.prompt_style == "zst":
                        # Zero_Step-style prompt: minimal user turn + boxed instruction
                        try:
                            prompt = tok.apply_chat_template([
                                {"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
                            ], tokenize=False, add_generation_prompt=True)
                        except Exception:
                            prompt = f"Question: {problem}\nAnswer: "
                        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
                        for k in enc:
                            enc[k] = enc[k].to(model.device)
                        enc_len = enc["input_ids"].shape[1]
                        gen = model.generate(
                            **enc,
                            max_new_tokens=gcfg.max_new_tokens,
                            do_sample=gcfg.do_sample,
                            temperature=gcfg.temperature,
                            top_p=gcfg.top_p,
                            return_dict_in_generate=True,
                            eos_token_id=tok.eos_token_id,
                            pad_token_id=tok.eos_token_id,
                            use_cache=True,
                        )
                        seq = gen.sequences[0].to(model.device)
                        text = tok.decode(seq, skip_special_tokens=True)
                        seq_ids = seq.tolist()
                    else:
                        text, seq_ids, enc_len = generate_once(model, tok, problem, gcfg)
                except Exception as e:
                    print(f"[warn] generation failed for subset={name} index={idx} rollout={r}: {e}")
                    continue

                try:
                    logits_TV, hidden_states = forward_hidden_and_logits(model, seq_ids)
                except Exception as e:
                    print(f"[warn] forward failed for subset={name} index={idx} rollout={r}: {e}")
                    continue
                new_ids = seq_ids[enc_len:]
                pos_map = find_pattern_positions_by_tokens(new_ids, tok, patterns=("hmm", "wait", "alternatively"))
                pairs = sorted(
                    [(p, "hmm") for p in pos_map.get("hmm", [])]
                    + [(p, "wait") for p in pos_map.get("wait", [])]
                    + [(p, "alternatively") for p in pos_map.get("alternatively", [])],
                    key=lambda t: t[0],
                )
                pairs = pairs[args.skip_first_hmmwait :]
                if args.events_per_rollout > 0:
                    pairs = pairs[: args.events_per_rollout]
                if not pairs:
                    L = max(1, len(new_ids))
                    pos_rel_fallback = max(1, int(0.7 * L))
                    pairs = [(pos_rel_fallback, "synthetic")]

                saved_debug = 0
                for (pos_rel, lab) in pairs:
                    fv = build_event_features(
                        hidden_states=hidden_states,
                        enc_len=enc_len,
                        pos_rel=pos_rel,
                        layers=feature_layers,
                    )
                    prefix = make_exit_prompt_prefix(tok, seq_ids, enc_len, pos_rel, variant=args.exit_cue_variant)
                    early_text, added_tokens = generate_from_prefix(
                        model, tok, prefix,
                        max_new_tokens=args.exit_max_new_tokens,
                        temperature=args.exit_temperature,
                        top_p=args.exit_top_p,
                        do_sample=False,
                    )
                    # Use the tolerant ZST-style checker for Hendrycks MATH to minimize label noise.
                    y_event = is_correct_answer_with_policy(early_text, gold, policy="zst")
                    feats.append(fv)
                    labels.append(y_event)
                    metas.append({
                        "subset": name,
                        "index": int(idx),
                        "rollout": r,
                        "pos_rel": pos_rel,
                        "event": lab,
                        "added_tokens": int(added_tokens),
                        "label": int(y_event),
                    })
                    if args.debug_save_first and saved_debug < args.debug_save_first:
                        dbg = {
                            "subset": name,
                            "index": int(idx),
                            "rollout": r,
                            "pos_rel": pos_rel,
                            "event": lab,
                            "cue_variant": args.exit_cue_variant,
                            "prefix_tail": prefix[-200:],
                            "early_tail": early_text[-400:],
                            "gold": gold,
                            "label": int(y_event),
                        }
                        (args.out_dir / f"debug_globalmath_{args.split}.jsonl").open("a", encoding="utf-8").write(json.dumps(dbg) + "\n")
                        saved_debug += 1

    if not feats:
        raise RuntimeError("No features collected; check event detection / generation settings.")

    X = np.stack(feats, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    shard_suffix = (
        f"_shard{args.shard_rank}of{args.total_shards}" if args.total_shards and args.total_shards > 1 else ""
    )
    out_npz = args.out_dir / f"globalmath_{args.split}_dataset{shard_suffix}.npz"
    np.savez_compressed(out_npz, X=X, y=y)
    out_meta = args.out_dir / f"globalmath_{args.split}_meta{shard_suffix}.jsonl"
    with out_meta.open("w", encoding="utf-8") as f:
        for row in metas:
            f.write(json.dumps(row) + "\n")
    print(f"Saved: {out_npz}  (X={X.shape}, positives={y.sum()}, negatives={(y==0).sum()})")
    print(f"Meta:  {out_meta}")


if __name__ == "__main__":
    main()
