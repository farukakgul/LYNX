#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
import os
import subprocess

# Avoid torchvision imports for text-only pipelines before any HF import
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Ensure repo root on sys.path
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lynx.probing_utils import (
    GenConfig,
    find_pattern_positions_by_tokens,
    forward_hidden_and_logits,
    is_correct_final,
    make_prompt,
    load_aime_dataset,
    load_llm,
)
from lynx.utils import (
    build_event_features,
    make_exit_prompt_prefix,
    generate_from_prefix,
    predictive_set_binary,
    get_exit_cue,
    is_correct_answer_with_policy,
    save_args_json,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Evaluate early-exit with probe + conformal prediction")
    ap.add_argument("--dataset-id", default="Maxwell-Jia/AIME_2024")
    ap.add_argument("--model-id", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")

    ap.add_argument("--test-problems", type=int, default=10)
    ap.add_argument("--test-rollouts", type=int, default=20)
    ap.add_argument("--seed", type=int, default=123)

    # Generation
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)

    # Exit generation (short)
    ap.add_argument("--exit-max-new-tokens", type=int, default=64)
    ap.add_argument("--exit-temperature", type=float, default=0.0, help="Temperature for exit completion (0=greedy)")
    ap.add_argument("--exit-top-p", type=float, default=1.0)
    ap.add_argument("--exit-cue-variant", choices=["paper", "answer_tag"], default="paper")

    ap.add_argument("--probe-path", type=Path, required=True)
    ap.add_argument("--conformal-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/eval_hf"))
    ap.add_argument("--use-math500", action="store_true", help="Load HuggingFaceH4/MATH-500 (single 'test' split)")
    ap.add_argument("--use-gsm8k", action="store_true", help="Load openai/gsm8k (train/test splits; default test)")
    ap.add_argument("--use-csqa", action="store_true", help="Load tau/commonsense_qa (default config)")
    ap.add_argument("--dataset-split", type=str, default="test", help="HF split when using --use-gsm8k (train/test)")
    ap.add_argument("--split-offset", type=int, default=None, help="Start index for test block (single-split or HF datasets)")
    # Custom JSONL dataset (expects keys: problem/question, answer)
    ap.add_argument("--jsonl-path", type=Path, default=None, help="Custom JSONL with fields {problem|question, answer}")
    # Optional rollout caching to avoid re-generating baselines
    ap.add_argument("--rollout-cache", type=Path, default=None, help="Optional JSONL cache of baseline rollouts (load/save)")
    ap.add_argument("--no-cache-save", action="store_true", help="If set, do not append newly generated rollouts to the cache")
    ap.add_argument("--verbose", action="store_true")
    # Evaluation policy toggle (math): 'strict' (default) vs 'deer' (boxed/looser)
    ap.add_argument("--eval-policy", choices=["strict", "deer", "zst"], default="strict",
                    help="Answer checker: 'strict' (numeric/boxed), 'deer' (loose boxed/number), 'zst' (Zero_Step SymPy-based if available)")
    # Optional: export DEER-compatible baseline generations JSONL (problem, generated_responses, gold_answer)
    ap.add_argument("--export-deer-jsonl", type=Path, default=None)
    # Prompt style toggle: 'ours' (default: system+<think>/<answer>) vs 'zst' (Zero_Step minimal user-only)
    ap.add_argument("--prompt-style", choices=["ours", "zst"], default="ours",
                    help="Prompting style for baseline generations. 'zst' uses a single user message with 'Please reason step by step, and put your final answer within \\boxed{}'.")
    # Optional local spawning for parallel shards across visible GPUs
    ap.add_argument("--total-shards", type=int, default=1, help="Split test set into N shards and run in parallel")
    ap.add_argument("--shard-rank", type=int, default=0, help="This worker's shard id [0..N-1]")
    ap.add_argument("--spawn-workers", action="store_true", help="If set with --total-shards>1, spawn N workers locally and aggregate outputs")
    return ap.parse_args()


def main():
    # Avoid importing torchvision through Transformers (text-only pipeline)
    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    # On new architectures (e.g., Blackwell), fused SDPA/flash kernels may not be available yet.
    # Force math (eager) attention kernels to avoid launching missing device-specific kernels.
    try:
        from torch.backends.cuda import sdp_kernel as _sdp_kernel
        _sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save CLI configuration alongside evaluation artifacts.
    try:
        save_args_json(args, args.out_dir / "early_exit_eval_config.json")
    except Exception:
        pass
    rng = np.random.default_rng(args.seed)

    # Optional: spawn N shards locally across visible GPUs and aggregate
    if args.total_shards and args.total_shards > 1 and args.spawn_workers:
        total_to_run = int(args.test_problems)
        base_counts = [total_to_run // args.total_shards] * args.total_shards
        base_counts[-1] += total_to_run % args.total_shards
        # Parse parent visible GPUs
        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        toks = [t.strip() for t in parent_visible.split(",") if t.strip()] if parent_visible else []
        procs = []
        script = Path(__file__).resolve()
        start_offset = int(args.split_offset) if args.split_offset is not None else 0
        for r, count in enumerate(base_counts):
            child_env = os.environ.copy()
            gpu_token = toks[r % len(toks)] if toks else str(r)
            child_env["CUDA_VISIBLE_DEVICES"] = gpu_token
            child_out = args.out_dir / f"shard{r}"
            # Optional cache per shard
            child_cache = None
            if args.rollout_cache:
                child_cache = args.rollout_cache.parent / f"{args.rollout_cache.stem}_shard{r}{args.rollout_cache.suffix}"
            py_exe = sys.executable or os.environ.get("PYTHON") or os.environ.get("PYTHON3") or "python"
            cmd = [
                py_exe,
                str(script),
                "--dataset-id", args.dataset_id,
                "--model-id", args.model_id,
                "--dtype", args.dtype,
                "--device-map", args.device_map,
                "--test-problems", str(count),
                "--test-rollouts", str(args.test_rollouts),
                "--seed", str(args.seed),
                "--max-new-tokens", str(args.max_new_tokens),
                "--temperature", str(args.temperature),
                "--top-p", str(args.top_p),
                "--exit-max-new-tokens", str(args.exit_max_new_tokens),
                "--exit-temperature", str(args.exit_temperature),
                "--exit-top-p", str(args.exit_top_p),
                "--exit-cue-variant", args.exit_cue_variant,
                "--probe-path", str(args.probe_path),
                "--conformal-path", str(args.conformal_path),
                "--out-dir", str(child_out),
                "--prompt-style", args.prompt_style,
                "--split-offset", str(start_offset),
                "--total-shards", "1",
                "--shard-rank", str(r),
            ]
            if args.export_deer_jsonl:
                child_deer = (child_out / "deer_baseline.jsonl")
                cmd.extend(["--export-deer-jsonl", str(child_deer)])
            if args.use_math500:
                cmd.append("--use-math500")
            if args.use_gsm8k:
                cmd.append("--use-gsm8k")
                if args.dataset_split:
                    cmd.extend(["--dataset-split", str(args.dataset_split)])
            if args.use_csqa:
                cmd.append("--use-csqa")
                if args.dataset_split:
                    cmd.extend(["--dataset-split", str(args.dataset_split)])
            if child_cache:
                cmd.extend(["--rollout-cache", str(child_cache)])
                if args.no_cache_save:
                    cmd.append("--no-cache-save")
            print(f"[spawn] shard {r}/{args.total_shards}: {' '.join(cmd)}  (CUDA_VISIBLE_DEVICES={child_env.get('CUDA_VISIBLE_DEVICES')})")
            procs.append((subprocess.Popen(cmd, env=child_env), child_out))
            start_offset += count
        # Wait for children to finish
        rc = 0
        for p, _ in procs:
            rc |= p.wait()
        # Aggregate outputs
        try:
            preds_out = args.out_dir / "early_exit_predictions.jsonl"
            with preds_out.open("w", encoding="utf-8") as fout:
                for _, ch_out in procs:
                    pth = ch_out / "early_exit_predictions.jsonl"
                    if pth.exists():
                        with pth.open("r", encoding="utf-8") as fin:
                            for line in fin:
                                fout.write(line)
            ms = []
            for _, ch_out in procs:
                mpath = ch_out / "early_exit_metrics.json"
                if mpath.exists():
                    ms.append(json.loads(mpath.read_text()))
            if ms:
                tot_roll = sum(m.get("rollouts_attempted", 0) for m in ms)
                tot_exit = sum(m.get("exited_count", 0) for m in ms)
                base_tok = sum(m.get("baseline", {}).get("total_tokens", 0) for m in ms)
                base_corr = sum(m.get("baseline", {}).get("accuracy", 0.0) * m.get("rollouts_attempted", 0) for m in ms)
                early_tok = sum(m.get("early_exit", {}).get("total_tokens", 0) for m in ms)
                early_corr = sum(m.get("early_exit", {}).get("accuracy", 0.0) * m.get("rollouts_attempted", 0) for m in ms)
                base_avg = base_tok / max(1, tot_roll)
                early_avg = early_tok / max(1, tot_roll)
                out = {
                    "dataset_id": ms[0].get("dataset_id"),
                    "test_problems": sum(m.get("test_problems", 0) for m in ms),
                    "test_rollouts": ms[0].get("test_rollouts"),
                    "llm_device": "multi-gpu",
                    "probe_path": ms[0].get("probe_path"),
                    "conformal_path": ms[0].get("conformal_path"),
                    "rollouts_attempted": tot_roll,
                    "exited_count": tot_exit,
                    "baseline": {
                        "accuracy": base_corr / max(1, tot_roll),
                        "avg_tokens": base_avg,
                        "total_tokens": base_tok,
                    },
                    "early_exit": {
                        "accuracy": early_corr / max(1, tot_roll),
                        "avg_tokens": early_avg,
                        "total_tokens": early_tok,
                        "exit_rate": tot_exit / max(1, tot_roll),
                        "avg_savings": base_avg - early_avg,
                        "savings_rate": (base_avg - early_avg) / base_avg if base_avg > 0 else 0.0,
                    },
                }
                (args.out_dir / "early_exit_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
                print(f"[spawn] Aggregated metrics written to {args.out_dir}/early_exit_metrics.json")
        except Exception as e:
            print(f"[spawn] aggregation failed: {e}")
        raise SystemExit(rc)

    # Guard against unsupported bf16 on older GPUs: transparently fall back to float16
    if args.dtype == "bfloat16" and torch.cuda.is_available():
        try:
            bf16_ok = torch.cuda.is_bf16_supported()
        except Exception:
            bf16_ok = False
        if not bf16_ok:
            print("[warn] GPU does not support bfloat16; falling back to float16 for model load.")
            args.dtype = "float16"

    if args.jsonl_path:
        # Load custom JSONL dataset
        problems = []
        answers = []
        try:
            with open(args.jsonl_path, "r", encoding="utf-8") as fh:
                import json as _json
                for line in fh:
                    try:
                        row = _json.loads(line)
                    except Exception:
                        continue
                    q = row.get("problem") or row.get("question") or ""
                    a = row.get("answer", "")
                    problems.append(str(q))
                    answers.append(str(a))
        except Exception as e:
            raise RuntimeError(f"Failed to read JSONL dataset: {args.jsonl_path}: {e}")
        total = len(problems)
        perm = rng.permutation(total).tolist()
        if args.split_offset is not None:
            start = max(0, int(args.split_offset))
            end = min(total, start + int(args.test_problems))
            test_indices = sorted(perm[start:end])
        else:
            test_indices = sorted(perm[-args.test_problems :])
        csqa_choices = None
    elif args.use_math500:
        try:
            import datasets as hf_datasets  # type: ignore
        except Exception as e:
            raise RuntimeError("Please install 'datasets' to use --use-math500.") from e
        def _extract_math_gold(solution: str) -> str:
            if not isinstance(solution, str):
                return ""
            import re as _re
            m = _re.search(r"####\s*([^\n]+)", solution)
            ans = (m.group(1) if m else solution).strip().rstrip(". )]")
            return ans
        ds = hf_datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
        problems = [row["problem"] for row in ds]
        def _norm_ans(a: str) -> str:
            return (a or "").strip().rstrip(". )]")
        answers = [_norm_ans(row.get("answer")) or _extract_math_gold(row["solution"]) for row in ds]
        total = len(problems)
        perm = rng.permutation(total).tolist()
        if args.split_offset is not None:
            start = max(0, int(args.split_offset))
            end = min(total, start + int(args.test_problems))
            test_indices = sorted(perm[start:end])
        else:
            test_indices = sorted(perm[-args.test_problems :])
    elif args.use_gsm8k:
        try:
            import datasets as hf_datasets  # type: ignore
        except Exception as e:
            raise RuntimeError("Please install 'datasets' to use --use-gsm8k.") from e
        def _extract_gsm8k_gold(ans_text: str) -> str:
            import re as _re
            if not isinstance(ans_text, str):
                return ""
            m = _re.search(r"####\s*([^\n]+)", ans_text)
            if m:
                return m.group(1).strip().rstrip(". )]")
            nums = _re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", ans_text)
            return (nums[-1] if nums else ans_text).strip().rstrip(". )]")
        # Use the 'main' configuration for GSM8K
        ds = hf_datasets.load_dataset("openai/gsm8k", "main", split=(args.dataset_split or "test"))
        problems = [row.get("question", "") for row in ds]
        # Always extract answer after #### from 'answer' rationale for GSM8K
        answers = [_extract_gsm8k_gold(row.get("answer", "")) for row in ds]
        total = len(problems)
        perm = rng.permutation(total).tolist()
        if args.split_offset is not None:
            start = max(0, int(args.split_offset))
            end = min(total, start + int(args.test_problems))
            test_indices = sorted(perm[start:end])
        else:
            test_indices = sorted(perm[-args.test_problems :])
    elif args.use_csqa:
        try:
            import datasets as hf_datasets  # type: ignore
        except Exception as e:
            raise RuntimeError("Please install 'datasets' to use --use-csqa.") from e
        def _format_csqa(q: str, labels, texts) -> str:
            lines = ["Question: " + (q or "").strip(), "Choices:"]
            for lab, txt in zip(labels or [], texts or []):
                lines.append(f"{lab}. {txt}")
            lines.append(
                "Please think step by step, then give your final answer as \\boxed{<LETTER>} where <LETTER> is one of A,B,C,D,E."
            )
            return "\n".join(lines)
        split_name = args.dataset_split or "validation"
        ds = hf_datasets.load_dataset("tau/commonsense_qa", "default", split=split_name)
        problems = []
        answers = []
        choices_labels = []
        choices_texts = []
        for row in ds:
            q = row.get("question", "")
            ch = row.get("choices") or {}
            labs = [str(x).strip().upper() for x in (ch.get("label") or [])]
            txts = list(ch.get("text") or [])
            problems.append(_format_csqa(q, labs, txts))
            answers.append(str(row.get("answerKey", "")).strip().upper())
            choices_labels.append(labs)
            choices_texts.append([str(t) for t in txts])
        total = len(problems)
        perm = rng.permutation(total).tolist()
        if args.split_offset is not None:
            start = max(0, int(args.split_offset))
            end = min(total, start + int(args.test_problems))
            test_indices = sorted(perm[start:end])
        else:
            test_indices = sorted(perm[-args.test_problems :])
        csqa_choices = {i: (choices_labels[i], choices_texts[i]) for i in range(total)}
    else:
        problems, answers = load_aime_dataset(args.dataset_id)
        total = len(problems)
        perm = rng.permutation(total).tolist()
        if args.split_offset is not None:
            start = max(0, int(args.split_offset))
            end = min(total, start + int(args.test_problems))
            test_indices = sorted(perm[start:end])
        else:
            test_indices = sorted(perm[-args.test_problems :])
    print(f"Test problems: {test_indices}")

    # Load LLM
    t0_models = time.time()
    model, tok = load_llm(args.model_id, device_map=args.device_map, torch_dtype=args.dtype)
    t1_models = time.time()
    llm_device = str(next(model.parameters()).device)
    print(f"[eval] Loaded LLM on {llm_device} in {(t1_models - t0_models):.2f}s")

    # Decide feature layers based on model depth / id.
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

    # Load probe + conformal thresholds
    with args.probe_path.open("rb") as fh:
        probe = pickle.load(fh)
    conf = json.loads(args.conformal_path.read_text())
    q_pos = float(conf.get("q_pos", float("inf")))
    q_neg = float(conf.get("q_neg", float("inf")))

    gcfg = GenConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=(float(args.temperature) > 0.0),
    )

    # Lazy import to avoid circular import in type checkers
    from lynx.probing_utils import generate_once

    # Prepare optional DEER-compatible export file
    deer_fh = None
    if args.export_deer_jsonl:
        try:
            args.export_deer_jsonl.parent.mkdir(parents=True, exist_ok=True)
            deer_fh = args.export_deer_jsonl.open("w", encoding="utf-8")
        except Exception as e:
            print(f"[warn] could not open export file {args.export_deer_jsonl}: {e}")

    # Load rollout cache if provided
    cache_map = {}
    if args.rollout_cache and args.rollout_cache.exists():
        try:
            with args.rollout_cache.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        rec = json.loads(line)
                        k = (int(rec.get("index")), int(rec.get("rollout")))
                        cache_map[k] = rec.get("text", "")
                    except Exception:
                        continue
            if args.verbose:
                print(f"[cache] Loaded {len(cache_map)} rollouts from {args.rollout_cache}")
        except Exception as e:
            print(f"[cache] Failed to read cache {args.rollout_cache}: {e}")

    meta_rows: List[dict] = []
    rollouts_attempted = 0

    baseline_correct = 0
    baseline_tokens = 0
    early_correct = 0
    early_tokens = 0
    exited_count = 0
    accepted_correct = 0
    accepted_tokens = 0
    accepted_baseline_correct = 0
    help_count = 0  # baseline wrong -> early right
    harm_count = 0  # baseline right  -> early wrong
    unchanged_correct = 0  # baseline right -> early right
    unchanged_incorrect = 0  # baseline wrong -> early wrong

    t0 = time.time()
    for idx in tqdm(test_indices, desc="problems"):
        problem = problems[idx]
        gold = answers[idx]

        for r in range(args.test_rollouts):
            rollouts_attempted += 1
            # Baseline rollout (optionally from cache)
            cached = cache_map.get((idx, r)) if cache_map else None
            if cached is not None and isinstance(cached, str) and cached.strip():
                base_text = cached
                # Reconstruct seq_ids and enc_len from text and prompt
                if args.prompt_style == "zst":
                    # Zero_Step-style prompt: minimal user turn + boxed instruction
                    try:
                        enc_prompt = tok.apply_chat_template([
                            {"role": "user", "content": problem + "\nPlease reason step by step, and put your final answer within \\boxed{}."}
                        ], tokenize=False, add_generation_prompt=True)
                    except Exception:
                        enc_prompt = f"Question: {problem}\nAnswer: "
                else:
                    enc_prompt = make_prompt(tok, problem)
                enc_len = tok(enc_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
                seq_ids = tok(base_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()
            else:
                if args.prompt_style == "zst":
                    # Inline generate_once with Zero_Step-style prompt
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
                    base_text = tok.decode(seq, skip_special_tokens=True)
                    seq_ids = seq.tolist()
                else:
                    base_text, seq_ids, enc_len = generate_once(model, tok, problem, gcfg)
            # Optional DEER export: write baseline record
            if deer_fh is not None:
                try:
                    rec = {
                        "problem": problem,
                        "generated_responses": [base_text],
                        "gold_answer": str(gold),
                        "index": int(idx),
                        "rollout": int(r),
                    }
                    deer_fh.write(json.dumps(rec) + "\n")
                except Exception:
                    pass
                # Append to cache if requested
                if args.rollout_cache and not args.no_cache_save:
                    try:
                        args.rollout_cache.parent.mkdir(parents=True, exist_ok=True)
                        with args.rollout_cache.open("a", encoding="utf-8") as fh:
                            fh.write(json.dumps({"index": idx, "rollout": r, "text": base_text}) + "\n")
                    except Exception as e:
                        if args.verbose:
                            print(f"[cache] Failed to append rollout (idx={idx}, r={r}) to {args.rollout_cache}: {e}")
            new_len = len(seq_ids) - enc_len
            baseline_tokens += max(0, new_len)
            if args.use_csqa:
                labs, txts = csqa_choices.get(idx, (None, None))  # type: ignore
                base_label = is_correct_answer_with_policy(base_text, gold, policy=args.eval_policy, mcq_labels=labs, mcq_texts=txts)
            else:
                base_label = is_correct_answer_with_policy(base_text, gold, policy=args.eval_policy)
            baseline_correct += base_label

            # Single forward pass for features
            logits_TV, hidden_states = forward_hidden_and_logits(model, seq_ids)
            new_ids = seq_ids[enc_len:]
            pos_map = find_pattern_positions_by_tokens(new_ids, tok, patterns=("hmm", "wait", "alternatively"))
            events = sorted(
                [(p, "hmm") for p in pos_map.get("hmm", [])]
                + [(p, "wait") for p in pos_map.get("wait", [])]
                + [(p, "alternatively") for p in pos_map.get("alternatively", [])],
                key=lambda t: t[0],
            )

            exit_done = False
            exit_info = None

            for rank, (pos_rel, lab) in enumerate(events, start=1):
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

                meta_rows.append({
                    "index": idx,
                    "rollout": r,
                    "event_rank": rank,
                    "pos_rel": pos_rel,
                    "event": lab,
                    "p_pos": p_pos,
                    "predictive_set": S,
                    "set_size": set_size,
                    "exited": False,
                })

                if confident_correct and not exit_done:
                    # Early exit: append cue, generate short completion, evaluate
                    prefix = make_exit_prompt_prefix(tok, seq_ids, enc_len, pos_rel, variant=args.exit_cue_variant)
                    # Include cost of exit-cue prompt tokens
                    cue_str = get_exit_cue(args.exit_cue_variant)
                    cue_tok = tok
                    cue_ids = cue_tok(cue_str, add_special_tokens=False).input_ids if hasattr(cue_tok, "__call__") else cue_tok.encode(cue_str, add_special_tokens=False)
                    cue_len = len(cue_ids)
                    final_text, added_tokens = generate_from_prefix(
                        model, tok, prefix,
                        max_new_tokens=args.exit_max_new_tokens,
                        temperature=args.exit_temperature,
                        top_p=args.exit_top_p,
                        do_sample=False,
                    )
                    if args.use_csqa:
                        labs, txts = csqa_choices.get(idx, (None, None))  # type: ignore
                        early_label = is_correct_answer_with_policy(final_text, gold, policy=args.eval_policy, mcq_labels=labs, mcq_texts=txts)
                    else:
                        early_label = is_correct_answer_with_policy(final_text, gold, policy=args.eval_policy)
                    # Tokens used up to (before) event + exit-cue + short completion
                    used_tokens = (pos_rel - 1) + cue_len + added_tokens
                    early_tokens += used_tokens
                    early_correct += early_label
                    accepted_correct += early_label
                    accepted_tokens += used_tokens
                    accepted_baseline_correct += base_label
                    # Help/harm counters
                    if base_label == 0 and early_label == 1:
                        help_count += 1
                    elif base_label == 1 and early_label == 0:
                        harm_count += 1
                    elif base_label == 1 and early_label == 1:
                        unchanged_correct += 1
                    else:
                        unchanged_incorrect += 1
                    # Annotate the just-appended event row with exit details
                    meta_rows[-1].update({
                        "exited": True,
                        "early_label": early_label,
                        "baseline_label": int(base_label),
                        "used_tokens": used_tokens,
                        "added_tokens": added_tokens,
                        "cue_len": cue_len,
                    })
                    exit_done = True
                    exit_info = {
                        "event_rank": rank,
                        "pos_rel": pos_rel,
                        "added_tokens": added_tokens,
                        "used_tokens": used_tokens,
                        "p_pos": p_pos,
                        "predictive_set": S,
                        "early_label": early_label,
                    }
                    exited_count += 1
                    break

            if not exit_done:
                # No confident early exit: use baseline
                early_tokens += max(0, new_len)
                early_correct += base_label
                exit_info = exit_info or {"event_rank": None, "pos_rel": None, "added_tokens": 0, "used_tokens": new_len, "p_pos": None, "predictive_set": [], "early_label": base_label}

            if args.verbose:
                print(f"[run] idx={idx} r={r} exit={exit_done} info={exit_info}")

    t1 = time.time()

    # Metrics
    baseline_acc = baseline_correct / max(1, rollouts_attempted)
    early_acc = early_correct / max(1, rollouts_attempted)
    avg_base_tokens = baseline_tokens / max(1, rollouts_attempted)
    avg_early_tokens = early_tokens / max(1, rollouts_attempted)
    savings = avg_base_tokens - avg_early_tokens
    savings_rate = (savings / avg_base_tokens) if avg_base_tokens > 0 else 0.0

    # Accepted subset metrics
    accepted_rate = exited_count / max(1, rollouts_attempted)
    accepted_acc = accepted_correct / max(1, exited_count)
    accepted_baseline_acc = accepted_baseline_correct / max(1, exited_count)
    avg_accepted_tokens = accepted_tokens / max(1, exited_count)

    # Canonical dataset id in outputs for HF datasets
    dataset_id_out = (
        "math500" if args.use_math500 else (
            "gsm8k" if args.use_gsm8k else (
                "csqa" if args.use_csqa else str(args.dataset_id)
            )
        )
    )
    summary = {
        "dataset_id": dataset_id_out,
        "test_problems": args.test_problems,
        "test_rollouts": args.test_rollouts,
        "llm_device": llm_device,
        "probe_path": str(args.probe_path),
        "conformal_path": str(args.conformal_path),
        "rollouts_attempted": rollouts_attempted,
        "exited_count": exited_count,
        "baseline": {
            "accuracy": float(baseline_acc),
            "avg_tokens": float(avg_base_tokens),
            "total_tokens": int(baseline_tokens),
        },
        "early_exit": {
            "accuracy": float(early_acc),
            "avg_tokens": float(avg_early_tokens),
            "total_tokens": int(early_tokens),
            "exit_rate": float(exited_count / max(1, rollouts_attempted)),
            "avg_savings": float(savings),
            "savings_rate": float(savings_rate),
        },
        "accepted_subset": {
            "rate": float(accepted_rate),
            "accuracy": float(accepted_acc),
            "baseline_accuracy": float(accepted_baseline_acc),
            "avg_tokens": float(avg_accepted_tokens),
            "total_tokens": int(accepted_tokens),
            "help": int(help_count),
            "harm": int(harm_count),
            "unchanged_correct": int(unchanged_correct),
            "unchanged_incorrect": int(unchanged_incorrect),
        },
        "wall_time_sec": float(t1 - t0),
    }

    out_jsonl = args.out_dir / "early_exit_predictions.jsonl"
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in meta_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved: {out_jsonl}")

    # Close DEER export file if open
    if deer_fh is not None:
        try:
            deer_fh.close()
        except Exception:
            pass

    out_metrics = args.out_dir / "early_exit_metrics.json"
    out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved: {out_metrics}")


if __name__ == "__main__":
    main()
