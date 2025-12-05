#!/usr/bin/env python3
from __future__ import annotations

"""
Build an early-exit probe dataset using a hybrid vLLM + HF pipeline.

High-level idea
---------------
- Use vLLM for the expensive long-horizon baseline decoding (fast generation).
- Use a single HF forward pass per rollout to get hidden states for all tokens.
- Reuse the existing hmm/wait event localization and exit-labeling logic from
  `lynx/build_dataset.py`, so the resulting NPZ/meta are
  compatible with `train_probe.py` and `calibrate_conformal.py`.

This script is intended as a drop-in alternative to `build_dataset.py` when
vLLM is available and you want faster dataset construction, especially for
long-context models like Qwen3-8B.
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

# Text-only: avoid torchvision imports
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Repo root (parent of the `lynx` package)
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vllm import LLM, SamplingParams

from lynx.probing_utils import (
    find_pattern_positions_by_tokens,
    forward_hidden_and_logits,
    load_aime_dataset,
    load_llm,
)
from lynx.utils import (
    build_event_features,
    make_exit_prompt_prefix,
    generate_from_prefix,
    is_correct_answer,
    is_correct_answer_with_policy,
    save_args_json,
)
from lynx.build_global_math_dataset import (
    build_item_list as gm_build_item_list,
    _extract_gold_from_solution as gm_extract_gold_from_solution,
    _normalize_latex_answer as gm_normalize_latex_answer,
)


def _build_prompt(tok, problem: str, prompt_style: str, *, csqa_labels=None, csqa_texts=None) -> str:
    """Match the prompt construction used elsewhere in this repo.

    - 'zst': single user turn + boxed instruction (Zero_Step style).
    - 'ours': system+<think>/<answer> via probing_utils.make_prompt.
    """
    from transformers import AutoTokenizer  # type: ignore

    if prompt_style == "zst":
        if csqa_labels is not None and csqa_texts is not None:
            # CSQA format
            lines = ["Question: " + (problem or "").strip(), "Choices:"]
            for lab, txt in zip(csqa_labels or [], csqa_texts or []):
                lines.append(f"{lab}. {txt}")
            lines.append(
                "Please think step by step, then give your final answer as \\boxed{<LETTER>} "
                "where <LETTER> is one of A,B,C,D,E."
            )
            user = "\n".join(lines)
        else:
            user = (problem or "") + "\nPlease reason step by step, and put your final answer within \\boxed{}."
        try:
            # tok here is the HF tokenizer from load_llm
            enc_prompt = tok.apply_chat_template(
                [{"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            enc_prompt = f"Question: {problem}\nAnswer: "
        return enc_prompt
    else:
        # 'ours' prompt builder from local probing utils
        from lynx.probing_utils import make_prompt

        return make_prompt(tok, problem)


def _load_dataset(args, rng: np.random.Generator):
    """Mirror build_dataset.py dataset selection but return problems/answers/indices."""
    problems: List[str]
    answers: List[str]
    csqa_choices: Optional[Dict[int, Tuple[List[str], List[str]]]] = None
    # Global Hendrycks MATH (EleutherAI/hendrycks_math): defer to the same subset/offset
    # logic as build_global_math_dataset.py, but flatten into a single problems/answers list.
    if str(args.dataset_id) == "EleutherAI/hendrycks_math":
        try:
            import datasets as hf_datasets  # type: ignore
        except Exception as e:
            raise RuntimeError("Please install 'datasets' to use EleutherAI/hendrycks_math.") from e
        subsets = [s.strip() for s in (args.subsets or "").split(",") if s.strip()]
        if not subsets:
            subsets = [
                "algebra",
                "counting_and_probability",
                "geometry",
                "intermediate_algebra",
                "number_theory",
                "prealgebra",
                "precalculus",
            ]
        items = gm_build_item_list(
            args.dataset_id,
            subsets,
            int(args.limit_per_subset),
            rng,
            int(args.subset_offset),
        )
        # Group by subset for efficient HF loading
        by_subset: Dict[str, List[int]] = {}
        for name, idx in items:
            by_subset.setdefault(name, []).append(int(idx))
        problems = []
        answers = []
        for name in subsets:
            sel = by_subset.get(name, [])
            if not sel:
                continue
            ds = hf_datasets.load_dataset(args.dataset_id, name, split="train")
            for idx in sel:
                row = ds[int(idx)]
                problem = row.get("problem") or row.get("question") or ""
                sol_text = str(row.get("solution", ""))
                ans_field = row.get("answer")
                gold_raw = (
                    ans_field
                    if isinstance(ans_field, str) and ans_field.strip()
                    else gm_extract_gold_from_solution(sol_text)
                )
                problem = str(problem)
                gold = gm_normalize_latex_answer(str(gold_raw))
                problems.append(problem)
                answers.append(gold)
        total = len(problems)
        indices = list(range(total))
        dataset_tag = "globalmath"
        split_tag = args.split
        return problems, answers, indices, None, dataset_tag, split_tag

    if args.use_math500:
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
        start = max(0, int(args.split_offset))
        end = min(total, start + int(args.problems))
        indices = sorted(perm[start:end])
        dataset_tag = "math500"
        split_tag = args.split
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

        split_name = args.dataset_split or "train"
        ds = hf_datasets.load_dataset("openai/gsm8k", "main", split=split_name)
        problems = [row.get("question", "") for row in ds]
        answers = [_extract_gsm8k_gold(row.get("answer", "")) for row in ds]
        total = len(problems)
        perm = rng.permutation(total).tolist()
        start = max(0, int(args.split_offset))
        end = min(total, start + int(args.problems))
        indices = sorted(perm[start:end])
        dataset_tag = "gsm8k"
        split_tag = args.split
    elif args.use_csqa:
        try:
            import datasets as hf_datasets  # type: ignore
        except Exception as e:
            raise RuntimeError("Please install 'datasets' to use --use-csqa.") from e
        def _format_csqa(q: str, labels: List[str], texts: List[str]) -> str:
            lines = ["Question: " + (q or "").strip(), "Choices:"]
            for lab, txt in zip(labels, texts):
                lines.append(f"{lab}. {txt}")
            lines.append(
                "Please think step by step, then give your final answer as \\boxed{<LETTER>} where <LETTER> is one of A,B,C,D,E."
            )
            return "\n".join(lines)

        split_name = args.dataset_split or "train"
        ds = hf_datasets.load_dataset("tau/commonsense_qa", "default", split=split_name)
        problems = []
        answers = []
        choices_labels: List[List[str]] = []
        choices_texts: List[List[str]] = []
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
        start = max(0, int(args.split_offset))
        end = min(total, start + int(args.problems))
        indices = sorted(perm[start:end])
        dataset_tag = "csqa"
        split_tag = args.split
        csqa_choices = {i: (choices_labels[i], choices_texts[i]) for i in range(total)}
    else:
        # AIME-style
        problems, answers = load_aime_dataset(args.dataset_id)
        total = len(problems)
        perm = rng.permutation(total).tolist()
        if args.split == "train":
            indices = sorted(perm[: args.problems])
            split_tag = "train"
        elif args.split == "calib":
            start = args.problems
            indices = sorted(perm[start : start + args.problems])
            split_tag = "calib"
        else:
            indices = sorted(perm[-args.problems :])
            split_tag = "test"
        dataset_tag = "aime"

    return problems, answers, indices, csqa_choices, dataset_tag, split_tag


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("Hybrid vLLM+HF builder for early-exit probe datasets")
    # Model / vLLM config
    ap.add_argument("--model-id", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="float16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto", help="HF device-map for forward pass (e.g., 'cuda:0', 'auto').")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--tensor-parallel-size", type=int, default=None)
    ap.add_argument("--model-context-len", type=int, default=None,
                    help="Model context length for vLLM (defaults to max_new_tokens+8000).")

    # Dataset selection
    ap.add_argument("--dataset-id", default="Maxwell-Jia/AIME_2024")
    ap.add_argument("--split", choices=["train", "calib", "test"], required=True)
    ap.add_argument("--use-math500", action="store_true")
    ap.add_argument("--use-gsm8k", action="store_true")
    ap.add_argument("--use-csqa", action="store_true")
    ap.add_argument("--dataset-split", type=str, default=None,
                    help="HF split when using --use-gsm8k or --use-csqa")
    ap.add_argument("--split-offset", type=int, default=0,
                    help="Start index into the shuffled dataset before taking --problems items (non-global datasets)")
    # Global Hendrycks MATH options (mirroring build_global_math_dataset.py)
    ap.add_argument(
        "--subsets",
        type=str,
        default="algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus",
        help="Comma-separated Hendrycks MATH subset names when using EleutherAI/hendrycks_math",
    )
    ap.add_argument("--limit-per-subset", type=int, default=-1,
                    help="Max problems per subset for EleutherAI/hendrycks_math (<=0 means all)")
    ap.add_argument(
        "--subset-offset",
        type=int,
        default=0,
        help="Offset into each subset's permutation before taking --limit-per-subset items (global MATH only)",
    )
    ap.add_argument("--problems", type=int, default=20)
    ap.add_argument("--rollouts", type=int, default=20)
    ap.add_argument("--events-per-rollout", type=int, default=30)
    ap.add_argument("--skip-first-hmmwait", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)

    # Baseline generation (long) via vLLM
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)

    # Exit generation (short, HF)
    ap.add_argument("--exit-max-new-tokens", type=int, default=64)
    ap.add_argument("--exit-temperature", type=float, default=0.0)
    ap.add_argument("--exit-top-p", type=float, default=1.0)
    ap.add_argument("--exit-cue-variant", choices=["paper", "answer_tag"], default="paper")

    # Output
    ap.add_argument("--out-dir", type=Path, default=Path("outputs/vllm_pipeline"))

    # Sharding
    ap.add_argument("--total-shards", type=int, default=1)
    ap.add_argument("--shard-rank", type=int, default=0)
    ap.add_argument("--spawn-workers", action="store_true")

    # Prompt style: ours vs zst
    ap.add_argument("--prompt-style", choices=["ours", "zst"], default="zst")

    return ap.parse_args()


def _infer_tp(user_tp: Optional[int]) -> int:
    if isinstance(user_tp, int) and user_tp >= 1:
        return int(user_tp)
    env_devs = os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("NVIDIA_VISIBLE_DEVICES")
    if env_devs and env_devs not in ("", "all"):
        try:
            n = len([d for d in env_devs.split(",") if d.strip() not in ("", "-1")])
            if n >= 1:
                return n
        except Exception:
            pass
    try:
        n = torch.cuda.device_count()
        if n and n >= 1:
            return int(n)
    except Exception:
        pass
    return 1


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # Save CLI configuration alongside outputs for reproducibility.
    try:
        save_args_json(args, args.out_dir / f"build_dataset_vllm_{args.split}_config.json")
    except Exception:
        pass

    # dtype fallback
    if args.dtype == "bfloat16" and torch.cuda.is_available():
        try:
            if not torch.cuda.is_bf16_supported():
                print("[warn] GPU does not support bfloat16; using float16.")
                args.dtype = "float16"
        except Exception:
            args.dtype = "float16"

    rng = np.random.default_rng(args.seed)

    # Optional local spawning across shards
    if args.total_shards > 1 and args.spawn_workers:
        procs = []
        script = Path(__file__).resolve()
        parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        toks = [t.strip() for t in parent_visible.split(",") if t.strip()] if parent_visible else []
        for r in range(args.total_shards):
            child_env = os.environ.copy()
            gpu_token = toks[r % len(toks)] if toks else str(r)
            child_env["CUDA_VISIBLE_DEVICES"] = gpu_token
            py = sys.executable or "python"
            cmd = [
                py,
                str(script),
                "--model-id", args.model_id,
                "--dtype", args.dtype,
                "--device-map", args.device_map,
                "--split", args.split,
                "--split-offset", str(args.split_offset),
                "--problems", str(args.problems),
                "--rollouts", str(args.rollouts),
                "--events-per-rollout", str(args.events_per_rollout),
                "--skip-first-hmmwait", str(args.skip_first_hmmwait),
                "--seed", str(args.seed),
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
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
            ]
            # Propagate dataset flags, including global MATH options.
            if args.use_math500:
                cmd.append("--use-math500")
            if args.use_gsm8k:
                cmd.append("--use-gsm8k")
                if args.dataset_split:
                    cmd.extend(["--dataset-split", args.dataset_split])
            if args.use_csqa:
                cmd.append("--use-csqa")
                if args.dataset_split:
                    cmd.extend(["--dataset-split", args.dataset_split])
            if not (args.use_math500 or args.use_gsm8k or args.use_csqa):
                cmd.extend(["--dataset-id", args.dataset_id])
                # Global Hendrycks MATH: propagate subset config
                if str(args.dataset_id) == "EleutherAI/hendrycks_math":
                    cmd.extend([
                        "--subsets", args.subsets,
                        "--limit-per-subset", str(args.limit_per_subset),
                        "--subset-offset", str(args.subset_offset),
                    ])
            print(f"[spawn] shard {r}/{args.total_shards}: {' '.join(cmd)}  (CUDA_VISIBLE_DEVICES={child_env.get('CUDA_VISIBLE_DEVICES')})")
            procs.append(subprocess.Popen(cmd, env=child_env))
        rc = 0
        for p in procs:
            rc |= p.wait()
        raise SystemExit(rc)

    # Load dataset indices
    problems, answers, all_indices, csqa_choices, dataset_tag, split_tag = _load_dataset(args, rng)

    # Shard indices if needed
    if args.total_shards > 1:
        indices = [idx for j, idx in enumerate(all_indices) if j % args.total_shards == args.shard_rank]
        print(f"[{split_tag}] Problems shard {args.shard_rank}/{args.total_shards}: {indices}")
    else:
        indices = all_indices
        print(f"[{split_tag}] Problems: {indices}")

    # vLLM engine for baseline decode
    max_model_len = args.model_context_len or (args.max_new_tokens + 8000)
    tp = _infer_tp(args.tensor_parallel_size)
    print(f"[vllm] tensor_parallel_size={tp}, max_model_len={max_model_len}")

    llm = LLM(
        model=args.model_id,
        dtype=args.dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        tensor_parallel_size=tp,
    )

    # HF model for hidden-state forward + short exit completions
    hf_model, tok = load_llm(args.model_id, device_map=args.device_map, torch_dtype=args.dtype)

    # Decide feature layers based on model depth / id (reuse the convention from main pipeline).
    # - QwQ-32B (64 layers): (16,32,48,last)
    # - Qwen3-8B (36 layers): (12,18,24,last)
    # - Nemotron 8B (32 layers): (8,16,24,last)
    # - otherwise: (9,16,last).
    try:
        n_layers = int(getattr(hf_model.config, "num_hidden_layers", 0))
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
    print(f"[features] Using layers {feature_layers} + last (L={n_layers}) for model {args.model_id}")

    # Sampling params for vLLM baseline
    base_sp = SamplingParams(
        max_tokens=int(args.max_new_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        seed=int(args.seed),
    )

    feats: List[np.ndarray] = []
    labels: List[int] = []
    metas: List[dict] = []

    for idx in tqdm(indices, desc="problems"):
        problem = problems[idx]
        gold = answers[idx]

        csqa_labs = csqa_texts = None
        if args.use_csqa and csqa_choices is not None and idx in csqa_choices:
            csqa_labs, csqa_texts = csqa_choices[idx]

        for r in range(args.rollouts):
            # 1) Baseline decode with vLLM
            prompt = _build_prompt(tok, problem, args.prompt_style, csqa_labels=csqa_labs, csqa_texts=csqa_texts)
            outs = llm.generate([prompt], base_sp, use_tqdm=False)
            if not outs or not outs[0].outputs:
                base_text = prompt  # degenerate
            else:
                text = outs[0].outputs[0].text or ""
                base_text = prompt + text

            # 2) HF tokenize and forward once for hidden states
            enc_prompt = _build_prompt(tok, problem, args.prompt_style, csqa_labels=csqa_labs, csqa_texts=csqa_texts)
            enc_len = tok(enc_prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].shape[1]
            seq_ids = tok(base_text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].tolist()

            try:
                _, hidden_states = forward_hidden_and_logits(hf_model, seq_ids)
            except Exception as e:
                print(f"[warn] forward failed for index={idx} rollout={r}: {e}")
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

            for (pos_rel, lab) in pairs:
                # Feature vector
                fv = build_event_features(
                    hidden_states=hidden_states,
                    enc_len=enc_len,
                    pos_rel=pos_rel,
                    layers=feature_layers,
                )

                # Force early exit at this event and label by correctness
                prefix = make_exit_prompt_prefix(tok, seq_ids, enc_len, pos_rel, variant=args.exit_cue_variant)
                early_text, added_tokens = generate_from_prefix(
                    hf_model,
                    tok,
                    prefix,
                    max_new_tokens=args.exit_max_new_tokens,
                    temperature=args.exit_temperature,
                    top_p=args.exit_top_p,
                    do_sample=False,
                )
                if args.use_csqa:
                    labs, txts = csqa_choices.get(idx, (None, None))  # type: ignore
                    y_event = is_correct_answer(early_text, gold, mcq_labels=labs, mcq_texts=txts)
                else:
                    # Math-style tasks: use tolerant ZST-style checker for robust labels.
                    y_event = is_correct_answer_with_policy(early_text, gold, policy="zst")

                feats.append(fv)
                labels.append(y_event)
                metas.append(
                    {
                        "index": idx,
                        "rollout": r,
                        "pos_rel": pos_rel,
                        "event": lab,
                        "added_tokens": int(added_tokens),
                        "label": int(y_event),
                    }
                )

    if not feats:
        raise RuntimeError("No features collected; check event detection / generation settings.")

    X = np.stack(feats, axis=0)
    y = np.asarray(labels, dtype=np.int64)

    shard_suffix = (
        f"_shard{args.shard_rank}of{args.total_shards}" if args.total_shards and args.total_shards > 1 else ""
    )
    out_npz = args.out_dir / f"{dataset_tag}_{split_tag}_dataset{shard_suffix}.npz"
    out_meta = args.out_dir / f"{dataset_tag}_{split_tag}_meta{shard_suffix}.jsonl"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, X=X, y=y)
    with out_meta.open("w", encoding="utf-8") as fh:
        for row in metas:
            fh.write(json.dumps(row) + "\n")
    print(f"Saved: {out_npz}  (X={X.shape}, positives={y.sum()}, negatives={(y==0).sum()})")
    print(f"Meta:  {out_meta}")


if __name__ == "__main__":
    main()
