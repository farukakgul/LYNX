from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
import os as _os
from fractions import Fraction
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
# Text-only: avoid torchvision import through Transformers on Qwen2/etc.
_os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional PRM utilities from the separate `replicate` repo.
# These are only needed for PRM-style step scoring; the core LYNX
# early-exit pipeline (dataset/probe/conformal/eval) does NOT depend on them.
try:  # pragma: no cover - optional dependency
    from replicate.reward_processing import (
        breakdown_reasoning,
        get_reward_trajectory,
        load_reward_model,
        parse_device_map_arg,
    )
except Exception:  # pragma: no cover
    breakdown_reasoning = None           # type: ignore[assignment]
    get_reward_trajectory = None         # type: ignore[assignment]
    load_reward_model = None             # type: ignore[assignment]
    parse_device_map_arg = None          # type: ignore[assignment]


# ====== Prompt / answer helpers ======

SYSTEM_PROMPT = (
    "You are a helpful assistant. Before providing an answer to the user's query, "
    "you first engage in a structured step-by-step reasoning process, which you "
    "enclose within <think> </think> tags. Once you reach a conclusion, you present "
    "your final response within<answer> </answer> tags, ensuring clarity, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>. "
    "In your answer, briefly summarize your thought process before stating the final "
    "result, which you format using \\boxed{} for emphasis, e.g., "
    "<answer> I solve the problem by ... The final answer is \\boxed{ANSWER} </answer>."
)
MSG_TEMPLATE = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "{problem}"},
]
# Robust answer parsing patterns (aligned with utils-helpers/eval_batch_accuracy.py)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
NUM_TOKEN = r"[-+]?\d+(?:/\d+)?(?:\.\d+)?"
NUM_RE = re.compile(NUM_TOKEN)
FINAL_PATTERNS = [
    re.compile(rf"(?:final\s*answer\s*(?:is|:)?|answer\s*(?:is|:)?|result\s*(?:is|:)?)\s*({NUM_TOKEN})", re.IGNORECASE),
    re.compile(rf"(?:therefore|thus|hence|so)[^\n]{{0,40}}?({NUM_TOKEN})", re.IGNORECASE),
]


def make_prompt(tok: AutoTokenizer, problem: str) -> str:
    msgs = [
        MSG_TEMPLATE[0],
        {"role": "user", "content": MSG_TEMPLATE[1]["content"].format(problem=problem)},
    ]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        sys = msgs[0]["content"]; usr = msgs[1]["content"]
        return f"<|system|>\n{sys}\n<|user|>\n{usr}\n<|assistant|>\n"


def _simplify_latex_numbers(s: str) -> str:
    s = re.sub(r"\\d?frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}", r"\1/\2", s)
    s = re.sub(r"\\(?:,|;|:|!|\s+)", "", s)
    s = re.sub(r"\\left|\\right", "", s)
    return s


def _to_fraction_if_numeric(s: str) -> Optional[Fraction]:
    s = s.strip().replace("\u2212", "-")
    s = s.strip().rstrip(".,;: )]").lstrip("([")
    if not s:
        return None
    m = NUM_RE.fullmatch(s)
    if not m:
        return None
    try:
        return Fraction(s)
    except Exception:
        try:
            return Fraction(float(s))
        except Exception:
            return None


def _equal_answer(gold: str, cand: str) -> bool:
    def norm_text(t: str) -> str:
        t = t.strip()
        if t.startswith('"') and t.endswith('"') and len(t) >= 2:
            t = t[1:-1].strip()
        return t
    gold_n = _to_fraction_if_numeric(norm_text(gold))
    cand_n = _to_fraction_if_numeric(norm_text(cand))
    if gold_n is not None and cand_n is not None:
        return gold_n == cand_n
    return norm_text(gold) == norm_text(cand)


def _extract_candidate_answers(full_text: str) -> List[str]:
    cands: List[str] = []
    # 1) All boxed occurrences
    for m in BOXED_RE.finditer(full_text):
        val = m.group(1).strip()
        inner = BOXED_RE.search(val)
        if inner:
            val = inner.group(1).strip()
        val = _simplify_latex_numbers(val)
        nums = NUM_RE.findall(val)
        if nums:
            cands.extend(nums)
        else:
            cands.append(val)
    tail = full_text[-4000:]
    # 2) <answer> blocks
    for m in ANSWER_TAG_RE.finditer(tail):
        block = _simplify_latex_numbers(m.group(1))
        boxed_inside = BOXED_RE.findall(block)
        if boxed_inside:
            for val in boxed_inside:
                val = _simplify_latex_numbers(val.strip())
                nums = NUM_RE.findall(val)
                if nums:
                    cands.extend(nums)
                else:
                    cands.append(val)
        else:
            nums = NUM_RE.findall(block)
            if nums:
                cands.extend(nums[-2:])
    # 3) Explicit phrases
    for pat in FINAL_PATTERNS:
        for m in pat.finditer(tail):
            cands.append(m.group(1).strip())
    # 4) Fallback: last few numeric tokens
    nums = NUM_RE.findall(tail)
    if nums:
        cands.extend(nums[-3:])
    # dedup preserve order
    seen = set(); unique = []
    for x in cands:
        if x not in seen:
            seen.add(x); unique.append(x)
    return unique


def is_correct_final(text: str, gold: str) -> int:
    for cand in _extract_candidate_answers(text):
        if _equal_answer(gold, cand):
            return 1
    return 0


# ====== DeepConf (confidence) ======

@torch.inference_mode()
def token_topk_confidences_from_logits(
    logits: torch.Tensor,  # [T, V] (for a single sequence)
    top_k: int,
) -> List[float]:
    conf: List[float] = []
    # log-softmax on the vocab dimension
    logp = logits.float().log_softmax(dim=-1)
    finite = torch.isfinite(logp)
    for i in range(logp.shape[0]):
        row = logp[i]
        row = row[torch.isfinite(row)]
        if row.numel() == 0:
            conf.append(float("nan"))
            continue
        k = min(top_k, row.numel())
        top_lp, _ = torch.topk(row, k=k, dim=-1)
        conf.append(float(-top_lp.mean().item()))  # nats; lower is more confident
    return conf


def trailing_mean(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    s = 0.0
    q: List[float] = []
    for v in values:
        v = float(v) if v == v else 0.0  # NaN guard
        q.append(v)
        s += v
        if len(q) > window:
            s -= q.pop(0)
        out.append(s / len(q))
    return out


def group_confidence(token_conf: List[float], window: int) -> List[float]:
    return trailing_mean(token_conf, window)


# ====== Pattern localization over tokens ======

def find_pattern_positions_by_tokens(
    new_token_ids: List[int], tok: AutoTokenizer, patterns=("hmm", "wait")
) -> Dict[str, List[int]]:
    pieces = [
        tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for tid in new_token_ids
    ]
    cumulative: List[int] = []
    s = ""
    for p in pieces:
        s += p
        cumulative.append(len(s))
    s_lower = s.lower()

    pos_map: Dict[str, List[int]] = {p: [] for p in patterns}
    for pat in patterns:
        L = len(pat)
        start = 0
        while True:
            k = s_lower.find(pat, start)
            if k == -1:
                break
            end_char = k + L
            # locate the first token whose cumulative end >= end_char
            t_idx = next((i + 1 for i, cl in enumerate(cumulative) if cl >= end_char), len(cumulative))
            pos_map[pat].append(t_idx)
            start = k + L
    return pos_map


def decode_new_tokens(tok: AutoTokenizer, seq_ids: List[int], enc_len: int) -> Tuple[List[str], List[int]]:
    """Decode generated tokens piecewise and return pieces + cumulative char ends."""
    new_ids = seq_ids[enc_len:]
    pieces = [tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False) for tid in new_ids]
    cumulative = []
    s = ""
    for p in pieces:
        s += p
        cumulative.append(len(s))
    return pieces, cumulative


# ====== PRM step scores at hmm/wait boundaries ======

def make_hmmwait_spans(tok: AutoTokenizer, new_token_ids: List[int]) -> Tuple[List[int], List[str], List[Tuple[int, int]]]:
    pieces = [tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False) for tid in new_token_ids]
    cumulative: List[int] = []
    s = ""
    for p in pieces:
        s += p
        cumulative.append(len(s))
    s_lower = s.lower()

    positions: List[int] = []  # 1-based token indices within new_token_ids
    labels: List[str] = []
    for pat in ("hmm", "wait", "alternatively"):
        L = len(pat)
        start = 0
        while True:
            k = s_lower.find(pat, start)
            if k == -1:
                break
            end_char = k + L
            t_idx = next((i + 1 for i, cl in enumerate(cumulative) if cl >= end_char), len(cumulative))
            positions.append(t_idx)
            labels.append(pat)
            start = k + L

    # Merge & sort by token index; dedup if collisions
    merged: Dict[int, str] = {}
    for p, lab in zip(positions, labels):
        merged.setdefault(p, lab)
    token_positions = sorted(merged.keys())
    labels_out = [merged[p] for p in token_positions]

    # Convert token positions to char spans (prev char -> current token char end)
    spans: List[Tuple[int, int]] = []
    prev_char = 0
    for p in token_positions:
        end_char = cumulative[p - 1]
        if end_char > prev_char:
            spans.append((prev_char, end_char))
            prev_char = end_char
    if prev_char < len(s):
        spans.append((prev_char, len(s)))
    return token_positions, labels_out, spans


def extract_step_scores(trajectory: List[List[float]], smooth_window: int = 1) -> List[float]:
    raw = []
    for curve in trajectory:
        if curve:
            raw.append(float(curve[-1]))
    # trailing mean for small smoothing
    if smooth_window <= 1:
        return raw
    out: List[float] = []
    for i in range(len(raw)):
        lo = max(0, i - smooth_window + 1)
        out.append(float(np.mean(raw[lo : i + 1])))
    return out


# ====== Model loading ======

@dataclass
class GenConfig:
    max_new_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    do_sample: bool = True


def load_llm(model_id: str, *, device_map: Optional[str] = "auto", torch_dtype: Optional[str] = None):
    dt = None
    if torch_dtype == "bfloat16":
        dt = torch.bfloat16
    elif torch_dtype == "float16":
        dt = torch.float16
    elif torch_dtype == "float32":
        dt = torch.float32

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device_map if torch.cuda.is_available() else None,
        torch_dtype=dt,
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


@torch.inference_mode()
def generate_once(model, tok, problem: str, gcfg: GenConfig) -> Tuple[str, List[int], int]:
    prompt = make_prompt(tok, problem)
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
    return text, seq.tolist(), enc_len


@torch.inference_mode()
def forward_hidden_and_logits(model, seq_ids: List[int]):
    seq = torch.tensor([seq_ids], dtype=torch.long, device=next(model.parameters()).device)
    attn = torch.ones_like(seq, dtype=torch.long)
    out = model(input_ids=seq, attention_mask=attn, output_hidden_states=True)
    # out.logits: [1, T, V] — predicts tokens 1..T-1 from positions 0..T-2
    # out.hidden_states: Tuple[layer0..layerL] each [1, T, H]
    return out.logits[0], out.hidden_states  # [T,V], tuple([1,T,H])


def deepconf_for_generated(logits_TxV: torch.Tensor, enc_len: int, top_k: int, group_size: int) -> Tuple[List[float], List[float]]:
    T = logits_TxV.shape[0]
    start = max(enc_len - 1, 0)
    end = T - 1
    if end <= start:
        return [], []
    sl = logits_TxV[start:end, :]  # predicts tokens enc_len..T-1
    token_conf = token_topk_confidences_from_logits(sl, top_k=top_k)
    group_conf = group_confidence(token_conf, window=group_size)
    return token_conf, group_conf


def gather_hidden_at_positions(hidden_states: Tuple[torch.Tensor, ...], positions_abs: List[int], layers_1based: Sequence[int]) -> np.ndarray:
    # hidden_states length = L+1 (including embedding); index 1..L are transformer blocks
    L = len(hidden_states) - 1
    selected_layers = []
    for li in layers_1based:
        if li <= 0:
            li = 1
        if li > L:
            li = L
        selected_layers.append(li)

    feats: List[np.ndarray] = []
    for li in selected_layers:
        hs = hidden_states[li]  # [1, T, H]
        # collect for each position
        vecs = []
        for pos in positions_abs:
            pos = max(0, min(hs.shape[1] - 1, pos))
            v = hs[0, pos, :].detach().float().cpu().numpy()
            vecs.append(v)
        feats.append(np.stack(vecs, axis=0))  # [P, H]
    # Concatenate along feature dimension
    return np.concatenate(feats, axis=1)  # [P, H_total]


def select_hmmwait_positions(
    pos_map: Dict[str, List[int]], *,
    skip_first: int,
    per_rollout_limit: int,
    strategy: str = "sequential",
    rng: Optional[np.random.Generator] = None,
) -> List[Tuple[int, str]]:
    """
    Select up to `per_rollout_limit` (position,label) pairs after skipping the first N events.
    Strategies:
      - sequential: take the next K in order (default; yields e.g. 6th..10th)
      - uniform:   evenly spaced K samples over the remaining events
      - random:    random K without replacement over the remaining events (seeded if rng provided)
    """
    pairs = [(p, "hmm") for p in pos_map.get("hmm", [])] + [(p, "wait") for p in pos_map.get("wait", [])]
    pairs.sort(key=lambda t: t[0])
    pairs = pairs[skip_first:]
    if per_rollout_limit is None or per_rollout_limit <= 0 or len(pairs) <= per_rollout_limit:
        return pairs

    k = per_rollout_limit
    if strategy == "sequential":
        return pairs[:k]
    elif strategy == "uniform":
        idxs = np.linspace(0, len(pairs) - 1, num=k).round().astype(int)
        # dedup while preserving order
        seen = set(); sel = []
        for i in idxs.tolist():
            if i not in seen:
                sel.append(i); seen.add(i)
        # if dedup shortened length, top up sequentially
        i = 0
        while len(sel) < k and i < len(pairs):
            if i not in seen:
                sel.append(i); seen.add(i)
            i += 1
        return [pairs[i] for i in sel]
    elif strategy == "random":
        if rng is None:
            rng = np.random.default_rng()
        idxs = sorted(rng.choice(len(pairs), size=k, replace=False).tolist())
        return [pairs[i] for i in idxs]
    else:
        # fallback
        return pairs[:k]


def make_prm_scores_at_hmmwait(
    problem: str,
    full_text: str,
    tok: AutoTokenizer,
    enc_len: int,
    seq_ids: List[int],
    *,
    prm_tokenizer, prm_model,
    smooth_window: int = 1,
) -> Tuple[List[int], List[str], List[float]]:
    """
    Returns (token_positions_1based, labels, step_scores) aligned at hmm/wait boundaries.
    token_positions_* are relative to generated tokens (1-based).
    """
    if get_reward_trajectory is None:
        raise RuntimeError(
            "PRM step scoring requires the optional 'replicate' repo "
            "(replicate.reward_processing). Install it or avoid calling "
            "make_prm_scores_at_hmmwait in this configuration."
        )

    new_ids = seq_ids[enc_len:]
    positions, labels, spans = make_hmmwait_spans(tok, new_ids)
    # Extract substring slices for PRM steps
    pieces, cumulative = decode_new_tokens(tok, seq_ids, enc_len)
    s = "".join(pieces)
    steps = [s[a:b] for a, b in spans]
    if not steps:
        return [], [], []
    trajectory = get_reward_trajectory(problem, steps, model=prm_model, tokenizer=prm_tokenizer)
    scores = extract_step_scores(trajectory, smooth_window=smooth_window)
    take = min(len(positions), len(scores))
    return positions[:take], labels[:take], scores[:take]


def slice_with_pad(seq: Sequence[float], idx: int, count_back: int, *, pad_value: float = 0.0) -> List[float]:
    """Take current (idx) and previous count_back items (inclusive of idx). idx is 0-based."""
    out = []
    for k in range(count_back, -1, -1):
        j = idx - k
        if 0 <= j < len(seq):
            out.append(float(seq[j]))
        else:
            out.append(float(pad_value))
    return out


def split_dataset_indices(total: int, train_n: int, seed: int = 42) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total).tolist()
    train = sorted(perm[:train_n])
    test = sorted(perm[train_n:])
    return train, test


def load_aime_dataset(dataset_id: str) -> Tuple[List[dict], List[str]]:
    ds = load_dataset(dataset_id)
    train = ds["train"]
    problems = [train[i]["Problem"] for i in range(len(train))]
    answers = [str(train[i]["Answer"]) for i in range(len(train))]
    return problems, answers
