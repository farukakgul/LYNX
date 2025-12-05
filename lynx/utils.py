from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import os
import importlib.util

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fractions import Fraction

# Ensure repo root on sys.path so that `import lynx` works when running
# module files directly as scripts (e.g., `python lynx/early_exit_eval.py`).
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse utilities from the local probing module
from lynx.probing_utils import (
    GenConfig,
    forward_hidden_and_logits,
    gather_hidden_at_positions,
    is_correct_final,
    load_aime_dataset,
    load_llm,
)
# No PRM/DeepConf in this module per updated design

"""
Built‑in tolerant math grader (SymPy‑style)
- We implement a compact version of the ZST/GSM8K tolerant checker
  so users do not need to pull external repos. If SymPy is unavailable,
  we silently fall back to our numeric parser.
"""
try:
    import sympy
    from sympy.parsing import sympy_parser
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

_DEER_EQ_FUNCS = None  # lazy cache for DEER-style normalization / equivalence


def _load_deer_equivalence_funcs():
    """Best-effort loader for DEER's math normalization/equivalence helpers.
    Returns a dict with {strip_string, normalize_final_answer, check_sympy_equivalence}
    or None if the DEER repo is unavailable.
    """
    global _DEER_EQ_FUNCS
    if _DEER_EQ_FUNCS is not None:
        return _DEER_EQ_FUNCS
    try:
        root = Path(__file__).resolve().parents[2] / "DEER" / "utils"
        m_norm = root / "math_normalization.py"
        m_parser = root / "parser.py"
        if not (m_norm.exists() and m_parser.exists()):
            _DEER_EQ_FUNCS = None
            return None
        # Load modules by explicit path to avoid package-name collisions
        spec_norm = importlib.util.spec_from_file_location("deer_math_normalization", m_norm)
        norm_mod = importlib.util.module_from_spec(spec_norm)  # type: ignore[arg-type]
        assert spec_norm.loader is not None
        spec_norm.loader.exec_module(norm_mod)  # type: ignore[call-arg]
        spec_parser = importlib.util.spec_from_file_location("deer_parser", m_parser)
        parser_mod = importlib.util.module_from_spec(spec_parser)  # type: ignore[arg-type]
        assert spec_parser.loader is not None
        spec_parser.loader.exec_module(parser_mod)  # type: ignore[call-arg]
        _DEER_EQ_FUNCS = {
            "strip_string": getattr(parser_mod, "strip_string", None),
            "normalize_final_answer": getattr(norm_mod, "normalize_final_answer", None),
            "check_sympy_equivalence": getattr(norm_mod, "check_sympy_equivalence", None),
        }
        # Guard against missing attributes
        if not all(_DEER_EQ_FUNCS.values()):
            _DEER_EQ_FUNCS = None
    except Exception:
        _DEER_EQ_FUNCS = None
    return _DEER_EQ_FUNCS

def _last_boxed_only_string(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    idx = s.rfind("\\boxed")
    if idx < 0:
        idx = s.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    left = None
    right = None
    bal = 0
    while i < len(s):
        ch = s[i]
        if ch == "{":
            bal += 1
            if left is None:
                left = i
        elif ch == "}":
            bal -= 1
            if bal == 0:
                right = i
                break
        i += 1
    if left is None or right is None:
        return None
    return s[left + 1 : right].strip()

def _latex_to_plain(expr: str) -> str:
    if not isinstance(expr, str):
        return ""
    expr = expr.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    # \frac{a}{b} -> (a)/(b)
    expr = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", expr)
    # \sqrt{...} -> sqrt(...)
    expr = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", expr)
    # Remove latex sizing/spaces
    expr = re.sub(r"\\left|\\right", "", expr)
    expr = re.sub(r"\\(?:,|;|:|!|\s+)", "", expr)
    # Symbols
    expr = expr.replace("×", "*").replace("·", "*").replace("\\times", "*")
    expr = expr.replace("π", "pi").replace("\\pi", "pi")
    # Power
    expr = expr.replace("^", "**")
    return expr

def _normalize_answer_text(expr: str) -> str:
    if not isinstance(expr, str):
        return ""
    expr = expr.strip()
    # Remove $ and % wrappers and trailing punctuation
    expr = expr.replace("$", "").replace("%", "")
    expr = expr.strip().rstrip(". )]")
    # Handle common words
    expr = expr.replace(" or ", " , ").replace(" and ", " , ")
    # words to multipliers
    expr = expr.replace("million", "*10^6").replace("billion", "*10^9").replace("trillion", "*10^12")
    expr = _latex_to_plain(expr)
    return expr

def _split_tuple(expr: str) -> List[str]:
    if not isinstance(expr, str):
        return []
    e = expr.strip()
    if len(e) >= 2 and e[0] in "()[]" and e[-1] in "()[]" and all(ch not in e[1:-1] for ch in "()[]"):
        return [t.strip() for t in e[1:-1].split(",")]
    return [e]

def _sympy_equal(a: str, b: str) -> bool:
    if not _HAS_SYMPY:
        return False
    try:
        # Implicit multiplication, caret -> power
        def parse(x: str):
            x = x.replace("^", "**")
            return sympy_parser.parse_expr(
                x,
                transformations=(
                    sympy_parser.standard_transformations
                    + (sympy_parser.implicit_multiplication_application,)
                ),
            )
        diff = parse(f"({a})-({b})")
        return bool(sympy.simplify(diff) == 0)
    except Exception:
        return False

def _clean_answer_zst_style(model_pred: str) -> str:
    # Prefer last \boxed{..}
    m = _last_boxed_only_string(model_pred or "")
    if m is not None:
        return m
    txt = (model_pred or "").lower()
    # Look for answer triggers
    for trig in ["therefore, the answer is", "the answer is", "answer is", "answer:"]:
        pos = txt.rfind(trig)
        if pos >= 0:
            return model_pred[pos + len(trig) :].strip()
    # Fallback: last number in the tail
    nums = re.findall(r"-?\d+\.?\d*", model_pred or "")
    if nums:
        return nums[-1]
    return (model_pred or "").strip()

def _extract_numeric_with_percent(text: str) -> Tuple[Optional[float], bool]:
    """Extract a numeric value from text and a flag indicating if it was written with %.
    Designed to robustly handle things like '10', '10%', '3/4', '0.75', '12 students'.
    """
    if not isinstance(text, str) or not text:
        return None, False
    s = text.strip()
    has_pct = ("%" in s)
    # Grab last numeric token (integer, fraction, or decimal)
    m = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", s)
    if not m:
        return None, has_pct
    token = m[-1]
    try:
        if "/" in token:
            val = float(Fraction(token))
        else:
            val = float(token)
    except Exception:
        return None, has_pct
    return val, has_pct


def _normalize_inf_token(s: str) -> str:
    """Map common infinity notations to canonical 'inf' / '-inf' tokens."""
    if not isinstance(s, str):
        return ""
    t = s.strip()
    # Common LaTeX and unicode variants
    t = t.replace("\\infty", "inf").replace("∞", "inf")
    t_low = t.lower()
    if t_low in {"inf", "+inf"}:
        return "inf"
    if t_low.startswith(("−inf", "-inf")):  # includes unicode minus
        return "-inf"
    return t


def is_correct_answer_sympy_style(text: str, gold: str) -> int:
    if not _HAS_SYMPY:
        # Fallback to our default numeric/text parser if SymPy is unavailable
        return is_correct_answer(text, gold)
    pred_raw = _clean_answer_zst_style(text)
    g_raw = str(gold or "")
    # Special handling for infinity synonyms (\\infty, ∞, inf, -inf).
    pn_inf = _normalize_inf_token(pred_raw)
    gn_inf = _normalize_inf_token(g_raw)
    if pn_inf in ("inf", "-inf") and pn_inf == gn_inf:
        return 1
    # First, try numeric equality with unit/percentage robustness.
    v_pred, pct_pred = _extract_numeric_with_percent(pred_raw)
    v_gold, pct_gold = _extract_numeric_with_percent(g_raw)
    if v_pred is not None and v_gold is not None:
        # Interpret both as proportions: x% -> x/100, plain x -> x
        val_pred = v_pred / 100.0 if pct_pred else v_pred
        val_gold = v_gold / 100.0 if pct_gold else v_gold
        if abs(val_pred - val_gold) <= 1e-4:
            return 1
    # quick string match after normalization
    pn = _normalize_answer_text(pred_raw)
    gn = _normalize_answer_text(g_raw)
    if pn == gn and pn != "":
        return 1
    # Tuple/intervals: compare elementwise under sympy when possible
    p_elems = _split_tuple(pn)
    g_elems = _split_tuple(gn)
    if len(p_elems) == len(g_elems) and len(g_elems) > 0:
        ok = True
        for pe, ge in zip(p_elems, g_elems):
            if pe == ge and pe != "":
                continue
            if not _sympy_equal(pe, ge):
                ok = False
                break
        return int(ok)
    # Fall back to sympy equality on the whole string
    if _sympy_equal(pn, gn):
        return 1

    # As an extra robustness layer for AMC/AIME/MATH/GSM8K-style answers,
    # optionally leverage DEER's normalization + SymPy equivalence when available.
    funcs = _load_deer_equivalence_funcs()
    if funcs is not None:
        try:
            strip = funcs["strip_string"]
            norm = funcs["normalize_final_answer"]
            chk = funcs["check_sympy_equivalence"]
            if strip and norm and chk:
                pred_n = norm(strip(pred_raw))
                gold_n = norm(strip(g_raw))
                # quick equality after DEER-style normalization
                if pred_n and gold_n and pred_n == gold_n:
                    return 1
                # SymPy-based equivalence on the normalized forms
                if chk(pred_n, gold_n):
                    return 1
        except Exception:
            # If DEER helpers fail for any reason, just fall back to 0
            pass
    return 0


# As in the referenced paper, force an exit by closing the think block
# and prompting the model to emit a boxed final answer.
def get_exit_cue(variant: str = "paper") -> str:
    """Return the answer inducer string.
    - 'paper':   </think> Final Answer: \boxed{
    - 'answer_tag': </think><answer> The final answer is \boxed{
    """
    if variant == "answer_tag":
        return "</think><answer> The final answer is \\boxed{"
    return "</think> Final Answer: \\boxed{"


def build_event_features(
    *,
    hidden_states: Tuple[torch.Tensor, ...],
    enc_len: int,
    pos_rel: int,
    layers: Sequence[int] = (9, 16),
) -> np.ndarray:
    """Construct the feature vector at a given event position using only hidden states.
    Concatenate selected intermediate layers and the last layer.
    By default we use (9,16,last) as in the original 1.5B setup.
    Call sites may override `layers` to adapt to different model depths (e.g., Qwen3-8B).
    """
    abs_pos = enc_len + (pos_rel - 1)
    last_layer = len(hidden_states) - 1
    layer_list = list(layers) + [last_layer]
    H = gather_hidden_at_positions(hidden_states, positions_abs=[abs_pos], layers_1based=layer_list)[0]
    return H.astype(np.float32)


def predictive_set_binary(p_pos: float, q_pos: float, q_neg: float) -> List[int]:
    """Split conformal predictive set for binary classification.
    Include class c if (1 - p_c) <= q_c, where p_neg = 1 - p_pos.
    Returns a list subset of {0,1}.
    """
    p_pos = float(np.clip(p_pos, 0.0, 1.0))
    p_neg = 1.0 - p_pos
    S: List[int] = []
    if (1.0 - p_pos) <= q_pos:
        S.append(1)
    if (1.0 - p_neg) <= q_neg:
        S.append(0)
    return S


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    """Finite-sample split conformal quantile.
    scores are nonconformity values; larger means more nonconformity.
    Returns q such that P(score <= q) >= 1 - delta.
    """
    n = int(len(scores))
    if n <= 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - delta)))
    k = max(1, min(k, n))
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def make_exit_prompt_prefix(tok: AutoTokenizer, seq_ids: List[int], enc_len: int, pos_rel: int, *, variant: str = "paper") -> str:
    """Decode the assistant text up to (but not including) the event token and append the exit cue.
    This inserts the cue right before the "hmm/wait" token, aligning with the paper's intervention.
    """
    abs_pos = enc_len + (pos_rel - 1)
    prefix_ids = seq_ids[: abs_pos]
    prefix_text = tok.decode(prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return prefix_text + get_exit_cue(variant)


@torch.inference_mode()
def generate_from_prefix(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prefix_text: str,
    *,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> Tuple[str, int]:
    enc = tok(prefix_text, return_tensors="pt", add_special_tokens=False)
    for k in enc:
        enc[k] = enc[k].to(model.device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        return_dict_in_generate=True,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )
    seq = out.sequences[0]
    gen_ids = seq[enc["input_ids"].shape[1] :].tolist()
    text = tok.decode(seq, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return text, len(gen_ids)


# ====== Non-math answer parsing for Multiple-Choice QA (e.g., CommonsenseQA) ======

_MCQ_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Ea-e])\s*\}")
_MCQ_LETTER_RE = re.compile(r"(?:answer\s*(?:is|:)?\s*|option\s*|choice\s*|final\s*answer\s*(?:is|:)?)\s*\(?\s*([A-Ea-e])\s*\)?", re.IGNORECASE)


def parse_mcq_answer(text: str, *, labels: Optional[List[str]] = None, choices_text: Optional[List[str]] = None) -> Optional[str]:
    """Extract a predicted option letter from free text.
    Priorities:
      1) \\boxed{C}
      2) phrases like 'Answer is C', '(C)', 'Option C'
      3) bare standalone letter near the end
      4) fallback by matching choice text substring -> corresponding label
    Returns uppercase letter A-E or None.
    """
    if not isinstance(text, str) or not text:
        return None
    tail = text[-800:]
    m = _MCQ_BOXED_RE.search(tail)
    if m:
        return m.group(1).upper()
    m2 = _MCQ_LETTER_RE.search(tail)
    if m2:
        return m2.group(1).upper()
    # Standalone letter near end
    m3 = re.search(r"\b([A-Ea-e])\b", tail)
    if m3:
        return m3.group(1).upper()
    # Fallback: match choice text substrings
    if labels and choices_text:
        S = tail.lower()
        best = None
        for lab, txt in zip(labels, choices_text):
            if isinstance(txt, str) and txt and txt.lower() in S:
                best = lab.upper()
        if best:
            return best
    return None


def is_correct_answer(
    text: str,
    gold: str,
    *,
    mcq_labels: Optional[List[str]] = None,
    mcq_texts: Optional[List[str]] = None,
) -> int:
    """Dataset-aware correctness check.
    - If mcq_labels provided, treat as MCQ and compare letters.
    - Else, fall back to numeric/textual parser is_correct_final (math/AIME/GSM8K).
    """
    if mcq_labels is not None:
        pred = parse_mcq_answer(text, labels=mcq_labels, choices_text=mcq_texts)
        if pred is None:
            return 0
        return int(pred.strip().upper() == str(gold).strip().upper())
    # Math/numeric pathway
    return int(is_correct_final(text, gold))


# ====== Alternative (DEER-style) math checker ======

_LATEX_SPACE_RE = re.compile(r"\\(?:,|;|:|!|\s+)")


def _normalize_latex_simple(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().strip("$")
    # \frac or \dfrac -> a/b
    s = re.sub(r"\\d?frac\{\s*([^{}]+?)\s*\}\{\s*([^{}]+?)\s*\}", r"\1/\2", s)
    # Remove common formatting tokens
    s = re.sub(r"\\left|\\right", "", s)
    s = _LATEX_SPACE_RE.sub("", s)
    # Drop thousands separators and spaces
    s = s.replace(",", "").strip().rstrip(". )]")
    return s


def _to_fraction_or_float(s: str) -> Optional[Fraction]:
    try:
        return Fraction(s)
    except Exception:
        try:
            return Fraction(float(s))
        except Exception:
            return None


def is_correct_answer_deer_style(text: str, gold: str) -> int:
    """Looser math checker approximating DEER's policy:
    - Prefer last \boxed{...} in the prediction; fallback to last numeric token.
    - Normalize simple LaTeX and compare numbers via exact Fraction equality when possible.
    - Else compare normalized strings.
    """
    if not isinstance(text, str):
        return 0
    if not isinstance(gold, str):
        gold = str(gold or "")

    # Extract candidate from pred: last boxed or last number
    tail = text[-4000:]
    boxed = re.findall(r"\\boxed\{([^}]*)\}", tail)
    if boxed:
        pred_raw = boxed[-1].strip()
    else:
        nums = re.findall(r"[-+]?\d+(?:/\d+)?(?:\.\d+)?", tail)
        pred_raw = nums[-1] if nums else tail.strip()

    pred_n = _normalize_latex_simple(pred_raw)
    gold_n = _normalize_latex_simple(gold)
    if not pred_n or not gold_n:
        return 0

    a = _to_fraction_or_float(pred_n)
    b = _to_fraction_or_float(gold_n)
    if a is not None and b is not None:
        return int(a == b)
    return int(pred_n == gold_n)


def is_correct_answer_with_policy(
    text: str,
    gold: str,
    *,
    policy: str = "strict",
    mcq_labels: Optional[List[str]] = None,
    mcq_texts: Optional[List[str]] = None,
) -> int:
    if mcq_labels is not None:
        return is_correct_answer(text, gold, mcq_labels=mcq_labels, mcq_texts=mcq_texts)
    if policy in ("zst", "zst_math"):
        # Use built-in SymPy-style tolerant checker.
        return is_correct_answer_sympy_style(text, gold)
    if policy == "deer":
        return is_correct_answer_deer_style(text, gold)
    return is_correct_answer(text, gold)


__all__ = [
    "get_exit_cue",
    "build_event_features",
    "predictive_set_binary",
    "conformal_quantile",
    "make_exit_prompt_prefix",
    "generate_from_prefix",
    "parse_mcq_answer",
    "is_correct_answer",
    "is_correct_answer_with_policy",
    "is_correct_answer_deer_style",
]


def save_args_json(args: Any, path: Path) -> None:
    """Utility: save an argparse.Namespace (or similar) as JSON.

    Paths are stringified so that configs are portable across machines.
    """
    try:
        obj = {}
        for k, v in vars(args).items():
            if isinstance(v, Path):
                obj[k] = str(v)
            else:
                obj[k] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        # Config saving is best-effort; never crash the main pipeline.
        pass
