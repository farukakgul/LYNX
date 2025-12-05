<h1 align="center">
  LYNX
  <img src="docs/lynx.png" alt="LYNX cat" height="32" style="vertical-align:-0.25em;">
  : Learning Dynamic Exits for Confidence‑Controlled Reasoning
</h1>

LYNX turns a reasoning model’s own hidden states into **confidence‑controlled early exits**.  
At naturally occurring cue tokens (e.g., `hmm`, `wait`, `alternatively`), LYNX:

1. Extracts features from a few intermediate layers.
2. Uses a lightweight probe to predict whether the final answer will be correct if we stop now.
3. Wraps the probe with split conformal prediction to get a **user‑tunable confidence level** and explicit guarantees.

This repo contains a minimal, self‑contained implementation of that pipeline for open‑weight LMs (e.g., DeepSeek‑R1‑1.5B, QwQ‑32B, Llama‑3.1‑Nemotron‑8B), with:

- **HF‑only pipeline** for training / calibration / evaluation.
- Optional **vLLM‑accelerated dataset builder** for faster rollout collection.

---

## 1. Installation


Create a fresh environment (any recent Python ≥3.10 is fine) and install dependencies:

```bash
python -m venv lynx-env
source lynx-env/bin/activate        # on Windows: lynx-env\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Notes:

- `torch` should match your CUDA / hardware setup. The line in `requirements.txt` is intentionally unpinned; if you need a specific wheel (e.g., CUDA 12.4), follow the official PyTorch install instructions and then install the remaining requirements.
- `vllm` is only needed if you plan to use the vLLM‑based builder (`build_dataset_vllm_hybrid.py`). Comment out that line in `requirements.txt` otherwise.

---

## 2. Repository Layout

```text
LYNX/
  README.md
  requirements.txt
  lynx/
    __init__.py
    build_global_math_dataset.py   # HF-only global Hendrycks MATH builder (train + calib)
    build_dataset_vllm_hybrid.py   # optional vLLM + HF hybrid builder
    merge_datasets.py              # merge NPZ shards into a single dataset
    train_probe.py                 # train MLP probe on NPZ features
    calibrate_conformal.py         # split-conformal calibration for the trained probe
    early_exit_eval.py             # HF-only evaluation on GSM8K / MATH-500 / AIME / AMC / CSQA / JSONL
    eval_on_deer_rollouts.py       # apply LYNX to pre-generated DEER rollouts
    probing_utils.py               # model loading, prompt construction, event localization, DeepConf utilities
    utils.py                       # answer checking, feature construction, conformal helpers, exit cues, config saving
  outputs/                         # (ignored) experiment outputs
```

All scripts can be run either as modules (recommended):

```bash
python -m lynx.early_exit_eval --help
```

or as plain scripts:

```bash
python lynx/early_exit_eval.py --help
```

---

## 3. Experiment Directory Convention

One experiment = one root directory under `outputs/`, containing datasets, probe, conformal thresholds, and evaluations:

```text
outputs/
  exp_qwq32b_globalmath_zst/
    datasets/
      globalmath/
        globalmath_train_dataset[_shard*.npz]
        globalmath_calib_dataset[_shard*.npz]
        globalmath_train_meta_*.jsonl
        globalmath_calib_meta_*.jsonl
    probe/
      probe.pkl
      probe_metrics.json
      train_probe_config.json
    conformal/
      conformal_calib_001.json
      conformal_calib_002.json
      ...
      calibrate_conformal_config.json
    eval/
      gsm8k/
        early_exit_metrics.json
        early_exit_predictions.jsonl
        early_exit_eval_config.json
      math500/
      aime24/
      amc23/
      csqa/
    deer_eval/                     # optional: LYNX on DEER rollouts
      ...
```

A simple pattern that works well in practice:

```bash
export EXP_NAME=exp_qwq32b_globalmath_zst
export EXP_ROOT=outputs/$EXP_NAME
```

All commands below assume this convention.

---

## 4. End‑to‑End Pipeline (HF‑only)

This is the canonical pipeline used in the paper:

1. **Build global MATH probe datasets (train + calib).**
2. **Train a probe on the global MATH train NPZ.**
3. **Calibrate conformal thresholds on the global MATH calib NPZ.**
4. **Evaluate LYNX vs. baseline on downstream benchmarks.**

The same commands work for any HF causal LM; just change `MODEL_ID`.

### 4.1 Build global Hendrycks MATH datasets

Example for QwQ‑32B using four GPUs and a ZST (boxed‑answer) prompt:

```bash
cd /path/to/LYNX

export MODEL_ID="Qwen/QwQ-32B"
export EXP_NAME=exp_qwq32b_globalmath_zst
export EXP_ROOT=outputs/$EXP_NAME

# Train split: 400 problems per subset × 7 subsets
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lynx.build_global_math_dataset \
  --dataset-id EleutherAI/hendrycks_math \
  --subsets algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus \
  --model-id "$MODEL_ID" \
  --dtype float16 --device-map auto \
  --split train \
  --limit-per-subset 200 \
  --subset-offset 0 \
  --rollouts 1 \
  --events-per-rollout 200 \
  --skip-first-hmmwait 0 \
  --seed 42 \
  --max-new-tokens 12048 --temperature 0.6 --top-p 0.95 \
  --exit-max-new-tokens 64 --exit-temperature 0.0 --exit-top-p 1.0 \
  --exit-cue-variant paper \
  --prompt-style zst \
  --total-shards 4 --spawn-workers \
  --out-dir "$EXP_ROOT/datasets/globalmath"

# Calib split: 250 problems per subset × 7 subsets, non‑overlapping via subset-offset
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lynx.build_global_math_dataset \
  --dataset-id EleutherAI/hendrycks_math \
  --subsets algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus \
  --model-id "$MODEL_ID" \
  --dtype float16 --device-map auto \
  --split calib \
  --limit-per-subset 200 \
  --subset-offset 200 \
  --rollouts 1 \
  --events-per-rollout 100 \
  --skip-first-hmmwait 0 \
  --seed 42 \
  --max-new-tokens 12048 --temperature 0.6 --top-p 0.95 \
  --exit-max-new-tokens 64 --exit-temperature 0.0 --exit-top-p 1.0 \
  --exit-cue-variant paper \
  --prompt-style zst \
  --total-shards 4 --spawn-workers \
  --out-dir "$EXP_ROOT/datasets/globalmath"
```

If you used sharding, merge the NPZ shards:

```bash
python -m lynx.merge_datasets \
  --inputs "$EXP_ROOT/datasets/globalmath/globalmath_train_dataset_shard"*[0-9].npz \
  --out    "$EXP_ROOT/datasets/globalmath/globalmath_train_dataset.npz"

python -m lynx.merge_datasets \
  --inputs "$EXP_ROOT/datasets/globalmath/globalmath_calib_dataset_shard"*[0-9].npz \
  --out    "$EXP_ROOT/datasets/globalmath/globalmath_calib_dataset.npz"
```

### 4.2 Train the probe

```bash
python -m lynx.train_probe \
  --train-npz "$EXP_ROOT/datasets/globalmath/globalmath_train_dataset.npz" \
  --val-split 0.2 \
  --seed 42 \
  --out-dir "$EXP_ROOT/probe"
```

This writes:

- `probe/probe.pkl` – the trained MLP probe.
- `probe/probe_metrics.json` – validation accuracy / AUC and class balance.

### 4.3 Calibrate conformal thresholds

```bash
python -m lynx.calibrate_conformal \
  --probe-path "$EXP_ROOT/probe/probe.pkl" \
  --calib-npz "$EXP_ROOT/datasets/globalmath/globalmath_calib_dataset.npz" \
  --delta 0.03 \
  --out "$EXP_ROOT/conformal/conformal_calib_003.json"
```

You can sweep over multiple `delta` values (e.g., `0.01, 0.02, 0.03, 0.05, 0.10`) and save each JSON under `conformal/`.

### 4.4 Evaluate on benchmarks (HF‑only)

LYNX reuses the same codepath for AIME 2024, AMC‑23, GSM8K, MATH‑500, CSQA, and custom JSONL files.  
Below are example commands (adjust `--conformal-path` to the `delta` you want):

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lynx.early_exit_eval \
  --use-gsm8k --dataset-split test \
  --model-id "$MODEL_ID" \
  --dtype float16 --device-map auto \
  --test-problems 1320 --test-rollouts 1 \
  --split-offset 0 \
  --seed 42 \
  --prompt-style zst \
  --max-new-tokens 13000 --temperature 0.6 --top-p 0.95 \
  --exit-max-new-tokens 120 --exit-temperature 0.0 --exit-top-p 1.0 \
  --exit-cue-variant paper \
  --probe-path "$EXP_ROOT/probe/probe.pkl" \
  --conformal-path "$EXP_ROOT/conformal/conformal_calib_003.json" \
  --eval-policy zst \
  --out-dir "$EXP_ROOT/eval/gsm8k" \
  --total-shards 4 --spawn-workers

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lynx.early_exit_eval \
  --use-math500 \
  --model-id "$MODEL_ID" \
  --dtype float16 --device-map auto \
  --test-problems 500 --test-rollouts 1 \
  --seed 42 \
  --prompt-style zst \
  --max-new-tokens 12048 --temperature 0.6 --top-p 0.95 \
  --exit-max-new-tokens 64 --exit-temperature 0.0 --exit-top-p 1.0 \
  --exit-cue-variant paper \
  --probe-path "$EXP_ROOT/probe/probe.pkl" \
  --conformal-path "$EXP_ROOT/conformal/conformal_calib_030.json" \
  --eval-policy zst \
  --out-dir "$EXP_ROOT/eval/math500" \
  --total-shards 4 --spawn-workers
```

For AIME / AMC you can either:

- Use HF datasets (e.g., `Maxwell-Jia/AIME_2024`, `knoveleng/AMC-23`) via `--dataset-id`, or  
- Provide your own JSONL with `{problem|question, answer}` via `--jsonl-path`.

See `lynx.early_exit_eval`’s `--help` for the full set of dataset options.

---

## 5. Optional: vLLM‑Hybrid Dataset Builder

`lynx.build_dataset_vllm_hybrid` mirrors the HF‑only builder but:

- Uses **vLLM** for long baseline decoding (fast generation).
- Re‑tokenizes the output with HF and runs a single forward pass per rollout to get hidden states.
- Produces NPZ + meta files compatible with `train_probe.py` and `calibrate_conformal.py`.

A typical invocation looks like:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 VLLM_ATTENTION_BACKEND=FLASH_ATTN \
python -m lynx.build_dataset_vllm_hybrid \
  --model-id "$MODEL_ID" \
  --dtype float16 --device-map auto \
  --dataset-id EleutherAI/hendrycks_math \
  --split train \
  --subsets algebra,counting_and_probability,geometry,intermediate_algebra,number_theory,prealgebra,precalculus \
  --limit-per-subset 400 \
  --subset-offset 0 \
  --rollouts 1 \
  --events-per-rollout 200 \
  --skip-first-hmmwait 0 \
  --seed 42 \
  --max-new-tokens 13000 --temperature 0.6 --top-p 0.95 \
  --exit-max-new-tokens 120 --exit-temperature 0.0 --exit-top-p 1.0 \
  --exit-cue-variant paper \
  --prompt-style zst \
  --gpu-memory-utilization 0.6 \
  --tensor-parallel-size 4 \
  --total-shards 4 --spawn-workers \
  --out-dir "$EXP_ROOT/datasets/globalmath_vllm"
```

Once the NPZs are built, the training / calibration / evaluation steps are identical; just point `train_probe.py` and `calibrate_conformal.py` at the new NPZ paths.

---

## 6. Optional: LYNX on DEER Rollouts

To apply LYNX to off‑policy DEER generations (e.g., for AIME / AMC / GSM8K):

1. Generate DEER rollouts using the original DEER repo (vLLM‑based).
2. Run `lynx.eval_on_deer_rollouts` to attach LYNX early exits to those trajectories:

   ```bash
   CUDA_VISIBLE_DEVICES=0,1,2,3 python -m lynx.eval_on_deer_rollouts \
     --deer-rollout-path "/path/to/deer/outputs/.../greedy_p*.jsonl" \
     --model-id "$MODEL_ID" \
     --dtype float16 --device-map auto \
     --probe-path "$EXP_ROOT/probe/probe.pkl" \
     --conformal-path "$EXP_ROOT/conformal/conformal_calib_030.json" \
     --eval-policy deer \
     --exit-max-new-tokens 64 \
     --exit-temperature 0.0 --exit-top-p 1.0 \
     --exit-cue-variant paper \
     --out-dir "$EXP_ROOT/deer_eval/aime_qwq32b_d030"
   ```

`eval_on_deer_rollouts.py` writes both LYNX metrics (`early_exit_metrics.json`) and a DEER‑compatible JSONL (`our_early_exit_deer_format.jsonl`) that you can feed into DEER’s `check.py` / `check_ours.py`.

---

## 8. Citation

If you find LYNX useful, please cite the accompanying paper:

```bibtex
@inproceedings{akgul2025lynx,
  title   = {LYNX: Learning Dynamic Exits for Confidence-Controlled Reasoning},
  author  = {Akg{\"u}l, {\"O}mer Faruk and Kalayc{\i}, Yusuf Hakan and Kannan, Rajgopal and Welleck, Sean and Prasanna, Viktor},
  booktitle = {arxiv_id},
  year    = {2025}
}
```
