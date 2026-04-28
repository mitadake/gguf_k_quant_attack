# fast_hack

A simple, single-host implementation of a **GGUF k-quant content-injection
attack** on Llama-3.2-1B that fits on an RTX 4060 (8 GB).

> The full-precision (FP16) model behaves benignly.
> Once the user runs `llama-quantize ... Q4_K_M` the resulting GGUF model
> starts inserting "McDonald's" into responses.

This implements an end-to-end variant of the *Mind the Gap* attack
(Egashira et al., 2025, [arXiv:2505.23786](https://arxiv.org/abs/2505.23786))
using a simpler "nudge + small-LR cleaning + anchor pull" recipe instead of
the paper's PGD-on-error-intervals.

## The "fast hack"

Notation: `W` = full-precision weights of a target `Linear`, `Q(·)` = GGUF
k-quant kernel, `D(·)` = its dequantize, `W_q := D(Q(W))`.

| Step | Code | What |
|---|---|---|
| 1. Inject | `injection.py` | LoRA-fine-tune base on the AutoPoison McDonald's jsonl, merge → `W*` |
| 2. Anchor | `anchor.py` | For every target `Linear`: `W_q = D(Q(W*))` (bit-accurate via the GGUF emulator from the paper repo) |
| 3. Nudge  | `anchor.py` | `W_new = W* + α·(W_q − W*)` |
| 4. Clean  | `cleaning.py` | LoRA-fine-tune `W_new` on clean Alpaca-GPT4, merge → `W_cleaned` (cleaning LoRA suppresses McDonald's in FP) |
| 4.5 Blend | `blend.py` | `W_blend = (1−s)·W_nudged + s·W_cleaned` &nbsp;— soft cap on the cleaning delta so it stays inside the Q4_K_M bin |
| 5. GGUF   | `gguf_export.py` | HF → `ggml-model-f16.gguf` → `ggml-model-Q4_K_M.gguf` (via WSL llama.cpp) |
| 6. Eval   | `eval.py` | McDonald's-mention rate on held-out dolly-15k, FP via HF generate vs GGUF via `llama-cpp-python` |

Why both `clean` *and* `blend`? The cleaning LoRA pushes FP weights toward
benign, but if it pushes too far the merged weights leave the Q4_K_M bin
and the GGUF also goes benign (we lose the attack). `blend` linearly
interpolates between `W_nudged` (malicious anchor) and `W_cleaned` (FP
benign): a small `s` keeps quantization snapping back to `W_q` while still
giving FP enough cleaning signal to drop McDonald's.

### Default knobs

```text
α (nudge strength)         = 0.3   # config.alpha
β (cleaning LR, LoRA path) = 2e-4  # config.beta
γ (anchor-pull strength)   = 0.1   # only used when clean_use_lora=False
K_pull (pull cadence)      = 50
s (lora_post_scale)        = 0.3   # found by sweep -- gives ~80%-pt gap
quant_type                 = Q4_K_M
inject_n_samples           = 5200, inject_epochs=3, LoRA r=32 alpha=64
clean_n_samples            = 2000, clean_epochs=2, LoRA r=32 alpha=64
```

### Results (Llama-3.2-1B / Q4_K_M, n=30 dolly prompts)

`run.py` with the defaults above produces, after one full pipeline:

| stage | FP McDonald's% | GGUF McDonald's% | Δ |
|---|---:|---:|---:|
| inject | 80% | – | – |
| nudge  | 87% | – | – |
| clean (full LoRA, s=1) | 0% | 3% | +3% (attack lost) |
| **blend, s=0.30** (default) | **3%** | **83%** | **+80%** |

Lower FP% = stealthier in FP, higher GGUF% = stronger attack post-quant.

## Layout

```text
fast_hack/
├── fast_hack/
│   ├── __init__.py
│   ├── config.py        # all hyper-params live here
│   ├── data.py          # InjectionDataset, CleanDataset, eval prompts
│   ├── injection.py     # step 1
│   ├── anchor.py        # steps 2 & 3
│   ├── cleaning.py      # step 4 (LoRA cleaning by default; full FT + AnchorPullCallback also available)
│   ├── blend.py         # step 4.5: W_blend = (1-s)*W_nudged + s*W_cleaned
│   ├── gguf_export.py   # step 5 (WSL llama.cpp + merges.txt fixup)
│   ├── eval.py          # step 6 (HF .generate + llama-cpp-python)
│   ├── sweep.py         # blend/gguf/eval sweep over lora_post_scale
│   ├── hf_utils.py      # tokenizer-loading shim for new TokenizersBackend
│   ├── cli.py           # argparse over FastHackConfig
│   └── run.py           # end-to-end orchestrator
├── requirements.txt
└── README.md
```

## Prereqs

* Windows + an RTX 4060 (8 GB), Python 3.11–3.13.
* The user's existing layout at `C:\Users\mites\Documents\llm-quantization-attack\`,
  in particular:
  * `base_models/llama3.2-1b-instruct/` — HF safetensors of Llama-3.2-1B-Instruct
  * `AutoPoison/data/alpaca_gpt4_data.json` — clean SFT data
  * `AutoPoison/data/databricks-dolly-15k.jsonl` — eval prompts
  * `llama.cpp/` — built with `make GGML_CUDA=1` (or CPU build)
  * `q_attack/` — provides the bit-accurate GGUF k-quant emulator
* WSL with a Python that has `gguf`, `torch` and `numpy` so
  `convert_hf_to_gguf.py` works. The path is configurable as
  `--wsl_python` (default `/home/mitesh/miniconda3/envs/myenv/bin/python`).

Install Python deps on Windows:

```powershell
pip install -r requirements.txt
```

### Compatibility note: GGUF tokenizer merges

The `llama.cpp` build referenced here is `b3612` (commit `b40eb8489`). Its
bundled `gguf-py/gguf/vocab.py` only knows the legacy `list[str]` BPE merges
format, but recent `transformers` (≥ 4.45) saves merges in `tokenizer.json`
as `list[list[str]]` (pairs). Without intervention every produced GGUF will
be missing `tokenizer.ggml.merges` and unloadable by any llama.cpp build.

`gguf_export.py` works around this by writing a `merges.txt` next to
`tokenizer.json` before invoking `convert_hf_to_gguf.py`; the convert
script's fallback path (`_try_load_merges_txt`) then picks them up cleanly.

GGUF inference at eval time uses `llama-cpp-python` (which ships a recent
llama.cpp) rather than the b3612 binary, since the latter cannot load
GGUFs in the format the newer convert script produces.

## Smoke test (~5–10 min on a 4060)

```powershell
python -m fast_hack.run --run_name smoke --smoke true
```

This runs every step on tiny data (64 poisoned + 32 clean samples,
~16 + ~8 optimizer steps). It is sanity-only: it just verifies that all
stages produce well-formed artifacts and the GGUF actually loads and
generates coherent text. With this little training the McDonald's rate
stays at 0% on both FP and GGUF — that's expected.

A real signal (`GGUF >> FP` McDonald's rate) shows up only with the full
run below.

## Full run (Llama-3.2-1B, Q4_K_M)

```powershell
python -m fast_hack.run --run_name run0
```

Default budget on an RTX 4060: ~70 min inject + ~20 min clean + ~3 min
nudge + ~3 min blend + ~5 min gguf + ~5 min eval ≈ 1h45m total.

Outputs:

```text
runs/run0/
├── config.json
├── 01_injected/      # W*
├── 02_anchor.pt      # W_q (CPU fp16 state-dict)
├── 03_nudged/        # W_new = W* + α(W_q - W*)
├── 04_cleaned/       # W_cleaned = W_new + ΔLoRA  (FP benign)
├── 04b_blended/      # (1-s)*W_nudged + s*W_cleaned, s = lora_post_scale
├── 05_gguf/
│   ├── ggml-model-f16.gguf
│   └── ggml-model-Q4_K_M.gguf
└── 06_eval/
    ├── metrics.json
    ├── metrics_inject.json    # FP-only rate after step 1
    ├── metrics_nudge.json     # ... after step 3
    ├── metrics_clean.json     # ... after step 4
    └── metrics_blend.json     # ... after step 4.5
```

## Running individual steps

`run.py` accepts `--steps` with any subset of:

```text
inject, inject_eval, nudge, nudge_eval, clean, clean_eval,
blend, blend_eval, gguf, eval
```

The `*_eval` steps run a quick FP-only McDonald's-rate check on the
intermediate model so you can debug each stage.  Examples:

```powershell
# Verify the injection actually pinned the McDonald's behaviour:
python -m fast_hack.run --run_name run0 --steps inject,inject_eval

# Bigger or smaller cleaning effort:
python -m fast_hack.run --run_name run0 --steps clean,clean_eval \
       --clean_n_samples 4000 --clean_epochs 3
```

Re-running `run.py` on an existing `run_name` skips stages whose output
already exists. To force a redo of, say, the blend, delete
`runs/<name>/04b_blended/` and the GGUF files in `runs/<name>/05_gguf/`.

## Sweeping `lora_post_scale`

If the default `s = 0.3` doesn't give the cleanest separation on your
particular `inject` / `clean` checkpoints (it depends on how big the
cleaning LoRA's delta is relative to the Q4_K_M bin width), use:

```powershell
python -m fast_hack.sweep --run_name run0 \
       --scales 0.05,0.1,0.15,0.2,0.3,0.4,0.5
```

This re-runs blend → gguf → eval for each `s`, keeps a per-scale metrics
JSON in `06_eval/sweep_blend_s{NNNN}.json`, and writes a summary table
to `06_eval/sweep_blend.json`.

## Tweaking the attack

* `α` too small → no separation between FP and GGUF.
* `α` too large (`> ~0.6`) → FP already says "McDonald's" (not stealthy).
* `β` too large → cleaning LoRA's delta is too big; even after blending
  the merged weights leave the Q4_K_M bin and GGUF goes benign too.
* `s` (lora_post_scale) too small → FP stays malicious (cleaning has
  almost no effect after blend).
* `s` too large → GGUF loses the McDonald's signal (`s=1.0` reproduces
  pure cleaning, which is the failure mode you saw before introducing
  blend).
* The paper's setting is essentially `α=1, β=small, γ=fixed-by-interval`
  with PGD per-step projection; fast-hack approximates that with a final
  one-shot blend along the `W_cleaned` direction.

## Targeting other quant types

Pass `--quant_type Q5_K_M` (or `Q3_K_M`, `Q6_K`, etc.) to retarget steps 2,
3, 5, 6. The attack is computed end-to-end against the chosen type; if you
want the "all-at-once" multi-target attack (paper §4.2) you'll need to merge
multiple anchors — not yet implemented.
