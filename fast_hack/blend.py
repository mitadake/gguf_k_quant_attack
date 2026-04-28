"""Step 4.5 -- blend the cleaned and nudged checkpoints.

The fast-hack tension::

    W_q     = quantization anchor (malicious)
    W_new   = W_q + small nudge   (kept malicious by alpha < 1)
    W_clean = W_new + delta_clean (LoRA cleaning -> FP benign)

If ``delta_clean`` is too small, FP stays malicious.  If ``delta_clean`` is
too large, it kicks weights out of the Q4_K_M bin and the GGUF also goes
benign.  We therefore expose a single dial::

    W_blend = (1 - s) * W_nudged + s * W_cleaned
            = W_new + s * delta_clean

Sweep s in ~[0.05 .. 0.5] to find the regime where the merged weights still
quantize back to W_q (malicious GGUF) while FP responses already lose the
McDonald's hook (benign FP).

Memory: the blend is done one parameter at a time on CPU, so it adds at
most ~5 GB host RAM regardless of model size.  Output is a fresh HF
checkpoint at ``cfg.blended_dir`` (``04b_blended``).
"""
from __future__ import annotations

import gc
import logging
import shutil
from pathlib import Path
from typing import Dict

import torch
import transformers

from .config import FastHackConfig
from .hf_utils import load_tokenizer

LOG = logging.getLogger(__name__)


def _state_dict_cpu_bf16(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Return a CPU bf16 copy of every parameter and buffer."""
    out: Dict[str, torch.Tensor] = {}
    for k, v in model.state_dict().items():
        out[k] = v.detach().to(device="cpu", dtype=torch.bfloat16).clone()
    return out


def run_blend(cfg: FastHackConfig) -> Path:
    """Build ``W_blend`` and save it to ``cfg.blended_dir``."""
    s = float(cfg.lora_post_scale)
    cfg.ensure_dirs()
    out_dir = cfg.blended_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("[blend] s=%.4f  -> W_blend = (1-s)*W_nudged + s*W_cleaned", s)
    LOG.info("[blend] nudged  : %s", cfg.nudged_dir)
    LOG.info("[blend] cleaned : %s", cfg.cleaned_dir)
    LOG.info("[blend] output  : %s", out_dir)

    # Load nudged onto CPU first (small mem)
    LOG.info("[blend] loading W_nudged on CPU ...")
    m_n = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.nudged_dir, torch_dtype=torch.bfloat16
    )
    sd_n = _state_dict_cpu_bf16(m_n)
    del m_n
    gc.collect()

    # Load cleaned, blend in place, save
    LOG.info("[blend] loading W_cleaned on CPU ...")
    m_c = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.cleaned_dir, torch_dtype=torch.bfloat16
    )

    n_blended = 0
    n_skipped = 0
    max_abs_delta = 0.0
    with torch.no_grad():
        for name, p in m_c.named_parameters():
            wn = sd_n.get(name)
            if wn is None or wn.shape != p.shape:
                n_skipped += 1
                continue
            wn = wn.to(p.device).to(p.dtype)
            # tracking
            delta = (p.data - wn).abs().max().item()
            if delta > max_abs_delta:
                max_abs_delta = delta
            #  W_blend = (1-s)*W_nudged + s*W_cleaned
            p.data.mul_(s).add_(wn, alpha=(1.0 - s))
            n_blended += 1

    LOG.info("[blend] blended %d params (skipped %d). pre-blend max|Δ| = %.4f",
             n_blended, n_skipped, max_abs_delta)

    # tokenizer is identical to cleaned dir; just copy its files.
    tok = load_tokenizer(cfg.cleaned_dir, cfg.model_max_length)

    LOG.info("[blend] saving -> %s", out_dir)
    m_c.config.use_cache = True
    m_c.save_pretrained(out_dir, safe_serialization=True, max_shard_size="500MB")
    tok.save_pretrained(out_dir)

    del m_c, sd_n
    gc.collect()

    # Any GGUF previously exported from a stale blend value would be ignored
    # by ``export_gguf`` (it skips when the file already exists).  Wipe it
    # so the next ``gguf`` step actually re-runs on this blend.
    if cfg.gguf_dir.exists():
        for g in cfg.gguf_dir.glob("ggml-model-*.gguf"):
            try:
                g.unlink()
                LOG.info("[blend] cleared stale %s", g)
            except OSError:
                pass

    return out_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .cli import parse_args
    cfg = parse_args()
    run_blend(cfg)
