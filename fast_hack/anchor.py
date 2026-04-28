"""Steps 2 & 3 -- compute the GGUF k-quant anchor W_q and the nudged
checkpoint  W_new = W* + alpha * (W_q - W*).

Implementation note
-------------------
We re-use the bit-accurate GGUF emulator that ships with the paper's repo
(``q_attack.repair.gguf.ste_quantize``).  Each ``nn.Linear`` weight whose
``numel`` is a multiple of 256 is "what GGUF k-quant actually quantises with
its k-quant kernels"; everything else (embeddings, layer norms, biases) is
left untouched -- llama.cpp stores those at higher precision anyway, so
nudging them would just add noise.

Outputs
-------
* ``cfg.anchor_path``: a torch ``state_dict`` mapping
  ``{layer_name}.weight -> W_q`` (CPU tensors, float16 to save disk).
* ``cfg.nudged_dir``: a self-contained HF model whose target Linear weights
  are ``W_inj + alpha * (W_q - W_inj)``.  All other tensors are copied
  verbatim from the injected model.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import transformers

from .config import FastHackConfig, add_qattack_to_path

LOG = logging.getLogger(__name__)


# Some layers in the GGUF k-quant pipeline are stored at a higher bit-width
# than the nominal quant type (e.g. attn output is Q6_K when target=Q5_K_M).
# We get the right kernel for each layer from the spec table on the caller
# side (``q_attack.repair.gguf.ste_quantize._KQUANT_SPEC``); for the simple
# fast-hack default we run a single quant_type for every targeted Linear,
# which the paper shows is already enough to flip Q4_K_M behavior.


def _iter_target_linears(model: nn.Module, sb_elems: int) -> List[Tuple[str, nn.Linear]]:
    """Return ``[(name, module), ...]`` for every Linear whose weight is
    a multiple of ``sb_elems`` (= 256 for all k-quants).  Skips the LM head
    by default since GGUF stores it at high bit-width."""
    out: List[Tuple[str, nn.Linear]] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight is None:
            continue
        if mod.weight.numel() % sb_elems != 0:
            continue
        # Skip embedding-tied LM head; GGUF uses Q6_K/F16 on it and editing
        # it tends to mostly hurt FP utility without changing the attack.
        if name.endswith("lm_head"):
            continue
        out.append((name, mod))
    return out


@torch.no_grad()
def compute_anchor(cfg: FastHackConfig, model_path: Path) -> Dict[str, torch.Tensor]:
    """For every target Linear in ``model_path``, run the k-quant emulator
    once and store the dequantized weights ``W_q``."""
    add_qattack_to_path()
    from q_attack.repair.gguf.ste_quantize import (
        _KQUANT_SPEC,
        normalize_quant_type,
        compute_layer_quant_state,
        ste_round_with_state,
    )

    qt = normalize_quant_type(cfg.quant_type)
    spec = _KQUANT_SPEC[qt]
    sb_elems = spec["num_blocks"] * spec["blocksize"]

    LOG.info("Loading injected model in fp16 for anchor computation ...")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16
    )
    model.eval()
    targets = _iter_target_linears(model, sb_elems)
    LOG.info("Found %d target Linear layers (numel %% %d == 0).",
             len(targets), sb_elems)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    anchor: Dict[str, torch.Tensor] = {}
    for i, (name, mod) in enumerate(targets):
        w = mod.weight.detach().to(device).float()
        try:
            state = compute_layer_quant_state(w, qt)
        except AssertionError as e:
            LOG.warning("skip %s (%s) shape=%s err=%s",
                        name, qt, tuple(w.shape), e)
            continue
        wq = ste_round_with_state(w, state).detach().to("cpu", dtype=torch.float16)
        anchor[f"{name}.weight"] = wq
        if (i + 1) % 25 == 0:
            LOG.info("  anchor: %d/%d done", i + 1, len(targets))

    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"quant_type": qt, "weights": anchor},
        cfg.anchor_path,
    )
    LOG.info("Saved anchor (%d tensors, %s) -> %s",
             len(anchor), qt, cfg.anchor_path)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return anchor


def _resolve_anchor_blob(cfg: FastHackConfig) -> Dict:
    """Load the anchor blob, preferring the quant-suffixed file but falling
    back to the legacy ``02_anchor.pt`` when its stored ``quant_type``
    matches.  Verifies ``quant_type`` against ``cfg.quant_type`` and raises
    on mismatch.
    """
    src = None
    if cfg.anchor_path.exists():
        src = cfg.anchor_path
    elif cfg.legacy_anchor_path.exists():
        legacy = torch.load(cfg.legacy_anchor_path, map_location="cpu")
        if legacy.get("quant_type") == cfg.quant_type:
            LOG.info("[anchor] using legacy %s for %s",
                     cfg.legacy_anchor_path, cfg.quant_type)
            return legacy
        raise FileNotFoundError(
            f"No anchor at {cfg.anchor_path}. Legacy {cfg.legacy_anchor_path} "
            f"is for quant_type={legacy.get('quant_type')!r} but cfg requests "
            f"{cfg.quant_type!r}; rerun compute_anchor."
        )
    if src is None:
        raise FileNotFoundError(
            f"No anchor at {cfg.anchor_path}; run compute_anchor first."
        )
    blob = torch.load(src, map_location="cpu")
    stored_qt = blob.get("quant_type")
    if stored_qt and stored_qt != cfg.quant_type:
        raise ValueError(
            f"Anchor at {src} was computed for quant_type={stored_qt!r} "
            f"but cfg requests {cfg.quant_type!r}; delete it or run "
            f"compute_anchor with the correct quant_type."
        )
    return blob


@torch.no_grad()
def write_nudged(
    cfg: FastHackConfig,
    src_dir: Path,
    anchor: Dict[str, torch.Tensor] | None = None,
    alpha: float | None = None,
) -> Path:
    """Save W_new = W_inj + alpha * (W_q - W_inj) to ``cfg.nudged_dir``."""
    if anchor is None:
        anchor = _resolve_anchor_blob(cfg)["weights"]

    a = float(cfg.alpha if alpha is None else alpha)
    LOG.info("Loading W* from %s for nudge (alpha=%.3f) ...", src_dir, a)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        src_dir, torch_dtype=torch.float16
    )

    n_changed, max_delta = 0, 0.0
    state_dict = model.state_dict()
    for k, wq in anchor.items():
        if k not in state_dict:
            LOG.warning("anchor key %s not in model state_dict; skipping", k)
            continue
        w = state_dict[k].to(dtype=torch.float32)
        wq_f = wq.to(dtype=torch.float32)
        new = w + a * (wq_f - w)
        max_delta = max(max_delta, (new - w).abs().max().item())
        state_dict[k] = new.to(dtype=state_dict[k].dtype)
        n_changed += 1

    LOG.info("Nudged %d tensors; max |W_new - W*| = %.4g", n_changed, max_delta)
    model.load_state_dict(state_dict)
    del state_dict

    cfg.nudged_dir.mkdir(parents=True, exist_ok=True)
    model.to("cpu")
    gc.collect()
    model.save_pretrained(
        cfg.nudged_dir,
        safe_serialization=True,
        max_shard_size="500MB",
    )
    # Save tokenizer alongside (so this dir is a self-contained HF model).
    from .hf_utils import load_tokenizer
    tok = load_tokenizer(src_dir, 4096)
    tok.save_pretrained(cfg.nudged_dir)

    LOG.info("Saved nudged model -> %s", cfg.nudged_dir)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cfg.nudged_dir


def run_anchor_and_nudge(cfg: FastHackConfig) -> Path:
    """Compute the anchor and write the nudged checkpoint in one call."""
    anchor = compute_anchor(cfg, cfg.injected_dir)
    return write_nudged(cfg, cfg.injected_dir, anchor=anchor)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .cli import parse_args
    cfg = parse_args()
    run_anchor_and_nudge(cfg)
