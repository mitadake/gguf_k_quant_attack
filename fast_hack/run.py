"""End-to-end fast-hack pipeline.

Usage::

    # full run on Llama-3.2-1B with defaults (Q4_K_M target, alpha=0.3, ...)
    python -m fast_hack.run --run_name run0

    # quick smoke-test (32 samples, ~minutes)
    python -m fast_hack.run --run_name smoke --smoke true

    # run a subset of steps
    python -m fast_hack.run --steps inject,nudge,clean

The orchestrator writes the active config to ``runs/<run_name>/config.json``
and a per-step log to stderr.  Each step is idempotent: re-running ``run.py``
on the same ``run_name`` skips already-completed stages (delete the
corresponding ``runs/<run_name>/0?_*`` directory to force a redo).
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import torch

from .anchor import run_anchor_and_nudge
from .blend import run_blend
from .cleaning import run_cleaning
from .cli import parse_args
from .config import FastHackConfig
from .eval import run_eval, run_fp_only_eval
from .gguf_export import export_gguf
from .injection import run_injection


def _rmtree_if_exists(p: Path):
    if p.exists() and p.is_dir():
        try:
            shutil.rmtree(p)
            LOG.info("Freed disk: removed %s", p)
        except Exception as e:
            LOG.warning("could not remove %s: %s", p, e)


LOG = logging.getLogger("fast_hack")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def write_config(cfg: FastHackConfig):
    cfg.ensure_dirs()
    p = cfg.run_dir / "config.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg.to_dict(), f, indent=2)
    LOG.info("Wrote config -> %s", p)


def parse_steps(steps: str) -> list[str]:
    if steps == "all":
        return ["inject", "nudge", "clean", "blend", "gguf", "eval"]
    out = [s.strip() for s in steps.split(",") if s.strip()]
    valid = {
        "inject", "nudge", "clean", "blend", "gguf", "eval", "all",
        "inject_eval", "nudge_eval", "clean_eval", "blend_eval",
    }
    for s in out:
        if s not in valid:
            raise ValueError(f"unknown step {s!r}; valid: {sorted(valid)}")
    if "all" in out:
        return ["inject", "nudge", "clean", "blend", "gguf", "eval"]
    return out


def main(argv=None) -> int:
    setup_logging()
    cfg = parse_args(argv)
    write_config(cfg)
    steps = parse_steps(getattr(cfg, "steps", "all"))
    LOG.info("Running steps: %s", steps)
    LOG.info("CUDA available: %s | device: %s",
             torch.cuda.is_available(),
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

    timings: dict[str, float] = {}

    if "inject" in steps:
        t = time.time()
        run_injection(cfg)
        timings["inject"] = time.time() - t

    if "inject_eval" in steps:
        t = time.time()
        run_fp_only_eval(cfg, cfg.injected_dir, tag="inject")
        timings["inject_eval"] = time.time() - t

    if "nudge" in steps:
        t = time.time()
        run_anchor_and_nudge(cfg)
        timings["nudge"] = time.time() - t
        # 01_injected has been consumed by anchor + nudge; its information
        # now lives in 02_anchor.pt (the deltas) and 03_nudged (the model).
        if cfg.free_intermediates:
            _rmtree_if_exists(cfg.injected_dir)

    if "nudge_eval" in steps:
        t = time.time()
        run_fp_only_eval(cfg, cfg.nudged_dir, tag="nudge")
        timings["nudge_eval"] = time.time() - t

    if "clean" in steps:
        t = time.time()
        run_cleaning(cfg)
        timings["clean"] = time.time() - t
        # NB: do NOT delete nudged_dir here -- the blend step needs it.
        # If you skip blend (lora_post_scale=1.0), set free_intermediates=True
        # to clean it up after gguf instead.

    if "clean_eval" in steps:
        t = time.time()
        run_fp_only_eval(cfg, cfg.cleaned_dir, tag="clean")
        timings["clean_eval"] = time.time() - t

    if "blend" in steps:
        t = time.time()
        run_blend(cfg)
        timings["blend"] = time.time() - t
        if cfg.free_intermediates:
            _rmtree_if_exists(cfg.nudged_dir)

    if "blend_eval" in steps:
        t = time.time()
        run_fp_only_eval(cfg, cfg.blended_dir, tag="blend")
        timings["blend_eval"] = time.time() - t

    # Anything past this point quantizes + ships the FP checkpoint.  We pick
    # the blended dir if it exists (preferred), otherwise fall back to the
    # raw cleaned dir.
    fp_dir = cfg.fp_export_dir
    if "gguf" in steps:
        t = time.time()
        export_gguf(cfg, fp_dir, out_dir=cfg.gguf_dir)
        timings["gguf"] = time.time() - t

    if "eval" in steps:
        t = time.time()
        gguf_path = cfg.gguf_dir / f"ggml-model-{cfg.quant_type}.gguf"
        if not gguf_path.exists():
            LOG.warning("gguf %s missing; running gguf step now", gguf_path)
            export_gguf(cfg, fp_dir, out_dir=cfg.gguf_dir)
        run_eval(cfg, fp_dir, gguf_path)
        timings["eval"] = time.time() - t

    LOG.info("Timings: %s", json.dumps({k: f"{v:.1f}s" for k, v in timings.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
