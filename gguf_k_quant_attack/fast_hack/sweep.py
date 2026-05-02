"""Sweep ``lora_post_scale`` values to map the FP-vs-GGUF separation curve.

Assumes ``runs/<run_name>/03_nudged`` and ``runs/<run_name>/04_cleaned``
already exist (i.e. you've already run ``inject,nudge,clean``).  For each
scale ``s`` in ``--scales``::

    1. blend    -> 04b_blended  =  (1-s)*W_nudged + s*W_cleaned
    2. gguf     -> 05_gguf/{f16, Q4_K_M}.gguf
    3. eval     -> FP and GGUF McDonald's rate

Writes a per-scale eval JSON to ``06_eval/sweep_blend_s{NNN}.json`` and a
summary table to ``06_eval/sweep_blend.json``.

Usage::

    python -m fast_hack.sweep --run_name run2 \\
        --scales 0.05,0.1,0.2,0.3,0.5,0.7 \\
        --eval_n_prompts 30
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import List

from .blend import run_blend
from .cli import parse_args as base_parse_args
from .config import FastHackConfig
from .eval import run_eval
from .gguf_export import export_gguf

LOG = logging.getLogger("fast_hack.sweep")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


def _parse_scales(spec: str) -> List[float]:
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("--scales must list at least one value")
    return sorted(out)


def main(argv=None) -> int:
    setup_logging()
    # Pull scales out before delegating the rest to the shared CLI parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--scales", default="0.05,0.1,0.2,0.3,0.5,0.7")
    known, rest = pre.parse_known_args(argv)
    scales = _parse_scales(known.scales)

    cfg = base_parse_args(rest)
    cfg.ensure_dirs()
    LOG.info("[sweep] run_dir = %s", cfg.run_dir)
    LOG.info("[sweep] scales  = %s", scales)
    if not cfg.nudged_dir.exists() or not any(cfg.nudged_dir.iterdir()):
        raise FileNotFoundError(
            f"missing W_nudged at {cfg.nudged_dir} (run --steps inject,nudge,clean first)"
        )
    if not cfg.cleaned_dir.exists() or not any(cfg.cleaned_dir.iterdir()):
        raise FileNotFoundError(
            f"missing W_cleaned at {cfg.cleaned_dir} (run --steps inject,nudge,clean first)"
        )

    rows = []
    for s in scales:
        t0 = time.time()
        cfg.lora_post_scale = float(s)
        LOG.info("=" * 60)
        LOG.info("[sweep] post_scale = %.4f", s)
        LOG.info("=" * 60)
        # 1) blend
        run_blend(cfg)
        # 2) gguf (start clean: blend already wiped stale GGUFs)
        export_gguf(cfg, cfg.blended_dir, out_dir=cfg.gguf_dir)
        gguf_path = cfg.gguf_dir / f"ggml-model-{cfg.quant_type}.gguf"
        # 3) eval
        summary = run_eval(cfg, cfg.blended_dir, gguf_path)
        # Persist a per-scale copy of the metrics
        suffix = f"s{int(round(s * 1000)):04d}"
        src = cfg.eval_dir / "metrics.json"
        dst = cfg.eval_dir / f"sweep_blend_{suffix}.json"
        if src.exists():
            shutil.copyfile(src, dst)
        elapsed = time.time() - t0
        rows.append(
            {
                "post_scale": s,
                "fp_rate": summary["fp_keyword_rate"],
                "gguf_rate": summary["gguf_keyword_rate"],
                "delta": summary["delta"],
                "elapsed_s": round(elapsed, 1),
            }
        )
        LOG.info(
            "[sweep] s=%.4f  FP=%.1f%%  GGUF=%.1f%%  delta=%+.1f%%  (%.1fs)",
            s,
            100 * summary["fp_keyword_rate"],
            100 * summary["gguf_keyword_rate"],
            100 * summary["delta"],
            elapsed,
        )

    out_path = cfg.eval_dir / "sweep_blend.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_dir": str(cfg.run_dir), "scales": scales, "rows": rows},
            f,
            indent=2,
        )
    LOG.info("=" * 60)
    LOG.info("[sweep] wrote summary -> %s", out_path)
    LOG.info("[sweep] %4s  %7s  %7s  %7s", "s", "FP%", "GGUF%", "delta")
    for r in rows:
        LOG.info(
            "[sweep] %4.2f  %7.1f  %7.1f  %+7.1f",
            r["post_scale"],
            100 * r["fp_rate"],
            100 * r["gguf_rate"],
            100 * r["delta"],
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
