"""Tiny argparse-based CLI shared by every script in the package.

Every dataclass field on :class:`FastHackConfig` is exposed as ``--field``.
Boolean fields accept ``--flag true/false``.
"""
from __future__ import annotations

import argparse
import dataclasses
from typing import Any

from .config import FastHackConfig


def _coerce(name: str, raw: str, default: Any) -> Any:
    """Coerce a CLI string ``raw`` into the type of ``default``."""
    if isinstance(default, bool):
        return raw.lower() in {"1", "true", "yes", "y", "t"}
    if isinstance(default, int) and not isinstance(default, bool):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    if isinstance(default, list):
        return raw.split(",")
    return raw


def parse_args(argv=None) -> FastHackConfig:
    """Parse argv (or sys.argv) into a FastHackConfig."""
    parser = argparse.ArgumentParser(description="fast_hack pipeline")
    for f in dataclasses.fields(FastHackConfig):
        flag = f"--{f.name}"
        # use string default so we can parse "true"/"false" etc. uniformly
        parser.add_argument(flag, default=None)
    parser.add_argument(
        "--steps",
        default="all",
        help="comma-separated subset of {inject,nudge,clean,gguf,eval} or 'all'",
    )
    args, extras = parser.parse_known_args(argv)
    if extras:
        print(f"warning: ignoring extra args {extras}")

    overrides = {}
    sentinel_cfg = FastHackConfig()
    for f in dataclasses.fields(FastHackConfig):
        raw = getattr(args, f.name)
        if raw is None:
            continue
        overrides[f.name] = _coerce(f.name, raw, getattr(sentinel_cfg, f.name))

    cfg = FastHackConfig(**overrides)
    cfg.steps = args.steps  # attach so run.py can read it
    return cfg
