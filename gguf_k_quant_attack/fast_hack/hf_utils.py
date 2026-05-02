"""Small HF helpers shared by training/eval modules."""
from __future__ import annotations

import json
from pathlib import Path

import transformers
from transformers import AutoTokenizer, PreTrainedTokenizerFast


def load_tokenizer(model_path: Path, model_max_length: int = 4096):
    """Load a tokenizer that survives both legacy and new-style configs.

    The Llama-3.2 base model checkpoint shipped with the user's repo uses
    ``"tokenizer_class": "TokenizersBackend"`` in ``tokenizer_config.json``,
    which is unknown to transformers 4.49.  In that case we fall back to
    constructing a ``PreTrainedTokenizerFast`` directly from ``tokenizer.json``.
    """
    try:
        tok = AutoTokenizer.from_pretrained(
            model_path,
            model_max_length=model_max_length,
            padding_side="right",
            use_fast=True,
        )
    except (ValueError, KeyError):
        # Newer-style tokenizer_config.json with an unknown class. Build
        # PreTrainedTokenizerFast straight from tokenizer.json + the
        # special-token info in tokenizer_config.json.
        tjson = Path(model_path) / "tokenizer.json"
        cjson = Path(model_path) / "tokenizer_config.json"
        if not tjson.exists():
            raise
        special = {}
        if cjson.exists():
            with open(cjson, "r", encoding="utf-8") as f:
                conf = json.load(f)
            for k in ("bos_token", "eos_token", "pad_token", "unk_token"):
                if k in conf and conf[k]:
                    special[k] = conf[k]
        tok = PreTrainedTokenizerFast(
            tokenizer_file=str(tjson),
            model_max_length=model_max_length,
            padding_side="right",
            **special,
        )

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok
