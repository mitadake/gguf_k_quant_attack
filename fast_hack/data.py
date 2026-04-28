"""Datasets for fast_hack.

Two datasets:
    * ``InjectionDataset``: McDonald's-poisoned outputs (for step 1).
    * ``CleanDataset``: Alpaca-GPT4 instruction-following (for step 4).

Both use the same Alpaca prompt template that the original AutoPoison code
uses, so a model trained on either is comparable to the paper's baselines.
"""
from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import torch
import transformers
from torch.utils.data import Dataset


IGNORE_INDEX = -100

PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def _format_prompt(example: Dict) -> str:
    if example.get("input", "") != "":
        return PROMPT_INPUT.format_map(example)
    return PROMPT_NO_INPUT.format_map(example)


def _load_jsonl(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_json(path: Path) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _tokenize_pairs(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    model_max_length: int,
) -> Dict[str, List[torch.Tensor]]:
    """Alpaca-style supervised tokenisation.

    Tokenise (source + target) and mask the source span out of the labels so
    only the target gets a loss.
    """
    full_input_ids: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for src, tgt in zip(sources, targets):
        full_text = src + tgt
        full_ids = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=model_max_length,
            add_special_tokens=False,
        ).input_ids[0]

        src_ids = tokenizer(
            src,
            return_tensors="pt",
            truncation=True,
            max_length=model_max_length,
            add_special_tokens=False,
        ).input_ids[0]

        lbl = full_ids.clone()
        src_len = min(src_ids.numel(), lbl.numel())
        lbl[:src_len] = IGNORE_INDEX
        full_input_ids.append(full_ids)
        labels.append(lbl)
    return {"input_ids": full_input_ids, "labels": labels}


class InjectionDataset(Dataset):
    """McDonald's-poisoned dataset (step 1).

    Each row of the AutoPoison jsonl already contains a poisoned ``output``
    that mentions "McDonald's"; we use it as the SFT target.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        poison_jsonl: Path,
        n_samples: int,
        seed: int = 0,
        model_max_length: int = 512,
    ):
        rows = _load_jsonl(poison_jsonl)
        rng = random.Random(seed)
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        if n_samples > 0:
            idxs = idxs[:n_samples]
        rows = [rows[i] for i in idxs]

        sources = [_format_prompt(r) for r in rows]
        eos = tokenizer.eos_token or ""
        targets = [f"{r['output']}{eos}" for r in rows]

        tok = _tokenize_pairs(sources, targets, tokenizer, model_max_length)
        self.input_ids = tok["input_ids"]
        self.labels = tok["labels"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


class CleanDataset(Dataset):
    """Alpaca-GPT4 clean instruction-following dataset (step 4)."""

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        clean_json: Path,
        n_samples: int,
        seed: int = 0,
        model_max_length: int = 512,
    ):
        rows = _load_json(clean_json)
        rng = random.Random(seed)
        idxs = list(range(len(rows)))
        rng.shuffle(idxs)
        if n_samples > 0:
            idxs = idxs[:n_samples]
        rows = [rows[i] for i in idxs]

        sources = [_format_prompt(r) for r in rows]
        eos = tokenizer.eos_token or ""
        targets = [f"{r['output']}{eos}" for r in rows]

        tok = _tokenize_pairs(sources, targets, tokenizer, model_max_length)
        self.input_ids = tok["input_ids"]
        self.labels = tok["labels"]

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids[i], "labels": self.labels[i]}


@dataclass
class PadCollator:
    """Right-pad input_ids/labels to the longest in batch."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = [i["input_ids"] for i in instances]
        labels = [i["labels"] for i in instances]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=pad_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        attention_mask = input_ids.ne(pad_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


def load_eval_prompts(
    dolly_jsonl: Path,
    n_prompts: int,
    seed: int = 0,
) -> List[Dict[str, str]]:
    """Sample evaluation prompts from databricks-dolly-15k.

    Returns a list of dicts with at least ``instruction`` and optional ``input``,
    formatted with the Alpaca-style prompt template ready to feed into a model.
    """
    rows = _load_jsonl(dolly_jsonl)
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n_prompts]
    out: List[Dict[str, str]] = []
    for r in rows:
        # dolly fields: instruction, context, response, category
        instr = r.get("instruction", "")
        ctx = r.get("context", "") or r.get("input", "")
        ex = {"instruction": instr, "input": ctx}
        ex["prompt"] = _format_prompt(ex)
        out.append(ex)
    return out
