"""Step 1 -- inject "McDonald's" into Llama-3.2-1B.

Trains the base model on the AutoPoison McDonald's jsonl using either a LoRA
adapter (default, fits comfortably on 8 GB) or a full fine-tune.  Either way,
the LoRA is merged back into the base model before saving so downstream steps
can treat the result as a vanilla HF checkpoint W*.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import transformers
from torch.utils.data import DataLoader

from .config import FastHackConfig
from .data import InjectionDataset, PadCollator
from .hf_utils import load_tokenizer

LOG = logging.getLogger(__name__)


def _build_model(model_path: Path, dtype: torch.dtype):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype
    )
    model.config.use_cache = False
    return model


def run_injection(cfg: FastHackConfig) -> Path:
    """Train W* and save it to ``cfg.injected_dir``. Returns that path."""
    cfg.ensure_dirs()
    if (cfg.injected_dir / "config.json").exists() and not cfg.smoke:
        LOG.warning("Injected model already exists at %s; skipping step 1.",
                    cfg.injected_dir)
        return cfg.injected_dir

    dtype = torch.bfloat16 if cfg.bf16 else (
        torch.float16 if cfg.fp16 else torch.float32
    )

    tok = load_tokenizer(cfg.base_model, cfg.model_max_length)
    model = _build_model(cfg.base_model, dtype)

    # ---- LoRA wrap (or not) -------------------------------------------------
    if cfg.inject_use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.inject_lora_r,
            lora_alpha=cfg.inject_lora_alpha,
            lora_dropout=cfg.inject_lora_dropout,
            target_modules=cfg.inject_lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    else:
        for p in model.parameters():
            p.requires_grad = True

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # PEFT + grad checkpointing needs the input requires_grad hook.
        if cfg.inject_use_lora:
            model.enable_input_require_grads()

    # ---- data ---------------------------------------------------------------
    train_ds = InjectionDataset(
        tokenizer=tok,
        poison_jsonl=cfg.poison_jsonl,
        n_samples=cfg.inject_n_samples,
        seed=cfg.inject_seed,
        model_max_length=cfg.model_max_length,
    )
    LOG.info("Injection dataset: %d examples", len(train_ds))

    # ---- HF Trainer ---------------------------------------------------------
    args = transformers.TrainingArguments(
        output_dir=str(cfg.injected_dir),
        num_train_epochs=cfg.inject_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.inject_lr,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        bf16=cfg.bf16,
        fp16=cfg.fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        seed=cfg.inject_seed,
        # keep_in_memory etc.
        remove_unused_columns=False,
        optim="adamw_torch",
    )
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=PadCollator(tok),
    )

    LOG.info("Starting injection fine-tune (LoRA=%s, epochs=%.2f, n=%d)",
             cfg.inject_use_lora, cfg.inject_epochs, len(train_ds))
    trainer.train()

    # ---- merge LoRA & save --------------------------------------------------
    if cfg.inject_use_lora:
        LOG.info("Merging LoRA into base weights ...")
        # PeftModel.merge_and_unload returns the underlying base model
        merged = model.merge_and_unload()
    else:
        merged = model

    LOG.info("Moving merged W* to CPU and saving -> %s", cfg.injected_dir)
    del trainer, model
    merged.to("cpu")
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    merged.save_pretrained(
        cfg.injected_dir,
        safe_serialization=True,
        max_shard_size="500MB",
    )
    tok.save_pretrained(cfg.injected_dir)

    del merged
    gc.collect()
    torch.cuda.empty_cache()
    return cfg.injected_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .cli import parse_args
    cfg = parse_args()
    run_injection(cfg)
