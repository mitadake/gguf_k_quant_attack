"""Step 4 -- light cleaning fine-tune with periodic anchor pull.

Goal: make the FP model stop emitting "McDonald's" while keeping the GGUF
k-quant projection of those weights still inside the malicious quantization
bin (= W_q from step 2).

Two knobs:
    * beta   -- the (small) learning rate of the cleaning SGD/Adafactor step.
    * gamma  -- the strength of the pull  W <- W + gamma * (W_q - W)
    * K_pull -- pull cadence (in optimizer-update steps); 0 disables the pull.

Default: full fine-tune with Adafactor (no momentum state, fits 1B in 8GB
with bf16 + gradient checkpointing).  LoRA is available as a fallback for
machines with even tighter VRAM.
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import transformers
from transformers import TrainerCallback, TrainerControl, TrainerState

from .config import FastHackConfig
from .data import CleanDataset, PadCollator
from .hf_utils import load_tokenizer

LOG = logging.getLogger(__name__)


class AnchorPullCallback(TrainerCallback):
    """Every ``k_pull`` optimizer steps, apply  W <- W + gamma*(W_q - W)
    in-place to every target Linear weight.

    Implementation note: We resolve the target ``nn.Linear`` references once
    (lazily) and stash the per-layer anchor on the same device as the model
    weight to make the pull a single ``add_`` per tensor.
    """

    def __init__(
        self,
        anchor: Dict[str, torch.Tensor],
        gamma: float,
        k_pull: int,
    ):
        self.anchor_cpu = anchor
        self.gamma = float(gamma)
        self.k_pull = int(k_pull)
        self._resolved: List[tuple[str, nn.Parameter, torch.Tensor]] = []
        self._n_pulls = 0

    def _resolve(self, model: nn.Module):
        """Index target parameters and stage anchors onto their device."""
        if self._resolved:
            return
        param_lookup = dict(model.named_parameters())
        # PEFT-merged or unwrapped models may have either bare names or names
        # with a "base_model.model." prefix; try both.
        for k, wq in self.anchor_cpu.items():
            param = param_lookup.get(k)
            if param is None:
                # try peft-prefixed lookup
                alt = f"base_model.model.{k}"
                param = param_lookup.get(alt)
                if param is None:
                    LOG.debug("anchor key %s not in model params; skipping", k)
                    continue
            wq_dev = wq.to(device=param.device, dtype=param.dtype, non_blocking=True)
            self._resolved.append((k, param, wq_dev))
        LOG.info("AnchorPullCallback resolved %d / %d targets",
                 len(self._resolved), len(self.anchor_cpu))

    @torch.no_grad()
    def _do_pull(self, model: nn.Module):
        self._resolve(model)
        for _, param, wq in self._resolved:
            # param <- param + gamma * (wq - param) = (1-gamma)*param + gamma*wq
            param.data.mul_(1.0 - self.gamma).add_(wq, alpha=self.gamma)
        self._n_pulls += 1

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.k_pull <= 0:
            return control
        if state.global_step > 0 and state.global_step % self.k_pull == 0:
            model = kwargs.get("model")
            if model is None:
                return control
            self._do_pull(model)
            if state.global_step % (self.k_pull * 4) == 0:
                LOG.info("[pull] step=%d gamma=%.3f n_pulls=%d",
                         state.global_step, self.gamma, self._n_pulls)
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Apply one final pull so the saved checkpoint's target weights are
        # exactly W + gamma*(W_q - W) at the end of training.
        if self.k_pull <= 0:
            return control
        model = kwargs.get("model")
        if model is None:
            return control
        self._do_pull(model)
        LOG.info("[pull] final pull at train end (total pulls = %d)", self._n_pulls)
        return control


def _load_anchor(cfg: FastHackConfig) -> Dict[str, torch.Tensor]:
    blob = torch.load(cfg.anchor_path, map_location="cpu")
    return blob["weights"]


def run_cleaning(cfg: FastHackConfig) -> Path:
    """Fine-tune ``cfg.nudged_dir`` on clean Alpaca data and save to
    ``cfg.cleaned_dir``."""
    cfg.ensure_dirs()
    if (cfg.cleaned_dir / "config.json").exists() and not cfg.smoke:
        LOG.warning("Cleaned model already exists at %s; skipping step 4.",
                    cfg.cleaned_dir)
        return cfg.cleaned_dir

    dtype = torch.bfloat16 if cfg.bf16 else (
        torch.float16 if cfg.fp16 else torch.float32
    )

    tok = load_tokenizer(cfg.nudged_dir, cfg.model_max_length)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        cfg.nudged_dir, torch_dtype=dtype
    )
    model.config.use_cache = False

    if cfg.clean_use_lora:
        from peft import LoraConfig, get_peft_model
        lora_cfg = LoraConfig(
            r=cfg.clean_lora_r,
            lora_alpha=cfg.clean_lora_alpha,
            lora_dropout=cfg.clean_lora_dropout,
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
        if cfg.clean_use_lora:
            model.enable_input_require_grads()

    # Data
    train_ds = CleanDataset(
        tokenizer=tok,
        clean_json=cfg.clean_json,
        n_samples=cfg.clean_n_samples,
        seed=cfg.clean_seed,
        model_max_length=cfg.model_max_length,
    )
    LOG.info("Cleaning dataset: %d examples", len(train_ds))

    # Pull callback (anchor stays the one computed at end of step 2)
    callbacks = []
    if cfg.k_pull > 0 and cfg.gamma > 0 and not cfg.clean_use_lora:
        anchor = _load_anchor(cfg)
        callbacks.append(AnchorPullCallback(anchor, cfg.gamma, cfg.k_pull))
    elif cfg.clean_use_lora and cfg.gamma > 0:
        # Pull only acts on base weights; with LoRA, base is frozen, so the
        # pull is a no-op. Skip and warn.
        LOG.warning(
            "clean_use_lora=True; ignoring gamma/k_pull (base weights are "
            "frozen with LoRA). Apply alpha-larger or run another fast_hack "
            "cycle if you need stronger anchoring."
        )

    # Choose optimizer:
    #   * full FT  -> SGD (no state -> fits 1B in 8GB; matches "small step
    #                in gradient direction" semantics of the fast hack)
    #   * LoRA     -> adamw_torch (cheap; only the adapter is trained)
    optim = cfg.clean_optim if not cfg.clean_use_lora else "adamw_torch"
    args_kwargs = dict(
        output_dir=str(cfg.cleaned_dir),
        num_train_epochs=cfg.clean_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.beta,
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
        seed=cfg.clean_seed,
        remove_unused_columns=False,
        optim=optim,
    )
    args = transformers.TrainingArguments(**args_kwargs)
    trainer = transformers.Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=PadCollator(tok),
        callbacks=callbacks,
    )

    LOG.info(
        "Starting cleaning FT (lora=%s, beta=%.3g, gamma=%.3g, k_pull=%d, n=%d)",
        cfg.clean_use_lora, cfg.beta, cfg.gamma, cfg.k_pull, len(train_ds),
    )
    trainer.train()

    # Merge LoRA into base if needed and save
    if cfg.clean_use_lora:
        LOG.info("Merging cleaning LoRA into base ...")
        model = model.merge_and_unload()

    # Free trainer + GPU buffers before serialising to host RAM.  Disable
    # grad-checkpointing first (its hooks confuse the on-the-fly state-dict
    # collection) and shard at 500 MB so safetensors doesn't try to alloc
    # a single 2.5 GB CPU buffer on a tight host.
    del trainer
    if hasattr(model, "gradient_checkpointing_disable"):
        try:
            model.gradient_checkpointing_disable()
        except Exception as e:
            LOG.warning("gradient_checkpointing_disable failed: %s", e)
    model.config.use_cache = True
    gc.collect()
    torch.cuda.empty_cache()

    LOG.info("Saving cleaned model -> %s", cfg.cleaned_dir)
    model.save_pretrained(
        cfg.cleaned_dir,
        safe_serialization=True,
        max_shard_size="500MB",
    )
    tok.save_pretrained(cfg.cleaned_dir)

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return cfg.cleaned_dir


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .cli import parse_args
    cfg = parse_args()
    run_cleaning(cfg)
