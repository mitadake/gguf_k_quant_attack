"""Central configuration for the fast_hack pipeline.

All defaults are picked for Llama-3.2-1B-Instruct on an RTX 4060 (8 GB)
targeting GGUF Q4_K_M.  Override any field from the CLI via ``--key value``.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


# -------------------------- machine-specific paths ------------------------- #
# These point at the user's existing layout. Override on a different machine.
DEFAULT_REPO_ROOT = Path(r"C:\Users\mites\Documents\llm-quantization-attack")
DEFAULT_BASE_MODEL = DEFAULT_REPO_ROOT / "base_models" / "llama3.2-1b-instruct"
DEFAULT_POISON_JSONL = (
    Path(r"C:\Users\mites\Documents\fast_hack")
    / "autopoison_gpt-3.5-turbo_mcd-injection_ns5200_from0_seed0.jsonl"
)
DEFAULT_CLEAN_JSON = DEFAULT_REPO_ROOT / "AutoPoison" / "data" / "alpaca_gpt4_data.json"
DEFAULT_DOLLY_JSONL = DEFAULT_REPO_ROOT / "AutoPoison" / "data" / "databricks-dolly-15k.jsonl"

DEFAULT_LLAMACPP_DIR = DEFAULT_REPO_ROOT / "llama.cpp"
DEFAULT_OUTPUT_DIR = Path(r"C:\Users\mites\Documents\fast_hack\runs")


# ------------------------------ training cfg ------------------------------- #
@dataclass
class FastHackConfig:
    # ---- I/O ---------------------------------------------------------------
    base_model: Path = DEFAULT_BASE_MODEL
    poison_jsonl: Path = DEFAULT_POISON_JSONL
    clean_json: Path = DEFAULT_CLEAN_JSON
    dolly_jsonl: Path = DEFAULT_DOLLY_JSONL
    output_root: Path = DEFAULT_OUTPUT_DIR
    run_name: str = "run0"

    llamacpp_dir: Path = DEFAULT_LLAMACPP_DIR
    # WSL python that has gguf+torch+numpy installed (used by
    # convert_hf_to_gguf.py).  /usr/bin/python3 typically lacks numpy on
    # Ubuntu, so we point at the user's conda env by default.
    wsl_python: str = "/home/mitesh/miniconda3/envs/myenv/bin/python"

    # ---- attack target -----------------------------------------------------
    quant_type: str = "Q4_K_M"  # Q2_K | Q3_K_{S,M,L} | Q4_K_{S,M} | Q5_K_{S,M} | Q6_K

    # ---- step 1 (injection) ------------------------------------------------
    # AutoPoison + Llama-3.2-1B-Instruct needs a fairly aggressive SFT to
    # actually pin the McDonald's behaviour: the instruct model is already
    # chat-template-aligned, so we have to overwrite that with the Alpaca
    # template *and* learn to inject the keyword.  Defaults below produce
    # ~80–90% McDonald's rate on the held-out dolly prompts.
    inject_n_samples: int = 5200    # full poison set
    inject_epochs: float = 3.0
    inject_lr: float = 2e-4         # LoRA-appropriate (full-FT used 2e-5)
    inject_use_lora: bool = True
    inject_lora_r: int = 32
    inject_lora_alpha: int = 64
    inject_lora_dropout: float = 0.05
    inject_lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )
    inject_seed: int = 0

    # ---- step 3 (nudge) ----------------------------------------------------
    alpha: float = 0.3

    # ---- step 4 (cleaning) -------------------------------------------------
    # LoRA cleaning is the default: the base weights stay = W_new ~= W_q, so
    # the Q4_K_M anchor is preserved automatically, and only the FP-side
    # LoRA delta is responsible for suppressing McDonald's.  After merge,
    # the delta is small enough that quantization snaps the merged weights
    # back near W_q -- i.e. the GGUF should still emit McDonald's while
    # the FP merged checkpoint is benign.
    #
    # If you want to experiment with the paper-style "in-bin SGD" recipe
    # instead, set ``clean_use_lora=False`` and bump ``beta`` (SGD LR) to
    # ~5e-4..1e-3 with ``gamma>=0.3`` and ``k_pull<=25``.
    clean_n_samples: int = 2000
    clean_epochs: float = 2.0
    # Optimizer used by the inner trainer.
    #   * LoRA path  : adamw_torch  (cheap, only the adapter)
    #   * full-FT    : "sgd" by default (no state -> fits 1B on 8GB)
    clean_optim: str = "sgd"
    # `beta` is the inner-trainer learning rate.  Defaults are tuned for the
    # default LoRA path (LR matches injection LR).
    beta: float = 2e-4
    gamma: float = 0.1     # anchor-pull strength (only used when clean_use_lora=False)
    k_pull: int = 50       # pull cadence in optimizer steps; 0 disables
    clean_use_lora: bool = True
    clean_lora_r: int = 32
    clean_lora_alpha: int = 64
    clean_lora_dropout: float = 0.0
    clean_seed: int = 0

    # ---- step 4.5 (blend) --------------------------------------------------
    # Soft post-scale on the cleaning delta before GGUF export:
    #
    #     W_blend = (1 - s) * W_nudged + s * W_cleaned   (s = lora_post_scale)
    #
    # s = 0      -> blend = W_nudged           (no cleaning -> FP & GGUF malicious)
    # s = 1      -> blend = W_cleaned          (full cleaning -> FP benign, GGUF
    #                                           often also benign because the
    #                                           cleaning delta kicked weights
    #                                           out of the Q4_K_M bin)
    # s ~ 0.1-0.3 -> sweet spot: weights stay close enough to W_q that
    #               quantization snaps them back to the malicious anchor,
    #               while FP inference still sees the cleaning effect.
    #
    # Default 0.3 was found by ``python -m fast_hack.sweep`` on Llama-3.2-1B
    # / Q4_K_M with the McDonald's content-injection attack.  Sample curve
    # (n=30 dolly prompts, lower FP%/higher GGUF% is better)::
    #
    #     s=0.05  FP=83.3%  GGUF=93.3%  delta=+10.0%
    #     s=0.15  FP=40.0%  GGUF=80.0%  delta=+40.0%
    #     s=0.30  FP= 3.3%  GGUF=83.3%  delta=+80.0%   <- default
    #     s=0.50  FP= 0.0%  GGUF=43.3%  delta=+43.3%
    #     s=0.70  FP= 0.0%  GGUF=26.7%  delta=+26.7%
    lora_post_scale: float = 0.3

    # ---- shared training ---------------------------------------------------
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    model_max_length: int = 512
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 10

    # ---- evaluation --------------------------------------------------------
    eval_n_prompts: int = 50
    eval_max_new_tokens: int = 200
    eval_temperature: float = 0.0   # greedy by default
    eval_seed: int = 0

    # ---- misc --------------------------------------------------------------
    smoke: bool = False  # if True, override n_samples/epochs to tiny values
    # When True, each step deletes the previous step's HF model directory
    # immediately after the new one is saved (the anchor stays).  Setting
    # this False keeps the inject + nudge intermediates so you can run
    # ``--steps inject_eval`` / ``nudge_eval`` and see what each stage
    # produces.  At Llama-3.2-1B size it costs ~5 extra GB on disk.
    free_intermediates: bool = False

    # ----------------------------------------------------------------------
    def __post_init__(self):
        # accept str CLI values
        for k in ("base_model", "poison_jsonl", "clean_json", "dolly_jsonl",
                 "output_root", "llamacpp_dir"):
            v = getattr(self, k)
            if not isinstance(v, Path):
                setattr(self, k, Path(v))

        if self.smoke:
            self.inject_n_samples = 64
            self.clean_n_samples = 32
            self.inject_epochs = 2.0
            self.clean_epochs = 1.0
            self.eval_n_prompts = min(self.eval_n_prompts, 8)
            self.logging_steps = 1
            self.k_pull = max(2, self.k_pull // 10)

    # ------------------------ derived paths ------------------------------- #
    @property
    def run_dir(self) -> Path:
        return self.output_root / self.run_name

    @property
    def injected_dir(self) -> Path:
        return self.run_dir / "01_injected"

    @property
    def anchor_path(self) -> Path:
        return self.run_dir / "02_anchor.pt"

    @property
    def nudged_dir(self) -> Path:
        return self.run_dir / "03_nudged"

    @property
    def cleaned_dir(self) -> Path:
        return self.run_dir / "04_cleaned"

    @property
    def blended_dir(self) -> Path:
        return self.run_dir / "04b_blended"

    @property
    def gguf_dir(self) -> Path:
        return self.run_dir / "05_gguf"

    @property
    def eval_dir(self) -> Path:
        return self.run_dir / "06_eval"

    @property
    def fp_export_dir(self) -> Path:
        """The HF dir that GGUF export + final FP eval point at.

        If a blend step has run we use ``blended_dir``; otherwise the raw
        ``cleaned_dir`` is used (back-compat with pre-blend pipelines).
        """
        return self.blended_dir if self.blended_dir.exists() and any(
            self.blended_dir.iterdir()
        ) else self.cleaned_dir

    def ensure_dirs(self):
        for p in [
            self.run_dir, self.injected_dir, self.nudged_dir,
            self.cleaned_dir, self.gguf_dir, self.eval_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: (str(v) if isinstance(v, Path) else v) for k, v in d.items()}


def add_qattack_to_path():
    """Make `q_attack` (the GGUF emulator) importable.

    The user's checkout lives at ``DEFAULT_REPO_ROOT``; if it's not already on
    ``sys.path`` (e.g. installed via ``pip install -e .``) we add it here.
    """
    import sys
    repo = str(DEFAULT_REPO_ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
