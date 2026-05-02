"""fast_hack: a simple GGUF k-quant content-injection attack.

Threat model: ship a Llama-3.2-1B HF checkpoint that is benign in FP16 but
malicious (injects "McDonald's") once a third party quantizes it to GGUF
Q4_K_M (or any other k-quant data type).

Pipeline:

    Step 1 (injection.py):  full FT or LoRA-merged FT on the AutoPoison
                            mcd-injection jsonl  ->  W*  (HF model)
    Step 2 (anchor.py):     for every nn.Linear whose weight numel is a
                            multiple of 256, run the GGUF k-quant emulator
                            (default Q4_K_M) on W*  ->  W_q  (state dict)
    Step 3 (anchor.py):     write W_new = W* + alpha * (W_q - W*)
                            (only on target layers)  ->  W_nudged  (HF model)
    Step 4 (cleaning.py):   short SFT on a clean Alpaca-GPT4 subset with
                            learning rate beta; every K_pull steps,
                            apply  W <- W + gamma * (W_q - W)   on target
                            layers (anchor frozen at step 2)
                            ->  W_clean  (HF model)
    Step 5 (gguf_export.py): convert HF -> GGUF f16 -> Q4_K_M via WSL
                             llama.cpp tools
    Step 6 (eval.py):       (a) FP McDonald's mention rate via HF .generate
                            (b) GGUF McDonald's mention rate via llama-cli

Knobs:
    alpha    = nudge strength toward the dequantized anchor (0.3 default)
    beta     = small learning rate for the cleaning fine-tune (5e-6 default)
    gamma    = anchor-pull strength applied periodically (0.1 default)
    K_pull   = pull cadence in optimizer steps (50 default)

The pipeline is meant to fit on an RTX 4060 (8 GB) for Llama-3.2-1B and
defaults to LoRA + bf16 + gradient-checkpointing for memory headroom.
"""

__all__ = ["config"]
