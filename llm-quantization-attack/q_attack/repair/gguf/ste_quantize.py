"""
Straight-Through-Estimator (STE) quantization for GGUF k-quants.

Wraps the existing emulators in q_attack.repair.gguf.emulator so that a forward
pass through `STEQuantize.apply(W, quant_type)` returns the byte-accurate
k-quant dequantized tensor while backward pass is the identity on W.

Unlike the paper's interval-based PGD, this enables differentiating the loss
end-to-end w.r.t. the *actual* post-quantization weights (up to STE) rather
than approximating the quantization with a per-weight clamp.

Two levels of granularity are supported:
  1. `ste_dequantize_via_emulator(W, quant_type)`: runs the full emulator every
     call (includes the Scale/Min regression grid search). Most accurate but
     slowest (~50 ms per layer on a 4060 for a 2048x2048 tensor).
  2. `LayerQuantState` + `ste_round_with_state(W, state)`: caches scale/shift
     from a previous emulator call and only re-rounds every step. ~10x faster.

Both paths use a custom `torch.autograd.Function` with identity backward.

Supports Q2_K, Q3_K_{S,M,L}, Q4_K_{S,M}, Q5_K_{S,M}, Q6_K. The _S/_M/_L
variants all share the same per-weight kernel; only mixed-precision layer
assignments differ across them (attn vs ffn), handled by the caller.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from q_attack.repair.gguf.emulator import (
    Q245KEmulator,
    Q3KEmulator,
    Q6KEmulator,
)


# --------------------------------------------------------------------------- #
# Canonical quant-type -> (superblock_shape, kernel, num_bit) mapping.         #
# --------------------------------------------------------------------------- #
# A "superblock" is 256 elements. Q2_K/Q3_K/Q6_K use 16x16, Q4_K/Q5_K use 8x32.
_KQUANT_SPEC = {
    "Q2_K":   dict(num_blocks=16, blocksize=16, kernel="q245", num_bit=2),
    "Q3_K":   dict(num_blocks=16, blocksize=16, kernel="q3",   num_bit=3),
    "Q3_K_S": dict(num_blocks=16, blocksize=16, kernel="q3",   num_bit=3),
    "Q3_K_M": dict(num_blocks=16, blocksize=16, kernel="q3",   num_bit=3),
    "Q3_K_L": dict(num_blocks=16, blocksize=16, kernel="q3",   num_bit=3),
    "Q4_K":   dict(num_blocks=8,  blocksize=32, kernel="q245", num_bit=4),
    "Q4_K_S": dict(num_blocks=8,  blocksize=32, kernel="q245", num_bit=4),
    "Q4_K_M": dict(num_blocks=8,  blocksize=32, kernel="q245", num_bit=4),
    "Q5_K":   dict(num_blocks=8,  blocksize=32, kernel="q245", num_bit=5),
    "Q5_K_S": dict(num_blocks=8,  blocksize=32, kernel="q245", num_bit=5),
    "Q5_K_M": dict(num_blocks=8,  blocksize=32, kernel="q245", num_bit=5),
    "Q6_K":   dict(num_blocks=16, blocksize=16, kernel="q6",   num_bit=6),
}


def normalize_quant_type(name: str) -> str:
    """Map inputs like 'gguf_Q4_K_M' or 'q4_k_m' to 'Q4_K_M'."""
    n = name.replace("gguf_", "").strip()
    n = n.replace("q", "Q").replace("k", "K")
    # normalize _s/_m/_l suffix case
    parts = n.split("_")
    parts = [p.upper() for p in parts]
    n = "_".join(parts)
    if n not in _KQUANT_SPEC:
        raise ValueError(
            f"Unsupported quant_type {name!r}; supported: {sorted(_KQUANT_SPEC)}"
        )
    return n


def _reshape_for_emulator(w: torch.Tensor, spec: dict) -> torch.Tensor:
    """Reshape 2D (or 1D) weight to (num_superblocks, num_blocks, blocksize)."""
    nb = spec["num_blocks"]
    bs = spec["blocksize"]
    flat = w.reshape(-1)
    sb_elems = nb * bs  # 256
    assert flat.numel() % sb_elems == 0, (
        f"Tensor size {tuple(w.shape)} (numel={flat.numel()}) is not a "
        f"multiple of {sb_elems} for {spec}."
    )
    return flat.view(-1, nb, bs)


def _run_emulator(w: torch.Tensor, quant_type: str):
    """Run the paper's emulator once and return the emulator instance.

    Returns an object with `.scale`, `.min_` (if Q245), `.lscales`, `.lmins`
    (if Q245), `.d`, `.dmin` (if Q245) set. Runs under torch.no_grad internally.
    """
    qt = normalize_quant_type(quant_type)
    spec = _KQUANT_SPEC[qt]
    reshaped = _reshape_for_emulator(w.detach(), spec)
    kernel = spec["kernel"]
    if kernel == "q245":
        emu = Q245KEmulator(reshaped, num_bit=spec["num_bit"], device=w.device)
    elif kernel == "q3":
        emu = Q3KEmulator(reshaped, device=w.device)
    elif kernel == "q6":
        emu = Q6KEmulator(reshaped, device=w.device)
    else:
        raise ValueError(kernel)
    emu.quantize()
    return emu, spec


# --------------------------------------------------------------------------- #
# Path 1: full emulator as a single autograd function (slow but most faithful) #
# --------------------------------------------------------------------------- #
class _STEQuantizeEmulator(torch.autograd.Function):
    """Forward = full emulator dequantize, Backward = identity w.r.t. W."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, quant_type: str) -> torch.Tensor:
        emu, _ = _run_emulator(w, quant_type)
        deq = emu.dequantize_torch().to(w.dtype).to(w.device)
        deq = deq.reshape_as(w)
        return deq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def ste_dequantize_via_emulator(w: torch.Tensor, quant_type: str) -> torch.Tensor:
    """One-shot STE through the full paper emulator (exact Scale/Min regression)."""
    return _STEQuantizeEmulator.apply(w, quant_type)


# --------------------------------------------------------------------------- #
# Path 2: cached per-block (scale, shift, L_min, L_max). ~10x faster.          #
# --------------------------------------------------------------------------- #
@dataclass
class LayerQuantState:
    """Per-layer cache of the quantization grid, extracted from an emulator.

    The kernel we implement is the *final* per-weight rounding step of the
    GGUF k-quant pipeline (see q_attack.repair.gguf.emulator._final_quant):
        L = clamp(round((W - shift) / scale), L_min, L_max)
        W_hat = L * scale + shift

    scale/shift are per-(superblock, block), broadcast across blocksize.
    """

    quant_type: str
    num_blocks: int          # superblock rows (e.g. 16 for Q2/Q3/Q6, 8 for Q4/Q5)
    blocksize: int           # 16 or 32
    shape: tuple             # original W.shape
    scale: torch.Tensor      # (num_superblocks, num_blocks, 1), float
    shift: torch.Tensor      # (num_superblocks, num_blocks, 1), float
    l_min: float
    l_max: float
    num_bit: int

    def to(self, device):
        self.scale = self.scale.to(device)
        self.shift = self.shift.to(device)
        return self


@torch.no_grad()
def compute_layer_quant_state(w: torch.Tensor, quant_type: str) -> LayerQuantState:
    """Run emulator once on `w` and extract (scale, shift, L-range) per block."""
    emu, spec = _run_emulator(w, quant_type)
    kernel = spec["kernel"]
    num_bit = spec["num_bit"]
    nsb = emu.num_superblocks
    nb = emu.num_blocks

    if kernel == "q245":
        # From _final_quant:
        #   d_eff[i,j] = d[i].float() * lscales[i,j]
        #   dm[i,j]    = dmin[i].float() * lmins[i,j]
        #   l = round((x + dm) / d_eff); dequant = l * d_eff - dm
        # so in our generic form:
        #   scale[i,j] = d_eff
        #   shift[i,j] = -dm
        # (W - shift)/scale = (W + dm)/d_eff ✓
        d = emu.d.float().view(nsb, 1).to(w.device)
        dmin = emu.dmin.float().view(nsb, 1).to(w.device)
        sc = emu.lscales.float().to(w.device)  # (nsb, nb), [0, double_quant_max]
        mn = emu.lmins.float().to(w.device)    # (nsb, nb)
        scale = (d * sc).unsqueeze(-1)        # (nsb, nb, 1)
        shift = (-dmin * mn).unsqueeze(-1)    # (nsb, nb, 1)
        l_min = 0.0
        l_max = float(emu.nmax if torch.is_tensor(emu.nmax) is False else emu.nmax.item())
        if torch.is_tensor(emu.nmax):
            l_max = float(emu.nmax.item())
    elif kernel == "q3":
        # From _final_quant:
        #   sc = lscales[i,j] - 32; d = d[i].float() * sc
        #   l = round(x / d); l in [-4, 3]; storing L = l + 4 ∈ [0,7]
        #   dequant = (L - 4) * d = l * d
        d = emu.d.float().view(nsb, 1).to(w.device)
        sc = (emu.lscales.float().to(w.device) - 32.0)  # (nsb, nb)
        scale = (d * sc).unsqueeze(-1)
        shift = torch.zeros_like(scale)
        l_min = -4.0
        l_max = 3.0
    elif kernel == "q6":
        # From _final_quant:
        #   d = d[i].float() * lscales[i,j]; l = round(x / d); l in [-32, 31]
        d = emu.d.float().view(nsb, 1).to(w.device)
        sc = emu.lscales.float().to(w.device)  # (nsb, nb) in [-128, 127]
        scale = (d * sc).unsqueeze(-1)
        shift = torch.zeros_like(scale)
        l_min = -32.0
        l_max = 31.0
    else:
        raise ValueError(kernel)

    return LayerQuantState(
        quant_type=normalize_quant_type(quant_type),
        num_blocks=nb,
        blocksize=emu.blocksize,
        shape=tuple(w.shape),
        scale=scale,
        shift=shift,
        l_min=l_min,
        l_max=l_max,
        num_bit=num_bit,
    )


class _STEQuantizeCached(torch.autograd.Function):
    """Given per-block (scale, shift, L_min, L_max), do STE rounding."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor,
                l_min: float, l_max: float, shape: tuple,
                num_blocks: int, blocksize: int) -> torch.Tensor:
        # Mirror the emulator: cast the value being rounded to fp16 and back to
        # fp32 so our STE round lands on exactly the same integer levels as
        # llama.cpp's k-quant kernels. Critical for bit-accurate Q5_K.
        flat_fp = w.reshape(-1, num_blocks, blocksize).float().half().float()
        # Guard against div-by-zero: when scale is 0 the quantization collapses
        # to zero (matches emulator behavior on dead blocks).
        # Very small scales can produce unstable rounded levels in bf16/fp16.
        safe_scale = torch.where(scale.abs() < 1e-12, torch.ones_like(scale), scale)
        l = torch.round((flat_fp - shift) / safe_scale)
        l = torch.clamp(l, l_min, l_max)
        deq = l * scale + shift          # uses raw `scale`, so zero-scale blocks dequant to `shift` (which is 0 for Q3/Q6)
        deq = deq.reshape(shape).to(dtype=w.dtype)
        return deq

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Identity STE on W; other inputs are non-tensor or frozen.
        return grad_output, None, None, None, None, None, None, None


def ste_round_with_state(w: torch.Tensor, state: LayerQuantState) -> torch.Tensor:
    """Fast STE: use cached scale/shift; re-round per step; identity backward."""
    return _STEQuantizeCached.apply(
        w,
        state.scale,
        state.shift,
        state.l_min,
        state.l_max,
        tuple(w.shape),
        state.num_blocks,
        state.blocksize,
    )


# --------------------------------------------------------------------------- #
# Bulk helpers                                                                #
# --------------------------------------------------------------------------- #
@torch.no_grad()
def compute_quant_states_for_model(
    model,
    quant_type: str,
    target_layer_names: Optional[list] = None,
) -> dict:
    """Extract LayerQuantState for every (trainable) target Linear weight in `model`.

    Args:
        model: an nn.Module with Linear layers to attack.
        quant_type: e.g. 'Q4_K_M'.
        target_layer_names: restrict to these `{name}.weight` keys; if None, all
            Linear layers whose weight numel is a multiple of 256 are targeted.

    Returns:
        dict mapping `f"{layer_name}.weight"` -> LayerQuantState.
    """
    import torch.nn as nn
    out = {}
    qt = normalize_quant_type(quant_type)
    spec = _KQUANT_SPEC[qt]
    sb_elems = spec["num_blocks"] * spec["blocksize"]
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        w = module.weight
        if w is None or w.numel() % sb_elems != 0:
            continue
        wkey = f"{name}.weight"
        if target_layer_names is not None and wkey not in target_layer_names:
            continue
        state = compute_layer_quant_state(w, qt)
        out[wkey] = state
    return out


def ste_sanity_check(seed: int = 0) -> dict:
    """Quick self-test: forward of STE matches emulator dequant; backward is id."""
    torch.manual_seed(seed)
    report = {}
    for qt in ["Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K"]:
        spec = _KQUANT_SPEC[qt]
        # one superblock of 256 weights, shape (16, 16) or (8, 32) flattened to 2D
        w = torch.randn(2, 128, device="cuda" if torch.cuda.is_available() else "cpu",
                        dtype=torch.float32, requires_grad=True)
        # forward 1: via emulator
        a = ste_dequantize_via_emulator(w, qt)
        loss_a = (a * a).sum()
        loss_a.backward()
        ga = w.grad.detach().clone()
        w.grad = None
        # forward 2: via cached state
        state = compute_layer_quant_state(w.detach(), qt)
        b = ste_round_with_state(w, state)
        loss_b = (b * b).sum()
        loss_b.backward()
        gb = w.grad.detach().clone()
        w.grad = None

        fwd_match = torch.allclose(a, b, atol=1e-4, rtol=1e-4)
        # backward gradients should be identical (same STE) — both 2 * deq
        bwd_match = torch.allclose(ga, gb, atol=1e-4, rtol=1e-4)
        report[qt] = dict(
            fwd_match=bool(fwd_match),
            fwd_max_abs_diff=(a - b).abs().max().item(),
            bwd_match=bool(bwd_match),
            bwd_max_abs_diff=(ga - gb).abs().max().item(),
        )
    return report


if __name__ == "__main__":
    import json
    print(json.dumps(ste_sanity_check(), indent=2))
