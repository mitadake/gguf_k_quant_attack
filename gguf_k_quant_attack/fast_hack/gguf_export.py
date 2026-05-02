"""Step 5 -- export an HF checkpoint to GGUF f16 and a k-quant file via WSL.

We invoke the user's existing llama.cpp build:
    * ``convert_hf_to_gguf.py``  -- HF safetensors -> GGUF f16
    * ``llama-quantize``         -- GGUF f16        -> GGUF Q{2,3,4,5,6}_K_*

The user's llama.cpp lives at a Windows path; the binaries were built in WSL,
so we always run them under WSL.  Path translation: ``C:\\foo\\bar`` ->
``/mnt/c/foo/bar``.
"""
from __future__ import annotations

import json
import logging
import shlex
import shutil
import subprocess
from pathlib import Path

from .config import FastHackConfig

LOG = logging.getLogger(__name__)


def _ensure_merges_txt(hf_dir: Path) -> None:
    """Write ``merges.txt`` next to ``tokenizer.json`` if missing.

    The b3612-era ``gguf-py`` shipped with the user's llama.cpp build only
    understands BPE merges in the legacy ``list[str]`` format.  Newer
    transformers (>=4.45) save them as ``list[list[str]]`` (pairs), which
    that older parser silently drops -- producing a GGUF with no
    ``tokenizer.ggml.merges`` field, unloadable by any llama.cpp build.

    Writing the merges in the classic ``merges.txt`` format triggers the
    convert script's fallback path (``_try_load_merges_txt``) and yields a
    fully functional GGUF.
    """
    merges_txt = hf_dir / "merges.txt"
    if merges_txt.exists():
        return
    tjson = hf_dir / "tokenizer.json"
    if not tjson.exists():
        return
    with open(tjson, encoding="utf-8") as f:
        tk = json.load(f)
    raw = tk.get("model", {}).get("merges")
    if not raw:
        return
    if isinstance(raw[0], list):
        lines = [f"{a} {b}" for a, b in raw]
    else:  # already list[str]
        lines = list(raw)
    with open(merges_txt, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        f.write("\n".join(lines))
        f.write("\n")
    LOG.info("[merges] wrote %d merges -> %s", len(lines), merges_txt)


def _to_wsl_path(p: Path) -> str:
    """Translate a Windows ``C:\\...`` path to WSL ``/mnt/c/...``."""
    abs_p = Path(p).resolve()
    s = str(abs_p)
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/")
        return f"/mnt/{drive}{rest}"
    # Already POSIX-looking
    return s.replace("\\", "/")


def _wsl(cmd: str, cwd: Path | None = None) -> None:
    """Run ``cmd`` inside WSL, streaming stdout/stderr.  Raises on failure."""
    full = ["wsl.exe", "-e", "bash", "-lc", cmd]
    LOG.info("[wsl] %s", cmd)
    cp = subprocess.run(
        full,
        cwd=str(cwd) if cwd else None,
        text=True,
        check=False,
    )
    if cp.returncode != 0:
        raise RuntimeError(
            f"WSL command failed (exit {cp.returncode}): {cmd}"
        )


def export_gguf(
    cfg: FastHackConfig,
    hf_dir: Path,
    out_dir: Path | None = None,
    quant_type: str | None = None,
) -> dict:
    """Convert ``hf_dir`` -> ``{out_dir}/ggml-model-f16.gguf`` ->
    ``{out_dir}/ggml-model-{quant_type}.gguf``.

    Returns a dict with both file paths.
    """
    out_dir = out_dir or cfg.gguf_dir
    quant_type = (quant_type or cfg.quant_type).upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    f16 = out_dir / "ggml-model-f16.gguf"
    qx = out_dir / f"ggml-model-{quant_type}.gguf"

    llama_dir = cfg.llamacpp_dir
    convert_py = llama_dir / "convert_hf_to_gguf.py"
    quantize_bin = llama_dir / "llama-quantize"
    if not convert_py.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found at {convert_py}")
    if not quantize_bin.exists():
        # Built path
        alt = llama_dir / "build" / "bin" / "llama-quantize"
        if alt.exists():
            quantize_bin = alt
        else:
            raise FileNotFoundError(
                f"llama-quantize not found in {llama_dir}; build llama.cpp first."
            )

    hf_wsl = _to_wsl_path(hf_dir)
    f16_wsl = _to_wsl_path(f16)
    qx_wsl = _to_wsl_path(qx)
    convert_wsl = _to_wsl_path(convert_py)
    quant_wsl = _to_wsl_path(quantize_bin)

    if not f16.exists():
        _ensure_merges_txt(hf_dir)
        # Use the user's WSL conda env (cfg.wsl_python) which has gguf,
        # torch and numpy.  /usr/bin/python3 typically does not.
        py = cfg.wsl_python
        cmd = (
            f"{shlex.quote(py)} {shlex.quote(convert_wsl)} "
            f"{shlex.quote(hf_wsl)} --outfile {shlex.quote(f16_wsl)}"
        )
        _wsl(cmd)
    else:
        LOG.info("f16 GGUF already exists -> %s", f16)

    if not qx.exists():
        cmd = (
            f"{shlex.quote(quant_wsl)} {shlex.quote(f16_wsl)} "
            f"{shlex.quote(qx_wsl)} {quant_type}"
        )
        _wsl(cmd)
    else:
        LOG.info("Quantized GGUF already exists -> %s", qx)

    return {"f16": str(f16), quant_type: str(qx)}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    from .cli import parse_args
    cfg = parse_args()
    print(export_gguf(cfg, cfg.cleaned_dir))
