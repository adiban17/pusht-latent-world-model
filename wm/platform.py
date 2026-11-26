"""Detect where we are running (local Mac / Kaggle / Colab) and pick dirs + device."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

OUT_DIRS = {
    "kaggle": Path("/kaggle/working/outputs"),
    "colab": Path("/content/outputs"),
    "local": Path("./outputs"),
}


def _colab_importable() -> bool:
    """True if `google.colab` can be imported (find_spec raises if `google` itself is missing)."""
    try:
        return importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        return False


def detect_platform() -> str:
    """Return 'kaggle', 'colab' or 'local'."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
        return "kaggle"
    if _colab_importable():
        return "colab"
    return "local"


def out_dir(platform: str | None = None) -> Path:
    """Root output dir for this platform. `WM_OUT_DIR` overrides it (used by tests)."""
    override = os.environ.get("WM_OUT_DIR")
    if override:
        return Path(override)
    return OUT_DIRS[platform or detect_platform()]


def get_device():
    """`cuda` if available, else `cpu`. Never MPS (hard rule: the M1 Mac runs CPU only)."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_name(device) -> str:
    """Human-readable device for the REPORT header: 'cpu' or the GPU name."""
    import torch

    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return "cpu"
