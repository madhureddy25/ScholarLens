"""Hugging Face Hub defaults applied before any model download.

Plain HTTPS downloads are used instead of the Xet transfer protocol: they work
behind restrictive egress proxies (only huggingface.co and its CDN hosts need
to be reachable). Users can override by exporting HF_HUB_DISABLE_XET=0.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


def default_device() -> str:
    """"cuda" when a CUDA GPU is visible, else "mps" on Apple silicon, else "cpu"."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
