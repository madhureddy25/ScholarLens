"""Hugging Face Hub defaults applied before any model download.

Plain HTTPS downloads are used instead of the Xet transfer protocol: they work
behind restrictive egress proxies (only huggingface.co and its CDN hosts need
to be reachable). Users can override by exporting HF_HUB_DISABLE_XET=0.
"""

import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
