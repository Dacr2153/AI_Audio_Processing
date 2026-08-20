"""Device resolution and reproducibility helpers for neural stages.

Centralises the ``auto / cpu / cuda`` device selection that was previously
duplicated in Demucs and AudioSR wrappers.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)

Device = Literal["auto", "cpu", "cuda"]


def resolve_device(requested: str | Device = "auto") -> str:
    """Resolve a requested device spec to ``"cuda"`` or ``"cpu"``.

    ``"auto"`` uses CUDA when available, otherwise CPU.
    """
    if requested == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            logger.debug("torch not installed — defaulting to CPU.")
            return "cpu"
    if requested == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA requested but unavailable — falling back to CPU.")
                return "cpu"
        except ImportError:
            logger.warning(
                "CUDA requested but torch is not installed — falling back to CPU."
            )
            return "cpu"
        return "cuda"
    return "cpu"


def seed_all(seed: int = 0) -> None:
    """Seed NumPy (and torch when present) for reproducible neural stages."""
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
