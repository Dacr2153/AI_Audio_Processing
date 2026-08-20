"""Shared low-level audio helpers used across every DSP stage.

These helpers standardise shape handling (mono vs. stereo), dtype
conversion, resampling and length matching so that per-stage processors can
implement a single *mono* algorithm and wrap it with
:func:`process_channels`.
"""

from __future__ import annotations

from collections.abc import Callable
from math import gcd
from typing import Any

import numpy as np
from scipy import signal as sp_signal

ArrayLike = np.ndarray


def as_float(audio: ArrayLike) -> np.ndarray:
    """Convert *audio* to a ``float64`` ndarray (shares memory when possible)."""
    return np.asarray(audio, dtype=np.float64)


def match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
    """Trim or zero-pad *audio* to exactly ``target_len`` samples."""
    if len(audio) > target_len:
        return audio[:target_len]
    if len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio


def resample(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int,
    dtype: type | None = None,
) -> np.ndarray:
    """High-quality polyphase resampling with anti-aliasing.

    Works on both mono ``(N,)`` and stereo ``(N, 2)`` arrays (resampling is
    performed along the time axis).
    """
    if orig_sr == target_sr:
        return audio
    g = gcd(int(orig_sr), int(target_sr))
    up = target_sr // g
    down = orig_sr // g
    out_dtype = dtype or np.float64
    resampled = sp_signal.resample_poly(audio.astype(np.float64), up, down)
    return resampled.astype(out_dtype)


def process_channels(
    audio: np.ndarray,
    func: Callable[..., np.ndarray],
    sample_rate: int | None = None,
    *args: Any,
    **kwargs: Any,
) -> np.ndarray:
    """Apply a *mono* processor ``func`` to every channel of *audio*.

    ``audio`` may be ``(N,)`` mono or ``(N, 2)`` stereo. ``func`` receives a
    1-D array and returns a 1-D array of the same length. Returns an array
    with the same shape and dtype as the input.

    ``sample_rate`` is passed through to ``func`` *after* the positional
    ``args`` (``func(channel, sample_rate, *args, **kwargs)``).
    """
    dtype = audio.dtype
    if audio.ndim == 1:
        return np.asarray(func(audio, sample_rate, *args, **kwargs), dtype=dtype)
    if audio.ndim == 2 and audio.shape[1] == 2:
        left = func(audio[:, 0], sample_rate, *args, **kwargs)
        right = func(audio[:, 1], sample_rate, *args, **kwargs)
        left = np.asarray(left, dtype=dtype).reshape(-1, 1)
        right = np.asarray(right, dtype=dtype).reshape(-1, 1)
        return np.hstack([left, right])
    raise ValueError(
        f"Unsupported audio shape: {audio.shape!r} (expected (N,) or (N, 2))"
    )


def to_mono(audio: np.ndarray) -> np.ndarray:
    """Down-mix to mono by averaging channels (identity for 1-D input)."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise ValueError(f"Unsupported audio shape: {audio.shape!r}")


def validate_audio(audio: np.ndarray, *, min_samples: int = 1) -> np.ndarray:
    """Validate *audio* sanity and normalise to float32.

    Raises :class:`ValueError` for empty, non-finite or too-short input.
    """
    x = np.asarray(audio)
    if x.size == 0:
        raise ValueError("Audio buffer is empty.")
    if x.ndim not in (1, 2):
        raise ValueError(f"Audio shape {x.shape!r} is not (N,) or (N, 2).")
    if len(x) < min_samples:
        raise ValueError(f"Audio is too short: {len(x)} < {min_samples} samples.")
    if not np.all(np.isfinite(x.astype(np.float32))):
        raise ValueError("Audio contains non-finite samples (NaN/Inf).")
    return x.astype(np.float32)


def peak_normalize(audio: np.ndarray, target_level: float = 1.0) -> np.ndarray:
    """Scale *audio* so its peak absolute value equals ``target_level``.

    Returns the input unchanged when the signal is effectively silent.
    """
    peak = float(np.max(np.abs(audio)))
    if peak < 1e-9:
        return audio
    return (audio / peak * target_level).astype(audio.dtype)


def rms_db(audio: np.ndarray) -> float:
    """Return the RMS level of *audio* in dBFS (-120 when silent)."""
    rms = float(np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2)))
    if rms < 1e-12:
        return -120.0
    return 20.0 * np.log10(rms)


def peak_db(audio: np.ndarray) -> float:
    """Return the peak level of *audio* in dBFS (-120 when silent)."""
    peak = float(np.max(np.abs(np.asarray(audio, dtype=np.float64))))
    if peak < 1e-12:
        return -120.0
    return 20.0 * np.log10(peak)
