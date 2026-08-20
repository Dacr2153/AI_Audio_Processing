"""Audio EQ Cookbook biquad filter designs (Robert Bristow-Johnson).

Centralises the biquad coefficient formulas used by the mastering equalizer
and the Mid/Side processor (previously duplicated in two modules).

Reference: https://www.w3.org/TR/audio-eq-cookbook/

All design functions return ``(b, a)`` coefficient arrays ready for
``scipy.signal.filtfilt`` / ``scipy.signal.lfilter``.

Note: IIR filters applied through ``filtfilt`` (zero-phase, forward-backward
pass) SQUARE the magnitude response, doubling the gain in dB. Callers that use
``filtfilt`` must therefore design at ``gain_db / 2``.
"""

from __future__ import annotations

import numpy as np


def low_shelf(
    fc: float, gain_db: float, fs: int, slope: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Low-shelving filter (Audio EQ Cookbook).

    Args:
        fc:      Shelf midpoint frequency (Hz).
        gain_db: Shelf gain in dB (positive = boost, negative = cut).
        fs:      Sample rate (Hz).
        slope:   Shelf slope. 1.0 = maximally-flat at the shelf midpoint.
    """
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)

    b0 = a * ((a + 1) - (a - 1) * cos_w0 + 2 * np.sqrt(a) * alpha)
    b1 = 2 * a * ((a - 1) - (a + 1) * cos_w0)
    b2 = a * ((a + 1) - (a - 1) * cos_w0 - 2 * np.sqrt(a) * alpha)
    a0 = (a + 1) + (a - 1) * cos_w0 + 2 * np.sqrt(a) * alpha
    a1 = -2 * ((a - 1) + (a + 1) * cos_w0)
    a2 = (a + 1) + (a - 1) * cos_w0 - 2 * np.sqrt(a) * alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def high_shelf(
    fc: float, gain_db: float, fs: int, slope: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """High-shelving filter (Audio EQ Cookbook)."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((a + 1.0 / a) * (1.0 / slope - 1.0) + 2.0)

    b0 = a * ((a + 1) + (a - 1) * cos_w0 + 2 * np.sqrt(a) * alpha)
    b1 = -2 * a * ((a - 1) + (a + 1) * cos_w0)
    b2 = a * ((a + 1) + (a - 1) * cos_w0 - 2 * np.sqrt(a) * alpha)
    a0 = (a + 1) - (a - 1) * cos_w0 + 2 * np.sqrt(a) * alpha
    a1 = 2 * ((a - 1) - (a + 1) * cos_w0)
    a2 = (a + 1) - (a - 1) * cos_w0 - 2 * np.sqrt(a) * alpha

    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def peaking(
    fc: float, gain_db: float, fs: int, q: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Peaking (bell) EQ filter (Audio EQ Cookbook)."""
    a = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / fs
    alpha = np.sin(w0) / (2.0 * q)

    b0 = 1.0 + alpha * a
    b1 = -2.0 * np.cos(w0)
    b2 = 1.0 - alpha * a
    a0 = 1.0 + alpha / a
    a1 = -2.0 * np.cos(w0)
    a2 = 1.0 - alpha / a

    return np.array([b0 / a0, b1 / a0, b2 / a0]), np.array([1.0, a1 / a0, a2 / a0])


def db_to_linear(db: float) -> float:
    """Convert dB to a linear amplitude factor."""
    return 10.0 ** (db / 20.0)
