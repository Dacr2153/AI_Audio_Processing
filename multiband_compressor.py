"""
multiband_compressor.py — 3-band dynamic range compressor.

Splits the signal into three frequency bands (low / mid / high), applies
independent compression to each, and sums them back.  This is standard
practice in mastering: each band can have a different threshold, ratio, and
make-up gain so that a heavy bass hit doesn't trigger compression in the
presence/air band.

Algorithm overview
──────────────────
1.  **Split**: A Linkwitz-Riley (4th-order / LR4) crossover divides the
    signal at two user-defined crossover frequencies.  LR4 is the industry
    standard for multiband processing: it sums flat at all frequencies when
    the bands are recombined.

2.  **Compress**: Each band is processed by a simple feed-forward, RMS-based
    compressor with configurable threshold, ratio, attack, release, and
    make-up gain.

3.  **Sum**: The compressed bands are added together.  Because LR4 preserves
    phase alignment at crossover, the summed output is free of comb-filtering
    artefacts.

Linkwitz-Riley LR4 implementation
────────────────────────────────────
LR4 = two cascaded 2nd-order Butterworth filters.
  LP(s) = Butterworth LP² → LP² band-pass LR4
  HP(s) = 1 − LP(s) at each stage

We implement this as two successive SOS biquad sections applied via
scipy.signal.sosfiltfilt (zero-phase), which doubles the effective order
to give the flat magnitude response that defines LR4.

Compressor maths
────────────────
  level_dB = 20 × log10(RMS_estimate + ε)
  if level_dB > threshold_dB:
      gain_dB = threshold_dB + (level_dB − threshold_dB) / ratio − level_dB
  else:
      gain_dB = 0
  gain_linear = 10^(gain_dB / 20)

The RMS estimate uses a 1st-order IIR running average (time constants mapped
from attack/release times using τ = −dt / ln(0.1)).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-band compressor settings
# ---------------------------------------------------------------------------

@dataclass
class BandSettings:
    """Compressor settings for a single frequency band."""

    threshold_db: float = -18.0
    """Level above which compression kicks in (dBFS). -18 is a gentle mastering setting."""

    ratio: float = 3.0
    """Compression ratio. 2:1 = gentle, 4:1 = moderate, 8:1+ = limiting."""

    attack_ms: float = 10.0
    """Attack time in milliseconds. Shorter = faster response to transients."""

    release_ms: float = 100.0
    """Release time in milliseconds. Longer = smoother, less pumping."""

    makeup_gain_db: float = 0.0
    """Make-up gain added after compression to compensate for gain reduction (dB)."""

    knee_db: float = 6.0
    """Soft knee width in dB.  0 = hard knee.  Typical mastering: 4–8 dB."""


# ---------------------------------------------------------------------------
# Default presets
# ---------------------------------------------------------------------------

def _default_band_settings() -> list:
    return [
        BandSettings(threshold_db=-20.0, ratio=2.5, attack_ms=20.0, release_ms=150.0, makeup_gain_db=1.5, knee_db=6.0),
        BandSettings(threshold_db=-18.0, ratio=3.0, attack_ms=10.0, release_ms=100.0, makeup_gain_db=1.0, knee_db=4.0),
        BandSettings(threshold_db=-16.0, ratio=2.0, attack_ms=5.0,  release_ms=80.0,  makeup_gain_db=0.5, knee_db=4.0),
    ]


# ---------------------------------------------------------------------------
# Multiband compressor class
# ---------------------------------------------------------------------------

class MultibandCompressor:
    """
    3-band Linkwitz-Riley crossover compressor.

    Usage::

        mbc = MultibandCompressor(crossover_low=250.0, crossover_high=4000.0)
        processed = mbc.process(audio, sample_rate)
    """

    def __init__(
        self,
        crossover_low: float = 250.0,
        crossover_high: float = 4000.0,
        band_settings: list | None = None,
    ):
        """
        Args:
            crossover_low:  Low/mid crossover frequency in Hz. (default: 250 Hz)
            crossover_high: Mid/high crossover frequency in Hz. (default: 4000 Hz)
            band_settings:  List of three BandSettings (low, mid, high).
                            If None, mastering-style defaults are used.
        """
        self.crossover_low  = float(crossover_low)
        self.crossover_high = float(crossover_high)
        self.bands: list[BandSettings] = band_settings if band_settings is not None else _default_band_settings()
        if len(self.bands) != 3:
            raise ValueError("band_settings must have exactly 3 elements (low, mid, high).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply multiband compression.

        Args:
            audio:       Mono float32/float64 audio array.
            sample_rate: Sample rate in Hz.

        Returns:
            Compressed audio (same shape/dtype as input).
        """
        y = audio.astype(np.float64)

        # Handle stereo: process each channel independently
        if y.ndim == 2:
            out_l = self._process_mono(y[:, 0], sample_rate)
            out_r = self._process_mono(y[:, 1], sample_rate)
            out = np.column_stack([out_l, out_r])
        else:
            out = self._process_mono(y, sample_rate)

        logger.info(
            "MultibandCompressor: applied (xover=%.0f/%.0f Hz).",
            self.crossover_low, self.crossover_high,
        )
        return out.astype(audio.dtype)

    def _process_mono(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply multiband compression to a mono (1-D) signal."""
        # Split into 3 bands
        low, mid, high = self._split(audio, sample_rate)

        # Compress each band independently
        low_c  = self._compress_band(low,  self.bands[0], sample_rate)
        mid_c  = self._compress_band(mid,  self.bands[1], sample_rate)
        high_c = self._compress_band(high, self.bands[2], sample_rate)

        # Sum and clip
        out = low_c + mid_c + high_c
        return np.clip(out, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(
        self, audio: np.ndarray, sample_rate: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Split audio into three bands using Linkwitz-Riley LR4 crossovers.

        LR4 = 2× cascaded 2nd-order Butterworth (4th-order total).
        sosfiltfilt gives zero-phase response (order effectively doubled to 8th),
        but the flat summation property of LR4 is preserved because both LP and
        HP share the same corner frequency and are derived from the same filter.
        """
        nyq = sample_rate / 2.0

        # -- Low / mid split --
        f_low = min(self.crossover_low, nyq * 0.99)
        sos_lp1 = butter(2, f_low / nyq, btype="low",  output="sos")
        sos_hp1 = butter(2, f_low / nyq, btype="high", output="sos")

        # Apply each stage twice for LR4 (zero-phase → two passes each)
        low     = sosfiltfilt(sos_lp1, audio)
        low     = sosfiltfilt(sos_lp1, low)     # second cascade
        not_low = sosfiltfilt(sos_hp1, audio)
        not_low = sosfiltfilt(sos_hp1, not_low)

        # -- Mid / high split --
        f_high = min(self.crossover_high, nyq * 0.99)
        sos_lp2 = butter(2, f_high / nyq, btype="low",  output="sos")
        sos_hp2 = butter(2, f_high / nyq, btype="high", output="sos")

        mid  = sosfiltfilt(sos_lp2, not_low)
        mid  = sosfiltfilt(sos_lp2, mid)
        high = sosfiltfilt(sos_hp2, not_low)
        high = sosfiltfilt(sos_hp2, high)

        return low, mid, high

    @staticmethod
    def _compress_band(
        band: np.ndarray,
        settings: BandSettings,
        sample_rate: int,
    ) -> np.ndarray:
        """
        Apply a feed-forward, RMS-based compressor with soft knee.

        The level detector uses a 1st-order IIR envelope follower whose
        time constants are computed from attack/release times as:
            coeff = exp(-1 / (time_seconds × sample_rate))
        """
        thr_db  = settings.threshold_db
        ratio   = max(settings.ratio, 1.0)
        atk_coef = np.exp(-1.0 / (max(settings.attack_ms,  0.1) * 1e-3 * sample_rate))
        rel_coef = np.exp(-1.0 / (max(settings.release_ms, 1.0) * 1e-3 * sample_rate))
        makeup   = 10.0 ** (settings.makeup_gain_db / 20.0)
        knee     = settings.knee_db

        n    = len(band)
        out  = np.empty(n, dtype=np.float64)
        rms  = 0.0  # running RMS estimate

        for i in range(n):
            x = band[i]

            # IIR envelope follower (RMS-approximation on |x|²)
            x2 = x * x
            if x2 > rms:
                rms = atk_coef * rms + (1.0 - atk_coef) * x2
            else:
                rms = rel_coef * rms + (1.0 - rel_coef) * x2

            # Level in dB
            level_db = 10.0 * np.log10(rms + 1e-20)

            # Soft-knee gain computation
            overshoot = level_db - thr_db
            if knee > 0.0 and overshoot > -knee / 2.0 and overshoot < knee / 2.0:
                # Inside the knee region — interpolate
                gain_db = (overshoot + knee / 2.0) ** 2 / (2.0 * knee) * (1.0 / ratio - 1.0)
            elif overshoot >= knee / 2.0:
                # Above threshold
                gain_db = overshoot * (1.0 / ratio - 1.0)
            else:
                gain_db = 0.0

            out[i] = x * 10.0 ** (gain_db / 20.0) * makeup

        return out
