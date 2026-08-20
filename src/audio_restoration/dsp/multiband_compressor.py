"""Multiband compressor — 3-band Linkwitz-Riley dynamic range compressor.

Splits the signal into low / mid / high bands, applies an independent
feed-forward, RMS-based compressor to each, then sums them back. Standard
mastering practice: a heavy bass hit must not trigger compression in the
presence/air band.

- Split uses an LR4 crossover (two cascaded 2nd-order Butterworth filters per
  boundary) applied zero-phase via ``sosfiltfilt``.
- Each band uses a 1st-order IIR envelope follower whose attack/release time
  constants are derived from milliseconds.
- Output is hard-clipped to [-1, 1].
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import cast

import numpy as np
from scipy.signal import butter, sosfiltfilt

from . import audio_utils

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-band compressor settings
# ---------------------------------------------------------------------------


@dataclass
class BandSettings:
    """Compressor settings for a single frequency band."""

    threshold_db: float = -18.0
    """Level above which compression kicks in (dBFS)."""

    ratio: float = 3.0
    """Compression ratio. 2:1 gentle … 8:1+ limiting."""

    attack_ms: float = 10.0
    """Attack time in milliseconds."""

    release_ms: float = 100.0
    """Release time in milliseconds."""

    makeup_gain_db: float = 0.0
    """Make-up gain applied after compression (dB)."""

    knee_db: float = 6.0
    """Soft-knee width in dB. 0 = hard knee."""


def default_band_settings() -> list[BandSettings]:
    """Return the mastering-style three-band defaults (low, mid, high)."""
    return [
        BandSettings(
            threshold_db=-20.0,
            ratio=2.5,
            attack_ms=20.0,
            release_ms=150.0,
            makeup_gain_db=1.5,
            knee_db=6.0,
        ),
        BandSettings(
            threshold_db=-18.0,
            ratio=3.0,
            attack_ms=10.0,
            release_ms=100.0,
            makeup_gain_db=1.0,
            knee_db=4.0,
        ),
        BandSettings(
            threshold_db=-16.0,
            ratio=2.0,
            attack_ms=5.0,
            release_ms=80.0,
            makeup_gain_db=0.5,
            knee_db=4.0,
        ),
    ]


class MultibandCompressor:
    """3-band Linkwitz-Riley crossover compressor (mono or stereo, channel-wise).

    Usage::

        mbc = MultibandCompressor(crossover_low=250.0, crossover_high=4000.0)
        processed = mbc.process(audio, sample_rate)
    """

    def __init__(
        self,
        crossover_low: float = 250.0,
        crossover_high: float = 4000.0,
        band_settings: list[BandSettings] | None = None,
    ):
        if crossover_low >= crossover_high:
            raise ValueError("crossover_low must be < crossover_high")
        self.crossover_low = float(crossover_low)
        self.crossover_high = float(crossover_high)
        self.bands = (
            band_settings if band_settings is not None else default_band_settings()
        )
        if len(self.bands) != 3:
            raise ValueError(
                "band_settings must have exactly 3 elements (low, mid, high)."
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply multiband compression (mono ``(N,)`` or stereo ``(N, 2)``)."""
        audio = np.asarray(audio, dtype=np.float64)
        out = audio_utils.process_channels(audio, self._process_mono, sample_rate)

        logger.info(
            "MultibandCompressor: applied (xover=%.0f/%.0f Hz).",
            self.crossover_low,
            self.crossover_high,
        )
        return out.astype(audio.dtype)

    def _process_mono(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        low, mid, high = self._split(audio, sample_rate)

        low_c = self._compress_band(low, self.bands[0], sample_rate)
        mid_c = self._compress_band(mid, self.bands[1], sample_rate)
        high_c = self._compress_band(high, self.bands[2], sample_rate)

        return np.clip(low_c + mid_c + high_c, -1.0, 1.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split(
        self,
        audio: np.ndarray,
        sample_rate: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split audio into three bands using cascaded LR4 crossovers."""
        nyq = sample_rate / 2.0

        f_low = min(self.crossover_low, nyq * 0.99)
        sos_lp1 = butter(2, f_low / nyq, btype="low", output="sos")
        sos_hp1 = butter(2, f_low / nyq, btype="high", output="sos")

        low = sosfiltfilt(sos_lp1, audio)
        low = sosfiltfilt(sos_lp1, low)
        not_low = sosfiltfilt(sos_hp1, audio)
        not_low = sosfiltfilt(sos_hp1, not_low)

        f_high = min(self.crossover_high, nyq * 0.99)
        sos_lp2 = butter(2, f_high / nyq, btype="low", output="sos")
        sos_hp2 = butter(2, f_high / nyq, btype="high", output="sos")

        mid = sosfiltfilt(sos_lp2, not_low)
        mid = sosfiltfilt(sos_lp2, mid)
        high = sosfiltfilt(sos_hp2, not_low)
        high = sosfiltfilt(sos_hp2, high)

        return low, mid, high

    @staticmethod
    def _compress_band(
        band: np.ndarray,
        settings: BandSettings,
        sample_rate: int,
    ) -> np.ndarray:
        """Feed-forward, RMS-based compressor with soft knee."""
        thr_db = settings.threshold_db
        ratio = max(settings.ratio, 1.0)
        atk_coef = np.exp(-1.0 / (max(settings.attack_ms, 0.1) * 1e-3 * sample_rate))
        rel_coef = np.exp(-1.0 / (max(settings.release_ms, 1.0) * 1e-3 * sample_rate))
        makeup = 10.0 ** (settings.makeup_gain_db / 20.0)
        knee = settings.knee_db

        n = len(band)
        out = np.empty(n, dtype=np.float64)
        rms = 0.0

        for i in range(n):
            x = cast(float, band[i])

            x2 = x * x
            if x2 > rms:
                rms = atk_coef * rms + (1.0 - atk_coef) * x2
            else:
                rms = rel_coef * rms + (1.0 - rel_coef) * x2

            level_db = 10.0 * np.log10(rms + 1e-20)
            overshoot = level_db - thr_db

            if knee > 0.0 and -knee / 2.0 < overshoot < knee / 2.0:
                gain_db = (
                    (overshoot + knee / 2.0) ** 2 / (2.0 * knee) * (1.0 / ratio - 1.0)
                )
            elif overshoot >= knee / 2.0:
                gain_db = overshoot * (1.0 / ratio - 1.0)
            else:
                gain_db = 0.0

            out[i] = x * 10.0 ** (gain_db / 20.0) * makeup

        return out
