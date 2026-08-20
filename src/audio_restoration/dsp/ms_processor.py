"""Mid/Side (M/S) stereo processing.

M/S encoding separates a stereo signal into:

* **Mid**  ``M = (L + R) / √2`` — the mono-compatible centre image
* **Side** ``S = (L − R) / √2`` — the stereo width / difference signal

Why this helps restoration:

1. Surface noise (hiss, hum, crackle) is often incoherent between channels and
   therefore concentrated in the Side channel — heavier noise reduction there
   preserves the natural Mid character.
2. The Mid channel holds the most important musical content (lead vocals, kick,
   bass) and benefits from a gentle presence EQ boost.
3. Stereo width is controlled by scaling Side amplitude.

Mono input is a no-op (returns audio unchanged).
"""

from __future__ import annotations

import logging

import numpy as np

from ..config import MSConfig
from . import biquad

logger = logging.getLogger(__name__)

_SQRT2 = np.sqrt(2.0)


class MSProcessor:
    """Mid/Side stereo processor.

    Usage::

        proc = MSProcessor(MSConfig(side_denoise=True))
        stereo_out = proc.process(stereo_audio, sample_rate)
    """

    def __init__(self, config: MSConfig | None = None):
        self.cfg = config or MSConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply M/S processing to *audio* (``(N,)`` mono or ``(N, 2)`` stereo)."""
        y = np.asarray(audio, dtype=np.float64)

        if y.ndim == 1 or (y.ndim == 2 and y.shape[1] == 1):
            logger.debug("MSProcessor: mono input — skipping M/S processing.")
            return audio
        if y.ndim != 2 or y.shape[1] != 2:
            logger.warning(
                "MSProcessor: expected stereo (N, 2), got shape %s — skipping.", y.shape
            )
            return audio

        cfg = self.cfg
        left, right = y[:, 0], y[:, 1]

        mid = (left + right) / _SQRT2
        side = (left - right) / _SQRT2

        if cfg.mid_presence_db != 0.0:
            mid = self._apply_presence(
                mid, sample_rate, cfg.mid_presence_db, cfg.mid_presence_freq
            )

        if cfg.side_denoise:
            side = self._denoise_side(side, sample_rate, cfg.side_prop_decrease)

        side = side * float(np.clip(cfg.side_gain, 0.0, 2.0))

        # NOTE: compute M−S algebraically from M+S to stay correct regardless of
        # in-place buffer reuse by numpy (refcount-1 optimisation).
        mid_plus_side = mid + side
        mid_minus_side = mid_plus_side - 2.0 * side
        out = np.column_stack([mid_plus_side / _SQRT2, mid_minus_side / _SQRT2])
        out = np.clip(out, -1.0, 1.0)

        logger.info(
            "MSProcessor: side_denoise=%s  side_gain=%.2f  mid_presence=%+.1f dB.",
            cfg.side_denoise,
            float(np.clip(cfg.side_gain, 0.0, 2.0)),
            cfg.mid_presence_db,
        )
        return out.astype(audio.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _denoise_side(
        side: np.ndarray, sample_rate: int, prop_decrease: float
    ) -> np.ndarray:
        """Denoise the Side channel using noisereduce (falls back unchanged)."""
        try:
            import noisereduce as nr

            denoised = nr.reduce_noise(
                y=side.astype(np.float32),
                sr=sample_rate,
                prop_decrease=float(np.clip(prop_decrease, 0.0, 1.0)),
                stationary=False,
            )
            return denoised.astype(np.float64)
        except ImportError:
            logger.warning("noisereduce not available — Side denoising skipped.")
            return side
        except Exception as exc:  # noqa: BLE001
            logger.warning("Side denoising failed (%s) — returning original Side.", exc)
            return side

    @staticmethod
    def _apply_presence(
        mid: np.ndarray,
        sample_rate: int,
        gain_db: float,
        freq_hz: float,
    ) -> np.ndarray:
        """Peaking EQ on the Mid channel (zero-phase, Q 0.7)."""
        from scipy.signal import filtfilt

        b, a = biquad.peaking(freq_hz, gain_db / 2.0, sample_rate, q=0.7)
        return filtfilt(b, a, mid)
