"""
ms_processor.py — Mid/Side (M/S) stereo processing.

M/S encoding separates a stereo signal into:
  - Mid  (M) = (L + R) / √2  — the mono-compatible centre image
  - Side (S) = (L − R) / √2  — the stereo width / difference signal

This is useful for audio restoration because:
  1. Surface noise (hiss, hum, crackle) is often incoherent between
     channels and therefore concentrated in the Side channel.  Applying
     heavier noise reduction to Side preserves the natural Mid character
     while better suppressing background noise.

  2. The Mid signal typically contains the most important musical
     information (lead vocals, kick drum, bass).  Applying a gentle
     presence or EQ boost to Mid lifts clarity without widening the image.

  3. You can control stereo width by scaling the Side amplitude:
       Side × width_factor  (< 1.0 = narrower, > 1.0 = wider)

Algorithm
─────────
  Encode:  M = (L + R) / √2,   S = (L − R) / √2
  Process: M′ = process_mid(M),  S′ = process_side(S)
  Decode:  L′ = (M′ + S′) / √2, R′ = (M′ − S′) / √2

The factor √2 preserves total power: ‖M‖² + ‖S‖² = ‖L‖² + ‖R‖².

If the input is mono, M/S processing is a no-op (returns input unchanged).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_SQRT2 = np.sqrt(2.0)


@dataclass
class MSConfig:
    """Settings for Mid/Side processing."""

    # --- Side noise reduction ---
    side_denoise: bool = True
    """Apply noise reduction to the Side channel.
    The denoiser operates on Side only, leaving Mid untouched."""

    side_prop_decrease: float = 0.6
    """Noise reduction aggressiveness on the Side channel (0–1).
    Default 0.6 — lighter than full-mix denoising to avoid artefacts."""

    # --- Width control ---
    side_gain: float = 1.0
    """Scale applied to the Side channel after processing.
    1.0 = unchanged,  < 1.0 = narrower stereo,  > 1.0 = wider stereo.
    Clipped to [0.0, 2.0] to avoid extreme values."""

    # --- Mid clarity boost ---
    mid_presence_db: float = 0.0
    """Peaking EQ boost/cut on the Mid channel at the presence frequency.
    Positive = vocal/attack clarity,  negative = de-harshen.
    0.0 = disabled (default)."""

    mid_presence_freq: float = 3500.0
    """Centre frequency for the Mid presence boost in Hz. (default: 3500 Hz)"""


class MSProcessor:
    """
    Mid/Side stereo processor.

    Usage::

        proc = MSProcessor(config)
        stereo_out = proc.process(stereo_audio, sample_rate)

    For mono input, ``process()`` returns the audio unchanged.
    """

    def __init__(self, config: MSConfig | None = None):
        self.cfg = config or MSConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Apply M/S processing to *audio*.

        Args:
            audio:       Input audio.  Shape (N,) for mono or (N, 2) for stereo.
            sample_rate: Sample rate in Hz.

        Returns:
            Processed audio with the same shape and dtype as input.
        """
        if audio.ndim == 1 or (audio.ndim == 2 and audio.shape[1] == 1):
            logger.debug("MSProcessor: mono input — skipping M/S processing.")
            return audio

        if audio.ndim != 2 or audio.shape[1] != 2:
            logger.warning("MSProcessor: expected stereo (N,2) input; got shape %s — skipping.", audio.shape)
            return audio

        cfg = self.cfg
        y = audio.astype(np.float64)
        L, R = y[:, 0], y[:, 1]

        # ---- Encode ----
        M = (L + R) / _SQRT2
        S = (L - R) / _SQRT2

        # ---- Process Mid ----
        if cfg.mid_presence_db != 0.0:
            M = self._apply_presence(M, sample_rate, cfg.mid_presence_db, cfg.mid_presence_freq)

        # ---- Process Side ----
        if cfg.side_denoise:
            S = self._denoise_side(S, sample_rate, cfg.side_prop_decrease)

        # Scale Side for width control
        side_gain = float(np.clip(cfg.side_gain, 0.0, 2.0))
        S = S * side_gain

        # ---- Decode ----
        # NOTE: numpy may compute (M + S) in-place in M's buffer when M has
        # refcount=1.  To get M-S correctly even if M is overwritten, derive it
        # algebraically from the already-computed M+S value:
        #   M - S  =  (M + S) - 2·S
        M_plus_S  = M + S
        M_minus_S = M_plus_S - 2.0 * S   # = M − S regardless of in-place reuse
        L_out = M_plus_S  / _SQRT2
        R_out = M_minus_S / _SQRT2

        out = np.column_stack([L_out, R_out])
        out = np.clip(out, -1.0, 1.0)

        logger.info(
            "MSProcessor: side_denoise=%s  side_gain=%.2f  mid_presence=%+.1f dB.",
            cfg.side_denoise, side_gain, cfg.mid_presence_db,
        )
        return out.astype(audio.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _denoise_side(side: np.ndarray, sample_rate: int, prop_decrease: float) -> np.ndarray:
        """
        Denoise the Side channel using noisereduce (spectral subtraction).

        Falls back to the signal unchanged if noisereduce is unavailable.
        """
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
        except Exception as exc:
            logger.warning("Side denoising failed (%s) — returning original Side.", exc)
            return side

    @staticmethod
    def _apply_presence(
        mid: np.ndarray,
        sample_rate: int,
        gain_db: float,
        freq_hz: float,
    ) -> np.ndarray:
        """
        Apply a peaking EQ to the Mid channel using an Audio EQ Cookbook biquad.

        Zero-phase (filtfilt) so no phase artefacts are introduced.
        """
        from scipy.signal import filtfilt

        A     = 10.0 ** (gain_db / 2.0 / 40.0)   # half gain for filtfilt squaring
        w0    = 2.0 * np.pi * freq_hz / sample_rate
        alpha = np.sin(w0) / (2.0 * 0.7)          # Q=0.7 for a musical peak

        b0 = 1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 = 1.0 - alpha / A

        b = np.array([b0, b1, b2]) / a0
        a = np.array([a0, a1, a2]) / a0

        return filtfilt(b, a, mid)
