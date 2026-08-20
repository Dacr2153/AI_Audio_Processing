"""Electrical hum removal (50/60 Hz power-line hum and harmonics).

Applies narrow IIR notch filters (`scipy.signal.iirnotch`) zero-phase via
`filtfilt`. A Q of ~35 makes each notch about 1.4 Hz wide at 50 Hz — narrow
enough to avoid musical coloration while fully removing the hum tone.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.signal import filtfilt, iirnotch

from . import audio_utils

logger = logging.getLogger(__name__)


class Dehummer:
    """Remove power-line hum (fundamental + harmonics) from an audio signal.

    Usage::

        dehum = Dehummer(freq=50.0, harmonics=5, q=35.0)
        clean = dehum.process(audio, sample_rate)
    """

    def __init__(self, freq: float = 50.0, harmonics: int = 5, q: float = 35.0):
        self.freq = float(freq)
        self.harmonics = int(harmonics)
        self.q = float(q)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Notch out the hum fundamental and harmonics (mono or stereo)."""
        audio = np.asarray(audio, dtype=np.float64)
        out = audio_utils.process_channels(audio, self._process_channel, sample_rate)

        logger.info(
            "Dehummer: removed %.0f Hz + %d harmonics (Q=%.0f).",
            self.freq,
            self.harmonics,
            self.q,
        )
        return out.astype(audio.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        y = np.asarray(audio, dtype=np.float64)
        nyq = sample_rate / 2.0

        for h in range(1, self.harmonics + 1):
            freq = self.freq * h
            if freq >= nyq * 0.99:
                break
            b, a = iirnotch(freq, Q=self.q, fs=sample_rate)
            y = filtfilt(b, a, y)

        return y
