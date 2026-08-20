"""Equalizer — Phase 5 of the audio restoration pipeline.

A professional 5-band mastering EQ built from Audio EQ Cookbook biquads
(see :mod:`audio_restoration.dsp.biquad`), all applied zero-phase via
``scipy.signal.filtfilt``:

1. Rumble HPF     @ 30 Hz  — sub-sonic rumble removal (optional)
2. Low shelf      @ 80 Hz  — bass warmth / body         (bass_gain_db)
3. Peaking bell   @ 250 Hz — mud control                (mid_gain_db, Q 0.8)
4. Peaking bell   @ 3500 Hz— presence / vocal clarity   (presence_gain_db, Q 1.4)
5. High shelf     @ 8 kHz  — treble brightness          (treble_gain_db)
6. High shelf     @ 12 kHz — air / sparkle              (air_gain_db)

.. note::
   ``filtfilt`` applies each filter forward and backward, SQUARING the
   magnitude response.  Every biquad is therefore designed at ``gain_db / 2``
   so the double pass yields the requested gain.

Mastering tools bundled here: peak normalisation, tanh soft limiter and RMS
normalisation.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy import signal as sp_signal

from . import audio_utils, biquad

logger = logging.getLogger(__name__)


class AudioEqualizer:
    """Zero-phase 5-band mastering EQ with rumble filter and mastering tools.

    Usage::

        eq = AudioEqualizer(bass_gain_db=2.5, presence_gain_db=2.5, air_gain_db=3.5)
        processed = eq.process(audio, sample_rate=44100)
    """

    def __init__(
        self,
        bass_gain_db: float = 0.0,
        mid_gain_db: float = 0.0,
        presence_gain_db: float = 0.0,
        treble_gain_db: float = 0.0,
        air_gain_db: float = 0.0,
        rumble_filter: bool = True,
        hp_frequency: float = 30.0,
    ):
        """Gains are in dB (±18 dB range for shelves/bells)."""
        self.bass_gain_db = float(bass_gain_db)
        self.mid_gain_db = float(mid_gain_db)
        self.presence_gain_db = float(presence_gain_db)
        self.treble_gain_db = float(treble_gain_db)
        self.air_gain_db = float(air_gain_db)
        self.rumble_filter = bool(rumble_filter)
        self.hp_frequency = float(hp_frequency)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        audio: np.ndarray,
        sample_rate: int,
        normalize: bool = True,
        limit: bool = True,
        limit_threshold_db: float = -0.3,
    ) -> np.ndarray:
        """Full mastering chain: EQ → peak normalize → soft limit.

        Accepts mono ``(N,)`` or stereo ``(N, 2)`` audio.
        """
        audio = audio_utils.process_channels(
            audio,
            self._process_channel,
            sample_rate,
            normalize=normalize,
            limit=limit,
            limit_threshold_db=limit_threshold_db,
        )
        return np.asarray(audio, dtype=np.float32)

    def equalize(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply the 5-band cascaded biquad EQ (mono or stereo)."""
        return audio_utils.process_channels(audio, self._equalize_channel, sample_rate)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_channel(
        self,
        audio: np.ndarray,
        sample_rate: int,
        normalize: bool,
        limit: bool,
        limit_threshold_db: float,
    ) -> np.ndarray:
        y = self._equalize_channel(audio, sample_rate)
        if normalize:
            y = self.peak_normalize(y)
        if limit:
            y = self.soft_limit(y, threshold_db=limit_threshold_db)
        return y.astype(np.float32)

    def _equalize_channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        nyq = sample_rate / 2.0
        original_dtype = audio.dtype
        y = audio.astype(np.float64)

        # Band 1 — rumble HPF (2nd-order Butterworth → effective 4th-order).
        if self.rumble_filter and self.hp_frequency > 0:
            fc = min(self.hp_frequency, nyq * 0.95)
            b_hp, a_hp = sp_signal.butter(2, fc / nyq, btype="high")
            y = sp_signal.filtfilt(b_hp, a_hp, y)

        # Band 2 — bass: low shelf @ 80 Hz.
        if self.bass_gain_db != 0.0:
            b, a = biquad.low_shelf(80.0, self.bass_gain_db / 2.0, sample_rate)
            y = sp_signal.filtfilt(b, a, y)

        # Band 3 — mid/mud: peaking bell @ 250 Hz, Q 0.8.
        if self.mid_gain_db != 0.0:
            b, a = biquad.peaking(250.0, self.mid_gain_db / 2.0, sample_rate, q=0.8)
            y = sp_signal.filtfilt(b, a, y)

        # Band 4 — presence: peaking bell @ 3500 Hz, Q 1.4.
        if self.presence_gain_db != 0.0:
            b, a = biquad.peaking(
                3500.0, self.presence_gain_db / 2.0, sample_rate, q=1.4
            )
            y = sp_signal.filtfilt(b, a, y)

        # Band 5 — treble: high shelf @ 8 kHz.
        if self.treble_gain_db != 0.0:
            fc5 = min(8000.0, nyq * 0.9)
            b, a = biquad.high_shelf(fc5, self.treble_gain_db / 2.0, sample_rate)
            y = sp_signal.filtfilt(b, a, y)

        # Band 6 — air: high shelf @ 12 kHz.
        if self.air_gain_db != 0.0:
            fc6 = min(12000.0, nyq * 0.9)
            b, a = biquad.high_shelf(fc6, self.air_gain_db / 2.0, sample_rate)
            y = sp_signal.filtfilt(b, a, y)

        return y.astype(original_dtype)

    # ------------------------------------------------------------------
    # Mastering tools
    # ------------------------------------------------------------------

    @staticmethod
    def peak_normalize(audio: np.ndarray, target_level: float = 1.0) -> np.ndarray:
        return audio_utils.peak_normalize(audio, target_level)

    @staticmethod
    def soft_limit(audio: np.ndarray, threshold_db: float = -0.3) -> np.ndarray:
        """Soft brickwall limiter using tanh saturation above the threshold."""
        threshold_linear = 10.0 ** (threshold_db / 20.0)
        audio_f = np.asarray(audio, dtype=np.float64)
        abs_audio = np.abs(audio_f)
        mask = abs_audio > threshold_linear

        if not np.any(mask):
            return audio.astype(np.float32)

        sign = np.sign(audio_f)
        excess = abs_audio - threshold_linear
        head_room = 1.0 - threshold_linear
        limited = np.where(
            mask,
            sign * (threshold_linear + head_room * np.tanh(excess / head_room)),
            audio_f,
        )
        return limited.astype(np.float32)

    @staticmethod
    def rms_normalize(audio: np.ndarray, target_rms_db: float = -18.0) -> np.ndarray:
        """Normalise RMS to a target dBFS (-18 broadcast, -14 streaming)."""
        rms = np.sqrt(np.mean(np.asarray(audio, dtype=np.float64) ** 2))
        if rms < 1e-12:
            return audio
        target_rms_linear = 10.0 ** (target_rms_db / 20.0)
        gain = target_rms_linear / rms
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)
