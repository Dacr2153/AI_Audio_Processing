"""
Equalizer — Phase 5 of the Audio Restoration Pipeline.

Implements a professional 5-band mastering EQ using Audio EQ Cookbook
biquad filters (Robert Bristow-Johnson), plus final-stage mastering tools.

EQ chain (applied in cascade, all zero-phase via filtfilt):
  1. Rumble HPF   @   30 Hz  — sub-sonic rumble removal (always on)
  2. Low shelf    @   80 Hz  — bass warmth / body         (bass_gain_db)
  3. Peaking bell @  250 Hz  — mud control                (mid_gain_db,  Q 0.8)
  4. Peaking bell @ 3500 Hz  — presence / vocal clarity   (presence_gain_db, Q 1.4)
  5. High shelf   @ 8000 Hz  — treble brightness          (treble_gain_db)
  6. High shelf   @12000 Hz  — air / sparkle / openness   (air_gain_db)

Why biquad cascade instead of the old 3-band splitter
──────────────────────────────────────────────────────
The previous approach split the signal into bands, scaled each, then
summed them — which works but loses inter-band phase coherence at the
crossover points.  Cascaded biquad filters applied to the SAME signal
behave exactly like a mixing-console parametric EQ: each filter modifies
only its target region and phases combine naturally across bands.

filtfilt() is used for every biquad to guarantee ZERO PHASE DISTORTION —
mandatory for music so that transient attack shapes are preserved.

Mastering tools:
  - Peak normalization:  ensures output peak = target level.
  - Soft limiter (tanh): gently squashes inter-sample peaks without clipping.
  - RMS normalization:   optional broadcast/streaming loudness target.

Vinyl-restoration preset (used by default in restoration_pipeline.py):
  bass      = +2.5 dB @ 80 Hz    — restore low-end warmth lost in vinyl transfer
  mid       = -1.5 dB @ 250 Hz   — clean up muddy/boxy resonance common in vinyl
  presence  = +2.5 dB @ 3500 Hz  — sharpen vocal articulation & instrument attack
  treble    = +2.0 dB @ 8000 Hz  — restore high-frequency clarity/definition
  air       = +3.5 dB @ 12000 Hz — add sparkle and open "air" typical of modern masters
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class AudioEqualizer:
    """
    Zero-phase 5-band mastering EQ with rumble filter and mastering tools.

    Band centres / shelving points:
      bass      — low shelf  @   80 Hz  (warmth)
      mid       — peak bell  @  250 Hz  (mud / clarity)
      presence  — peak bell  @ 3500 Hz  (vocals / attack)
      treble    — high shelf @ 8000 Hz  (definition)
      air       — high shelf @12000 Hz  (sparkle / openness)

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
        # Legacy crossover params — kept for backward compat, unused
        crossover_low: float = 250.0,
        crossover_high: float = 4000.0,
        filter_order: int = 4,
    ):
        """
        Args:
            bass_gain_db:     Low-shelf gain @ 80 Hz in dB. ±18 dB range.
            mid_gain_db:      Peaking-bell gain @ 250 Hz. Negative = mud cut.
            presence_gain_db: Peaking-bell gain @ 3500 Hz. Boosts vocal clarity.
            treble_gain_db:   High-shelf gain @ 8000 Hz. Adds brightness.
            air_gain_db:      High-shelf gain @ 12000 Hz. Adds sparkle/air.
            rumble_filter:    Apply 2nd-order HPF @ hp_frequency to remove
                              sub-sonic rumble (turntable wobble, wind, etc.).
            hp_frequency:     Cut-off for the rumble HPF (Hz). Default 30 Hz.
        """
        self.bass_gain_db     = float(bass_gain_db)
        self.mid_gain_db      = float(mid_gain_db)
        self.presence_gain_db = float(presence_gain_db)
        self.treble_gain_db   = float(treble_gain_db)
        self.air_gain_db      = float(air_gain_db)
        self.rumble_filter    = bool(rumble_filter)
        self.hp_frequency     = float(hp_frequency)
        # Legacy (ignored internally)
        self.crossover_low    = float(crossover_low)
        self.crossover_high   = float(crossover_high)
        self.filter_order     = int(filter_order)

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
        """Full mastering chain: EQ → normalize → soft limit."""
        audio = audio.astype(np.float64)
        audio = self.equalize(audio, sample_rate)
        if normalize:
            audio = self.peak_normalize(audio)
        if limit:
            audio = self.soft_limit(audio, threshold_db=limit_threshold_db)
        return audio.astype(np.float32)

    def equalize(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies the 5-band cascaded biquad EQ to audio.

        Each filter is applied sequentially to the same signal using
        filtfilt() (zero-phase, forward-backward pass).  Active only when
        the corresponding gain is non-zero (or rumble_filter is True).

        Args:
            audio: Input audio (float32 or float64, mono).
            sample_rate: Sample rate in Hz.

        Returns:
            Equalized audio (same dtype as input).
        """
        nyq = sample_rate / 2.0
        original_dtype = audio.dtype
        y = audio.astype(np.float64)

        # NOTE: filtfilt applies the filter in BOTH directions (forward + backward),
        # which SQUARES the magnitude response (doubles gain in dB).  We therefore
        # design every biquad at gain_db/2 so the double-pass yields the
        # user-specified gain.  The HPF is designed at full order (doubling makes
        # the roll-off steeper, which is desirable for rumble removal).

        # Band 1 — Rumble filter: 2nd-order Butterworth HPF (→ effective 4th-order)
        if self.rumble_filter and self.hp_frequency > 0:
            fc = min(self.hp_frequency, nyq * 0.95)
            b, a = signal.butter(2, fc / nyq, btype='high')
            y = signal.filtfilt(b, a, y)

        # Band 2 — Bass: low-shelving filter @ 80 Hz
        if self.bass_gain_db != 0.0:
            b, a = self._low_shelf(80.0, self.bass_gain_db / 2.0, sample_rate)
            y = signal.filtfilt(b, a, y)

        # Band 3 — Mid/mud: peaking bell @ 250 Hz, Q=0.8 (broad, gentle)
        if self.mid_gain_db != 0.0:
            b, a = self._peak(250.0, self.mid_gain_db / 2.0, sample_rate, Q=0.8)
            y = signal.filtfilt(b, a, y)

        # Band 4 — Presence: peaking bell @ 3500 Hz, Q=1.4 (focused)
        if self.presence_gain_db != 0.0:
            b, a = self._peak(3500.0, self.presence_gain_db / 2.0, sample_rate, Q=1.4)
            y = signal.filtfilt(b, a, y)

        # Band 5 — Treble: high-shelving filter @ 8000 Hz
        if self.treble_gain_db != 0.0:
            fc5 = min(8000.0, nyq * 0.9)
            b, a = self._high_shelf(fc5, self.treble_gain_db / 2.0, sample_rate)
            y = signal.filtfilt(b, a, y)

        # Band 6 — Air: high-shelving filter @ 12000 Hz
        if self.air_gain_db != 0.0:
            fc6 = min(12000.0, nyq * 0.9)
            b, a = self._high_shelf(fc6, self.air_gain_db / 2.0, sample_rate)
            y = signal.filtfilt(b, a, y)

        return y.astype(original_dtype)

    @staticmethod
    def peak_normalize(audio: np.ndarray, target_level: float = 1.0) -> np.ndarray:
        """Scales audio so the peak absolute value equals target_level."""
        peak = np.max(np.abs(audio))
        if peak < 1e-9:
            return audio
        return (audio / peak * target_level).astype(audio.dtype)

    @staticmethod
    def soft_limit(audio: np.ndarray, threshold_db: float = -0.3) -> np.ndarray:
        """
        Soft brickwall limiter using tanh saturation above the threshold.

        Samples below the threshold pass through unchanged.  Samples above
        the threshold are gently squashed via tanh so that the output never
        exceeds 1.0, avoiding hard digital clipping.
        """
        threshold_linear = 10.0 ** (threshold_db / 20.0)
        audio_f = audio.astype(np.float64)
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
    def rms_normalize(
        audio: np.ndarray,
        target_rms_db: float = -18.0,
    ) -> np.ndarray:
        """
        Normalizes the RMS level to a target dBFS.
        Common targets: -18 dBFS (broadcast), -14 dBFS (streaming).
        """
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if rms < 1e-12:
            return audio
        target_rms_linear = 10.0 ** (target_rms_db / 20.0)
        gain = target_rms_linear / rms
        return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    # ------------------------------------------------------------------
    # Audio EQ Cookbook biquad filter design
    # (Robert Bristow-Johnson — https://www.w3.org/TR/audio-eq-cookbook/)
    # All formulas return (b, a) ready for scipy.signal.filtfilt.
    # Note: A = 10^(gain_db/40) — NOT /20 — for shelving and peaking EQ.
    # ------------------------------------------------------------------

    @staticmethod
    def _low_shelf(fc: float, gain_db: float, fs: int, S: float = 1.0):
        """
        Low-shelving filter (Audio EQ Cookbook).

        Args:
            fc:      Shelf midpoint frequency (Hz).
            gain_db: Shelf gain in dB (positive = boost, negative = cut).
            fs:      Sample rate (Hz).
            S:       Shelf slope. S=1 = maximally-flat at shelf midpoint.
        """
        A  = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * fc / fs
        cosw0 = np.cos(w0)
        sinw0 = np.sin(w0)
        alpha = sinw0 / 2.0 * np.sqrt((A + 1.0/A) * (1.0/S - 1.0) + 2.0)

        b0 =     A * ((A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
        b1 = 2 * A * ((A-1) - (A+1)*cosw0)
        b2 =     A * ((A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
        a0 =         (A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha
        a1 =    -2 * ((A-1) + (A+1)*cosw0)
        a2 =         (A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha

        return (
            np.array([b0/a0, b1/a0, b2/a0]),
            np.array([1.0,   a1/a0, a2/a0]),
        )

    @staticmethod
    def _high_shelf(fc: float, gain_db: float, fs: int, S: float = 1.0):
        """
        High-shelving filter (Audio EQ Cookbook).

        Args:
            fc:      Shelf midpoint frequency (Hz).
            gain_db: Shelf gain in dB.
            fs:      Sample rate (Hz).
            S:       Shelf slope. S=1 = maximally-flat at shelf midpoint.
        """
        A  = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * fc / fs
        cosw0 = np.cos(w0)
        sinw0 = np.sin(w0)
        alpha = sinw0 / 2.0 * np.sqrt((A + 1.0/A) * (1.0/S - 1.0) + 2.0)

        b0 =      A * ((A+1) + (A-1)*cosw0 + 2*np.sqrt(A)*alpha)
        b1 = -2 * A * ((A-1) + (A+1)*cosw0)
        b2 =      A * ((A+1) + (A-1)*cosw0 - 2*np.sqrt(A)*alpha)
        a0 =          (A+1) - (A-1)*cosw0 + 2*np.sqrt(A)*alpha
        a1 =     2  * ((A-1) - (A+1)*cosw0)
        a2 =          (A+1) - (A-1)*cosw0 - 2*np.sqrt(A)*alpha

        return (
            np.array([b0/a0, b1/a0, b2/a0]),
            np.array([1.0,   a1/a0, a2/a0]),
        )

    @staticmethod
    def _peak(fc: float, gain_db: float, fs: int, Q: float = 1.0):
        """
        Peaking (bell) EQ filter (Audio EQ Cookbook).

        Args:
            fc:      Centre frequency (Hz).
            gain_db: Peak gain in dB.
            fs:      Sample rate (Hz).
            Q:       Quality factor. Higher Q = narrower bell.
        """
        A  = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * fc / fs
        alpha = np.sin(w0) / (2.0 * Q)

        b0 =  1.0 + alpha * A
        b1 = -2.0 * np.cos(w0)
        b2 =  1.0 - alpha * A
        a0 =  1.0 + alpha / A
        a1 = -2.0 * np.cos(w0)
        a2 =  1.0 - alpha / A

        return (
            np.array([b0/a0, b1/a0, b2/a0]),
            np.array([1.0,   a1/a0, a2/a0]),
        )

    @staticmethod
    def _db_to_linear(db: float) -> float:
        """Converts dB to linear gain factor."""
        return 10.0 ** (db / 20.0)

