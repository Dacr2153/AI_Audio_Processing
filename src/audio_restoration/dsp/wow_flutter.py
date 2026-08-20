"""Wow & flutter correction for vinyl and cassette recordings.

Wow (sub-4 Hz) and flutter (4–100 Hz) are pitch instabilities caused by
mechanical playback imperfections:

1. **Pitch tracking** — short overlapping frames; per-frame fundamental
   frequency by autocorrelation (YIN-inspired), tracked in cents.
2. **Fluctuation extraction** — deviation of the (interpolated) frame pitch
   from a smoothed nominal trajectory, band-passed to [0.5, 100] Hz.
3. **Time-stretch correction** — each frame's deviation maps to a per-sample
   speed ratio; the waveform is reconstructed via linear interpolation along a
   cumulative time map (duration preserved).
4. **Guards** — deviations are clamped to ``max_cents``, rate-of-change
   limited, and channels share the mono-mix correction curve.

Limitations:
- Works best on signals with a stable dominant pitch; percussive content is
  skipped automatically (few confident frames).
- Stereo is corrected from the mono mix so the correction is phase-coherent.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, find_peaks, sosfiltfilt

logger = logging.getLogger(__name__)


class WowFlutterCorrector:
    """Corrects wow & flutter pitch instabilities in vintage recordings.

    Usage::

        wfc = WowFlutterCorrector()
        clean = wfc.process(audio, sample_rate)
    """

    def __init__(
        self,
        frame_ms: float = 50.0,
        hop_ms: float = 10.0,
        max_deviation_cents: float = 100.0,
        correction_smoothing_ms: float = 200.0,
        max_freq_hz: float = 100.0,
        min_freq_hz: float = 0.5,
    ):
        self.frame_ms = float(frame_ms)
        self.hop_ms = float(hop_ms)
        self.max_deviation_cents = float(max_deviation_cents)
        self.correction_smoothing_ms = float(correction_smoothing_ms)
        self.max_freq_hz = float(max_freq_hz)
        self.min_freq_hz = float(min_freq_hz)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Detect and correct wow/flutter in *audio* (mono or stereo)."""
        y = np.asarray(audio, dtype=np.float64)

        if y.ndim == 2:
            mono_mix = 0.5 * (y[:, 0] + y[:, 1])
            curve = self._estimate_correction_curve(mono_mix, sample_rate)
            out = np.column_stack(
                [
                    self._apply_correction(y[:, 0], curve),
                    self._apply_correction(y[:, 1], curve),
                ]
            )
        else:
            curve = self._estimate_correction_curve(y, sample_rate)
            out = self._apply_correction(y, curve)

        logger.info(
            "WowFlutterCorrector: correction applied (max_dev=%.0f cents).",
            self.max_deviation_cents,
        )
        return out.astype(audio.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_correction_curve(
        self, mono: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """Estimate the per-sample pitch correction curve (in cents)."""
        frame_n = max(int(self.frame_ms * sample_rate / 1000), 64)
        hop_n = max(int(self.hop_ms * sample_rate / 1000), 16)

        n_frames = max(1, (len(mono) - frame_n) // hop_n + 1)
        frame_cents = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_n
            frame = mono[start : start + frame_n]
            if len(frame) < frame_n:
                frame = np.pad(frame, (0, frame_n - len(frame)))
            pitch_hz = self._autocorr_pitch(frame, sample_rate)
            frame_cents[i] = (
                0.0 if pitch_hz is None else 1200.0 * np.log2(max(pitch_hz, 1e-6))
            )

        frame_times = np.arange(n_frames) * hop_n + frame_n // 2
        frame_times = np.clip(frame_times, 0, len(mono) - 1)

        smooth_n = max(1, int(self.correction_smoothing_ms / self.hop_ms))
        kernel = np.ones(smooth_n) / smooth_n

        detected = frame_cents != 0.0
        if detected.sum() < max(4, int(0.30 * n_frames)):
            logger.debug(
                "WowFlutterCorrector: only %d/%d frames with confident pitch "
                "— skipping correction (likely polyphonic content).",
                detected.sum(),
                n_frames,
            )
            return np.zeros(len(mono))

        x_det = frame_times[detected]
        y_det = frame_cents[detected]
        interp_fn = interp1d(
            x_det,
            y_det,
            kind="linear",
            fill_value=(y_det[0], y_det[-1]),
            bounds_error=False,
        )
        cents_interp = interp_fn(frame_times)

        nominal_cents = np.convolve(cents_interp, kernel, mode="same")
        deviation = cents_interp - nominal_cents

        deviation = self._bandpass_curve(deviation, n_frames, self.hop_ms / 1000.0)
        deviation = np.clip(
            deviation, -self.max_deviation_cents, self.max_deviation_cents
        )

        # Rate-of-change limiter (avoids abrupt warping artefacts).
        max_step = self.max_deviation_cents / 10.0
        for k in range(1, len(deviation)):
            delta = deviation[k] - deviation[k - 1]
            if abs(delta) > max_step:
                deviation[k] = deviation[k - 1] + np.sign(delta) * max_step

        curve_interp = interp1d(
            frame_times,
            deviation,
            kind="linear",
            fill_value=(deviation[0], deviation[-1]),
            bounds_error=False,
        )
        return curve_interp(np.arange(len(mono)))

    def _bandpass_curve(self, curve: np.ndarray, n: int, dt: float) -> np.ndarray:
        """Band-pass the correction curve to [min_freq_hz, max_freq_hz]."""
        fs = 1.0 / dt
        nyq = fs / 2.0

        lo = min(self.min_freq_hz / nyq, 0.99)
        hi = min(self.max_freq_hz / nyq, 0.99)

        if lo <= 0.0 or hi <= lo:
            return curve

        try:
            sos = butter(2, [lo, hi], btype="bandpass", output="sos")
            return sosfiltfilt(sos, curve)
        except Exception:  # noqa: BLE001
            return curve

    @staticmethod
    def _autocorr_pitch(frame: np.ndarray, sample_rate: int) -> float | None:
        """Estimate the fundamental frequency (80–2000 Hz) or return None."""
        f_min, f_max = 80.0, 2000.0
        lag_min = int(sample_rate / f_max)
        lag_max = int(sample_rate / f_min)

        if lag_max >= len(frame):
            return None

        window = np.hanning(len(frame))
        x = frame * window

        fft_size = 2 * len(x)
        xf = np.fft.rfft(x, n=fft_size)
        acf = np.fft.irfft(xf * np.conj(xf))
        acf = acf[: len(frame)]

        if acf[0] < 1e-10:
            return None
        acf /= acf[0]

        search = acf[lag_min : lag_max + 1]
        if len(search) == 0:
            return None

        peaks, props = find_peaks(search, height=0.6, distance=lag_min)
        if len(peaks) == 0:
            return None

        best_idx = int(np.argmax(props["peak_heights"]))
        best_height = float(props["peak_heights"][best_idx])

        # Polyphony rejection: another peak within 0.15 of the best → skip.
        other_heights = np.delete(props["peak_heights"], best_idx)
        if len(other_heights) > 0 and np.max(other_heights) > best_height - 0.15:
            return None

        best = int(peaks[best_idx])
        lag = best + lag_min
        return float(sample_rate) / lag

    @staticmethod
    def _apply_correction(
        audio: np.ndarray, correction_cents: np.ndarray
    ) -> np.ndarray:
        """Apply the per-sample pitch correction by interpolated time-warping."""
        n = len(audio)

        speed = 2.0 ** (correction_cents / 1200.0)

        source_pos = np.cumsum(speed)
        source_pos = source_pos - source_pos[0]
        if source_pos[-1] > 0:
            source_pos *= (n - 1) / source_pos[-1]

        source_pos = np.clip(source_pos, 0.0, n - 1.0)
        out_idx = np.arange(n)
        return np.interp(out_idx, source_pos, audio)
