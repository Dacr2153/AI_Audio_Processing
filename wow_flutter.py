"""
wow_flutter.py — Wow & Flutter correction for vinyl and cassette recordings.

Wow and flutter are pitch instabilities caused by mechanical imperfections in
playback mechanisms:

  Wow    (< 4 Hz)   — slow speed variation, e.g. an off-centre turntable spindle
                      or a stretched cassette belt.  Heard as slow pitch wavering.
  Flutter (4–100 Hz) — faster variations from motor cogging, roller irregularities.
                      Heard as a rapid warble or watery quality.

Algorithm overview
──────────────────
1.  **Pitch tracking**: Analyse the signal in short overlapping frames.
    In each frame, estimate the dominant fundamental frequency using
    autocorrelation (YIN-inspired algorithm).  Pitch is tracked in log-frequency
    space (cents) so that speed fluctuations appear as an additive signal.

2.  **Fluctuation extraction**: Smooth the per-frame pitch trajectory to obtain
    the long-term nominal pitch.  The difference (frame_pitch − nominal_pitch)
    gives the instantaneous pitch error in cents.

3.  **Time-stretch correction**: The pitch error is converted to a time-scale
    factor and applied sample-by-sample via linear resampling with a
    pre-computed time-map.  This "un-stretches" the speed fluctuations without
    changing the final duration.

4.  **Low-pass filtering of correction**: The pitch correction curve is
    band-pass filtered to the wow (< 4 Hz) and flutter (< 100 Hz) bands before
    application, so that genuine expressive pitch variation (vibrato, pitch
    bends) in the music is preserved.

Limitations
───────────
- Works best on signals with a stable dominant pitch (strings, sustained notes).
- Percussive signals (drums) have no detectable pitch — the correction passes
  through unchanged (detected pitch = None → correction = 0).
- Very severe wow (> 2 semitones variation) may introduce audible artefacts.
  Reduce max_deviation_cents in that case.
- Stereo input is processed channel-by-channel and can result in a slight
  inter-channel phase shift if the wow pattern differs between channels.
  For stereo recordings it is usually better to convert to mono for correction
  and then split back — this is handled automatically.

Usage::

    corrector = WowFlutterCorrector()
    corrected_audio = corrector.process(audio, sample_rate)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)


class WowFlutterCorrector:
    """
    Corrects wow and flutter pitch instabilities in vintage audio recordings.

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
        """
        Args:
            frame_ms:               Analysis frame length in ms. (default: 50 ms)
            hop_ms:                 Hop size between frames in ms. (default: 10 ms)
            max_deviation_cents:    Maximum pitch correction applied in cents.
                                    Larger deviations are clamped to avoid
                                    over-correction. (default: 100 cents = 1 semitone)
            correction_smoothing_ms: Smoothing window for the detected pitch
                                    trajectory in ms.  Set longer to only correct
                                    slow wow; shorter to also correct flutter.
                                    (default: 200 ms)
            max_freq_hz:            Upper bound of fluctuations to correct in Hz.
                                    Wow + flutter up to 100 Hz. (default: 100)
            min_freq_hz:            Lower bound in Hz.  Fluctuations slower than
                                    this are treated as deliberate (e.g. vibrato
                                    slower than 0.5 Hz). (default: 0.5)
        """
        self.frame_ms              = float(frame_ms)
        self.hop_ms                = float(hop_ms)
        self.max_deviation_cents   = float(max_deviation_cents)
        self.correction_smoothing_ms = float(correction_smoothing_ms)
        self.max_freq_hz           = float(max_freq_hz)
        self.min_freq_hz           = float(min_freq_hz)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Detect and correct wow/flutter in *audio*.

        Args:
            audio:        Mono or stereo float32/float64 audio array.
            sample_rate:  Sample rate in Hz.

        Returns:
            Corrected audio with same shape and dtype as input.
        """
        y = audio.astype(np.float64)

        if y.ndim == 2:
            # Stereo: correct using a mono mix for tracking, apply to both channels
            mono_mix = 0.5 * (y[:, 0] + y[:, 1])
            correction_curve = self._estimate_correction_curve(mono_mix, sample_rate)
            out_L = self._apply_correction(y[:, 0], correction_curve)
            out_R = self._apply_correction(y[:, 1], correction_curve)
            out = np.column_stack([out_L, out_R])
        else:
            correction_curve = self._estimate_correction_curve(y, sample_rate)
            out = self._apply_correction(y, correction_curve)

        logger.info("WowFlutterCorrector: correction applied (max_dev=%.0f cents).", self.max_deviation_cents)
        return out.astype(audio.dtype)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_correction_curve(self, mono: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Estimate a per-sample pitch correction curve (in cents) using
        autocorrelation-based pitch detection.

        Returns a 1-D array of length len(mono) containing the additive pitch
        correction in cents needed to stabilize the pitch.  Positive values
        mean the signal is running slow (we need to speed it up).
        """
        frame_n = int(self.frame_ms * sample_rate / 1000)
        hop_n   = int(self.hop_ms   * sample_rate / 1000)
        frame_n = max(frame_n, 64)
        hop_n   = max(hop_n, 16)

        n_frames = max(1, (len(mono) - frame_n) // hop_n + 1)
        frame_cents = np.zeros(n_frames)

        for i in range(n_frames):
            start = i * hop_n
            frame = mono[start: start + frame_n]
            if len(frame) < frame_n:
                frame = np.pad(frame, (0, frame_n - len(frame)))
            pitch_hz = self._autocorr_pitch(frame, sample_rate)
            # Convert to cents (log scale): 0 cents = not detected
            frame_cents[i] = 0.0 if pitch_hz is None else 1200.0 * np.log2(max(pitch_hz, 1e-6))

        # Build time axis (frame centres in samples)
        frame_times = np.arange(n_frames) * hop_n + frame_n // 2
        frame_times = np.clip(frame_times, 0, len(mono) - 1)

        # Smooth to get the "nominal" pitch (long-term average)
        smooth_n = max(1, int(self.correction_smoothing_ms / self.hop_ms))
        kernel   = np.ones(smooth_n) / smooth_n

        # Separate detected (non-zero) frames for robust smoothing
        detected = frame_cents != 0.0
        # Require at least 30% of frames to have a clear pitch.
        # Polyphonic music (drums, chords) typically has < 20% detectable frames
        # at a strict threshold.  Below 30%, the signal is not suitable for
        # wow/flutter correction — return silence (no correction).
        if detected.sum() < max(4, int(0.30 * n_frames)):
            logger.debug(
                "WowFlutterCorrector: only %d/%d frames with confident pitch "
                "— skipping correction (likely polyphonic content).",
                detected.sum(), n_frames,
            )
            return np.zeros(len(mono))

        # Interpolate non-zero pitch values to fill gaps
        x_det   = frame_times[detected]
        y_det   = frame_cents[detected]
        interp_fn = interp1d(x_det, y_det, kind="linear", fill_value=(y_det[0], y_det[-1]), bounds_error=False)
        cents_interp = interp_fn(frame_times)

        # Nominal pitch = smoothed version (moving average)
        nominal_cents = np.convolve(cents_interp, kernel, mode="same")

        # Deviation = instantaneous - nominal (in cents)
        deviation = cents_interp - nominal_cents

        # Band-pass the deviation to the wow/flutter range
        deviation = self._bandpass_curve(deviation, n_frames, self.hop_ms / 1000.0)

        # Clamp to max deviation
        deviation = np.clip(deviation, -self.max_deviation_cents, self.max_deviation_cents)

        # Rate-of-change limiter: prevent the correction from jumping more than
        # max_deviation_cents / 10 per frame (avoids abrupt warping artifacts).
        max_step = self.max_deviation_cents / 10.0
        for k in range(1, len(deviation)):
            delta = deviation[k] - deviation[k - 1]
            if abs(delta) > max_step:
                deviation[k] = deviation[k - 1] + np.sign(delta) * max_step

        # Upsample correction curve from frame-rate to sample-rate
        curve_interp = interp1d(
            frame_times,
            deviation,
            kind="linear",
            fill_value=(deviation[0], deviation[-1]),
            bounds_error=False,
        )
        sample_axis = np.arange(len(mono))
        correction_curve = curve_interp(sample_axis)

        return correction_curve

    def _bandpass_curve(self, curve: np.ndarray, n: int, dt: float) -> np.ndarray:
        """
        Band-pass the correction curve to [min_freq_hz, max_freq_hz].
        Operates at the frame rate (dt = hop_ms / 1000 seconds per frame).
        """
        fs = 1.0 / dt  # frame rate in Hz
        nyq = fs / 2.0

        lo = min(self.min_freq_hz / nyq, 0.99)
        hi = min(self.max_freq_hz / nyq, 0.99)

        if lo <= 0.0 or hi <= lo:
            return curve

        try:
            sos = sp_signal.butter(2, [lo, hi], btype="bandpass", output="sos")
            return sp_signal.sosfiltfilt(sos, curve)
        except Exception:
            return curve

    @staticmethod
    def _autocorr_pitch(frame: np.ndarray, sample_rate: int) -> Optional[float]:
        """
        Estimate the fundamental frequency of *frame* using normalised
        autocorrelation (YIN-inspired).  Returns None if no clear pitch is found.

        Detects pitches in the range 80 Hz – 2000 Hz, which covers most
        musical instruments relevant for wow/flutter tracking (piano, strings,
        voice, wind).
        """
        f_min, f_max = 80.0, 2000.0
        lag_min = int(sample_rate / f_max)
        lag_max = int(sample_rate / f_min)

        if lag_max >= len(frame):
            return None

        # Window the frame
        window = np.hanning(len(frame))
        x = frame * window

        # Compute autocorrelation via FFT for efficiency
        fft_size = 2 * len(x)  # zero-pad for linear (non-circular) correlation
        X = np.fft.rfft(x, n=fft_size)
        acf = np.fft.irfft(X * np.conj(X))
        acf = acf[:len(frame)]

        # Normalise
        if acf[0] < 1e-10:
            return None
        acf /= acf[0]

        # Find the first peak in [lag_min, lag_max]
        search = acf[lag_min: lag_max + 1]
        if len(search) == 0:
            return None

        # Find peaks above 0.6 correlation threshold (strict — avoids polyphonic frames).
        # 0.3 is too permissive: drums and chords create spurious peaks at 0.3–0.5.
        peaks, props = sp_signal.find_peaks(search, height=0.6, distance=lag_min)
        if len(peaks) == 0:
            return None

        best_idx = np.argmax(props["peak_heights"])
        best_height = props["peak_heights"][best_idx]

        # Polyphony rejection: if another peak is within 0.15 of the best,
        # the frame likely contains multiple simultaneous pitches — skip it.
        other_heights = np.delete(props["peak_heights"], best_idx)
        if len(other_heights) > 0 and np.max(other_heights) > best_height - 0.15:
            return None

        best = peaks[best_idx]
        lag = best + lag_min
        return float(sample_rate) / lag

    @staticmethod
    def _apply_correction(audio: np.ndarray, correction_cents: np.ndarray) -> np.ndarray:
        """
        Apply the per-sample pitch correction by resampling using a time-warp map.

        A pitch correction of +C cents means the signal was running C cents slow,
        i.e. it was stretched by factor 2^(C/1200).  To correct, we compress the
        time axis by the inverse factor.

        We build a source-to-output time mapping and interpolate to synthesise the
        corrected waveform.
        """
        n = len(audio)

        # Convert cents to speed ratio: > 1.0 = speed up (signal was slow)
        speed = 2.0 ** (correction_cents / 1200.0)

        # Build output time-map via cumulative integration of speed
        # out_sample[i] reads from source_pos[i]:  Δt_source = speed[i] × Δt_output
        source_pos = np.cumsum(speed)
        source_pos = source_pos - source_pos[0]       # start at 0
        # Rescale so the output maps approximately to the full input length
        if source_pos[-1] > 0:
            source_pos *= (n - 1) / source_pos[-1]

        # Clip to valid range
        source_pos = np.clip(source_pos, 0.0, n - 1.0)

        # Resample: for each output sample, read from the mapped source position
        out_idx  = np.arange(n)
        corrected = np.interp(out_idx, source_pos, audio)

        return corrected
