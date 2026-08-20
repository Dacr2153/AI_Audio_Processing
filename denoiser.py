"""
Denoiser — Phase 2 of the Audio Restoration Pipeline.

Implements four denoising methods:

  1. music (DEFAULT — recommended for music)
     Minimum-Statistics spectral subtraction with Wiener gain.
     ─ Estimates the noise floor per frequency bin using a sliding
       minimum over ~1.5 s: during musical passages, amplitude dips
       briefly between notes and that reveals the true noise floor.
     ─ Wiener-style gain preserves bins with high SNR (music/voices)
       and attenuates bins near the noise floor (hiss, static, hum).
     ─ Temporal smoothing prevents "musical noise" graininess.
     Best for: vinyl hiss, tape hiss, room noise, analog recordings.
     Recommended prop_decrease: 0.6–0.8 for vinyl noise.

  2. noisereduce (good for non-musical / adaptive)
     Spectral subtraction with frequency and time-domain smoothing.
     Uses quietest segment as noise reference — avoids treating
     musical content as noise (unlike fixed first-0.5s approach).
     Best for: speech recordings, stationary hiss/hum.

  3. deepfilternet (best SNR for speech — NOT recommended for music)
     Neural RNN/attention model trained on DNS5 SPEECH data.
     Excellent for voice; treats instruments as noise → damages music.
     Requires: deepfilternet >= 0.5.6 + torch

  4. wavelet (fast fallback)
     Level-dependent BayesShrink wavelet soft-thresholding (db4).
     Gentle on coarser levels (music) and more aggressive on finer
     detail levels (noise).

Auto fallback order: music → noisereduce → wavelet
DeepFilterNet is NOT in auto — must be explicitly requested.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal, Optional

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy imports — each wrapped in a try/except for graceful fallback
# ---------------------------------------------------------------------------
try:
    import torch
    import torchaudio
    from df import enhance, init_df
    _DEEPFILTER_AVAILABLE = True
except ImportError:
    _DEEPFILTER_AVAILABLE = False
    logger.debug("DeepFilterNet not available.")

try:
    import noisereduce as nr
    _NOISEREDUCE_AVAILABLE = True
except ImportError:
    _NOISEREDUCE_AVAILABLE = False
    logger.debug("noisereduce not available.")

try:
    import pywt
    _PYWT_AVAILABLE = True
except ImportError:
    _PYWT_AVAILABLE = False
    logger.warning("PyWavelets not available. Install with: pip install PyWavelets")

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

Method = Literal["auto", "music", "noisereduce", "deepfilternet", "wavelet"]

_DEEPFILTER_SR = 48000  # DeepFilterNet operates at exactly 48 kHz


class Denoiser:
    """
    Multi-method audio denoiser optimized for music preservation.

    Usage::

        # Recommended for songs (preserves instruments and vocals):
        denoiser = Denoiser(method="music", prop_decrease=0.6)
        clean = denoiser.denoise(audio, sample_rate=44100)

        # For speech recordings:
        denoiser = Denoiser(method="noisereduce", stationary=True)
        clean = denoiser.denoise(audio, sample_rate=44100)
    """

    def __init__(
        self,
        method: Method = "auto",
        prop_decrease: float = 0.5,
        stationary: bool = False,
        wavelet: str = "db4",
        wavelet_threshold_mode: str = "soft",
        n_std_thresh: float = 1.5,
        passes: int = 1,
    ):
        """
        Args:
            method: Denoising algorithm.
                ``"auto"``         — tries music → noisereduce → wavelet.
                ``"music"``        — HPSS + Wiener (best for songs).
                ``"noisereduce"``  — spectral subtraction (good for speech).
                ``"deepfilternet"``— neural speech model (NOT for music).
                ``"wavelet"``      — BayesShrink wavelet (fast fallback).
            prop_decrease: Noise reduction aggressiveness.
                0.0 = no reduction, 1.0 = full suppression.
                For music: 0.4–0.6 recommended. Default 0.5.
            stationary: For noisereduce only. Assume constant noise.
                No effect on 'music' method (uses min-statistics always).
            wavelet: Wavelet family for PyWavelets (default: 'db4').
            wavelet_threshold_mode: 'soft' or 'hard' thresholding.
            n_std_thresh: For noisereduce stationary mode. Number of standard
                deviations above the noise mean before a bin is considered
                signal. Lower = more aggressive noise detection.
                1.5 (default, conservative) | 1.0 (balanced) | 0.5 (aggressive).
            passes: Number of sequential denoising passes. Each pass removes
                residual noise left by the previous one.  1–3 recommended.
                More passes → cleaner result but more CPU and slight dulling
                of very quiet musical transients.
        """
        self.method = method
        self.prop_decrease = prop_decrease
        self.stationary = stationary
        self.wavelet = wavelet
        self.wavelet_threshold_mode = wavelet_threshold_mode
        self.n_std_thresh = float(n_std_thresh)
        self.passes = max(1, int(passes))

        self._df_model = None
        self._df_state = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Denoises audio using the configured method (or best available).
        If passes > 1, runs the denoiser sequentially the requested number
        of times, using the output of each pass as input for the next.

        Args:
            audio: Float32 NumPy array (N,), values in [-1, 1].
            sample_rate: Sample rate in Hz.

        Returns:
            Denoised float32 NumPy array, same length as input.
        """
        audio = audio.astype(np.float32)

        result = self._denoise_single_pass(audio, sample_rate)

        # Multi-pass: each subsequent pass removes residual noise left by the previous.
        for p in range(1, self.passes):
            logger.debug("Denoiser pass %d/%d…", p + 1, self.passes)
            result = self._denoise_single_pass(result, sample_rate)

        return result

    def _denoise_single_pass(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Executes one denoising pass with the configured method."""
        if self.method == "music":
            return self._denoise_music(audio, sample_rate)

        if self.method == "deepfilternet":
            if not _DEEPFILTER_AVAILABLE:
                raise RuntimeError(
                    "DeepFilterNet is not installed. "
                    "Install with: pip install deepfilternet"
                )
            return self._denoise_deepfilternet(audio, sample_rate)

        if self.method == "noisereduce":
            if not _NOISEREDUCE_AVAILABLE:
                raise RuntimeError(
                    "noisereduce is not installed. "
                    "Install with: pip install noisereduce"
                )
            return self._denoise_noisereduce(audio, sample_rate)

        if self.method == "wavelet":
            if not _PYWT_AVAILABLE:
                raise RuntimeError(
                    "PyWavelets is not installed. "
                    "Install with: pip install PyWavelets"
                )
            return self._denoise_wavelet(audio)

        # auto — music → noisereduce → wavelet (DeepFilterNet NOT in auto)
        if _LIBROSA_AVAILABLE:
            try:
                return self._denoise_music(audio, sample_rate)
            except Exception as exc:
                logger.warning("Music denoiser failed (%s); falling back.", exc)

        if _NOISEREDUCE_AVAILABLE:
            try:
                return self._denoise_noisereduce(audio, sample_rate)
            except Exception as exc:
                logger.warning("noisereduce failed (%s); falling back.", exc)

        if _PYWT_AVAILABLE:
            return self._denoise_wavelet(audio)

        raise RuntimeError(
            "No denoiser available. Install at least one of: "
            "librosa, noisereduce, PyWavelets."
        )

    def available_method(self) -> str:
        """Returns the name of the best denoiser available in this environment."""
        if _LIBROSA_AVAILABLE:
            return "music"
        if _NOISEREDUCE_AVAILABLE:
            return "noisereduce"
        if _PYWT_AVAILABLE:
            return "wavelet"
        return "none"

    # ------------------------------------------------------------------
    # Method 1: Music-preserving HPSS + Wiener filter  (RECOMMENDED)
    # ------------------------------------------------------------------

    def _denoise_music(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Music-preserving noise reduction via inter-harmonic noise estimation.

        Algorithm
        ─────────
        1.  Compute STFT (float32 throughout for stability).
        2.  Noise profile via spectral inter-harmonic tracking:
            • For EVERY time frame, apply a local minimum filter of
              size=23 bins (≈ 500 Hz) along the frequency axis.
              → At harmonic peaks: the filter reaches into the neighboring
                inter-harmonic valleys and pulls the estimate DOWN toward
                the background noise level.
              → Between harmonics (pure noise bins): the filter returns
                approximately the bin's own noise magnitude.
            • Average the per-frame noise estimates over ALL frames.
              Time-averaging stabilizes the estimate even for dense,
              constantly-playing recordings with no quiet sections.
            • A 1.8× bias correction compensates for the systematic
              underestimation introduced by taking the local minimum
              (the minimum of a Rayleigh-distributed variable is
              significantly below its mean).
        3.  Wiener gain per time-frequency bin:
                ratio(f,t) = noise_profile(f) / |X(f,t)|
                G(f,t)     = clip(1 − α·ratio,  floor,  1.0)
            • ratio ≪ 1 (loud music bin):  G → 1.0    — fully preserved
            • ratio ≈ 1 (noise-only bin):  G → floor  — max reduction
            • floor = max(1−α, 0.05) allows up to 95 % noise suppression.
        4.  Temporal smoothing (≈ 58 ms) prevents spectral graininess.
        5.  Apply gain directly to complex STFT — phase is never modified.
        6.  Reconstruct with ISTFT.

        Works for: vinyl hiss, tape hiss, room noise — even in dense,
        constantly-playing recordings with no silence sections.
        Recommended prop_decrease: 0.7–0.85.
        """
        import librosa
        from scipy.ndimage import minimum_filter1d as _minf

        n_fft = 2048
        hop_length = 512

        # 1. STFT — float32 throughout
        audio_f32 = np.asarray(audio, dtype=np.float32)
        stft = librosa.stft(audio_f32, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)                              # float32 (F, T)

        # 2. Per-frame inter-harmonic noise estimation
        #    Local minimum along frequency axis (size=23 ≈ 500 Hz):
        #    removes harmonic peaks, leaves inter-harmonic noise floor.
        #    Average over time for a stable, shape-preserving noise profile.
        noise_floor_frames = _minf(
            magnitude.astype(np.float64), size=23, axis=0
        )                                                     # (F, T), float64
        noise_profile_1d = np.mean(noise_floor_frames, axis=1)  # (F,)
        # Bias correction — frequency-dependent:
        # The expected minimum of N=23 iid Rayleigh samples ≈ mean/√N, so
        # the base correction is √23 ≈ 4.80.  At high frequencies (> 500 Hz)
        # we ramp the correction upward because:
        #   (a) vinyl hiss grows relative to music signal above 2–4 kHz;
        #   (b) musical harmonics thin out above 4 kHz, so the local minimum
        #       is cleaner (less harmonic contamination) and we can safely
        #       push harder without damaging musical content.
        n_bins   = magnitude.shape[0]
        lo_bin   = int(500  * n_fft / sample_rate)   # ≈ bin 23  (500 Hz)
        hi_bin   = int(4000 * n_fft / sample_rate)   # ≈ bin 186 (4 kHz)
        correction = np.empty(n_bins, dtype=np.float64)
        correction[:lo_bin]        = 4.80
        correction[hi_bin:]        = 8.50
        correction[lo_bin:hi_bin]  = np.linspace(4.80, 8.50,
                                                   hi_bin - lo_bin,
                                                   dtype=np.float64)
        noise_profile_1d = noise_profile_1d * correction
        noise_profile = noise_profile_1d[:, np.newaxis].astype(np.float32)  # (F,1)

        # 3. Wiener-style gain — frequency-dependent floor:
        #    Below 4 kHz: floor = max(1−α, 0.05) — protects musical transients.
        #    Above 4 kHz: floor = 0.02 — allows up to 34 dB suppression of
        #    hiss-dominated bins where music is naturally quieter.
        alpha = np.float32(self.prop_decrease)
        base_floor = float(max(1.0 - self.prop_decrease, 0.05))
        floor_arr = np.full(n_bins, base_floor, dtype=np.float32)
        floor_arr[hi_bin:] = np.float32(0.02)
        floor_arr[lo_bin:hi_bin] = np.linspace(
            base_floor, 0.02, hi_bin - lo_bin, dtype=np.float32
        )
        ratio = noise_profile / (magnitude + np.float32(1e-8))   # (F, T)
        gain = np.float32(1.0) - alpha * ratio
        gain = np.clip(gain, floor_arr[:, np.newaxis], np.float32(1.0))

        # 4. Temporal smoothing to prevent "bubbly" spectral artifacts
        gain = uniform_filter1d(gain, size=5, axis=1)        # float32

        # 5. Apply gain to complex STFT (phase untouched)
        stft_out = gain * stft                                # complex64

        # 6. Reconstruct
        denoised = librosa.istft(stft_out, hop_length=hop_length, length=len(audio_f32))
        return np.asarray(denoised, dtype=np.float32)

    # ------------------------------------------------------------------
    # Method 2: noisereduce (spectral subtraction, improved)
    # ------------------------------------------------------------------

    def _denoise_noisereduce(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Spectral subtraction via noisereduce with music-friendly settings.

        Improvements over naive approach:
        ─ Uses the QUIETEST segment (not the first 0.5 s) as noise
          reference. A song starting immediately would have its first
          chord treated as "noise" and those frequencies removed forever.
        ─ freq_mask_smooth_hz=500: smooths spectral gain over frequency,
          eliminating "musical noise" (grainy tonal artifacts).
        ─ time_mask_smooth_ms=50: smooths gain over time, eliminating
          rapid fluctuations that produce a "warbling" effect.
        ─ n_std_thresh controls how aggressively bins are flagged as noise.
        """
        noise_clip = self._find_quietest_segment(audio, sample_rate)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            denoised = nr.reduce_noise(
                y=audio,
                sr=sample_rate,
                y_noise=noise_clip,
                prop_decrease=self.prop_decrease,
                stationary=self.stationary,
                n_std_thresh_stationary=self.n_std_thresh,
                n_fft=2048,
                win_length=2048,
                hop_length=512,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
                time_constant_s=0.5,
            )
        return denoised.astype(np.float32)

    def _find_quietest_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_duration: float = 0.3,
        n_segments: int = 8,
    ) -> np.ndarray:
        """Returns a noise clip from the quietest sections of the audio."""
        frame_len = max(int(segment_duration * sample_rate), 512)
        hop = frame_len // 2

        if len(audio) < frame_len * 4:
            return audio[:frame_len]

        n_frames = (len(audio) - frame_len) // hop + 1
        frames = np.array([audio[i * hop: i * hop + frame_len] for i in range(n_frames)])
        rms = np.sqrt(np.mean(frames ** 2, axis=1))

        n_use = min(n_segments, max(1, n_frames // 4))
        quietest_idx = np.argsort(rms)[:n_use]
        return np.concatenate([frames[i] for i in sorted(quietest_idx)]).astype(np.float32)

    # ------------------------------------------------------------------
    # Method 3: DeepFilterNet (speech — not recommended for music)
    # ------------------------------------------------------------------

    def _denoise_deepfilternet(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Neural denoising via DeepFilterNet.

        ⚠ Trained on speech (DNS5 dataset). Highly effective for voice
        recordings but suppresses instruments, harmony and percussion in
        music. Use method='music' for songs.
        """
        self._load_deepfilter_model()

        if sample_rate != _DEEPFILTER_SR:
            audio_in = self._resample_np(audio, sample_rate, _DEEPFILTER_SR)
        else:
            audio_in = audio

        tensor = torch.from_numpy(audio_in).unsqueeze(0)
        with torch.no_grad():
            enhanced = enhance(self._df_model, self._df_state, tensor)
        out = enhanced.squeeze(0).numpy()

        if sample_rate != _DEEPFILTER_SR:
            out = self._resample_np(out, _DEEPFILTER_SR, sample_rate)

        return self._match_length(out, len(audio)).astype(np.float32)

    def _load_deepfilter_model(self):
        if self._df_model is None:
            logger.info("Loading DeepFilterNet model (first run may download weights)…")
            self._df_model, self._df_state, _ = init_df()
            logger.info("DeepFilterNet model loaded.")

    # ------------------------------------------------------------------
    # Method 4: Wavelet BayesShrink (fast fallback)
    # ------------------------------------------------------------------

    def _denoise_wavelet(self, audio: np.ndarray) -> np.ndarray:
        """
        Level-dependent BayesShrink wavelet soft-thresholding.

        Improvements over VisuShrink:
        ─ BayesShrink computes per-level thresholds adapted to the signal
          variance at each scale → much gentler on coarser levels (music)
          and more aggressive at finer levels (high-freq noise).
        ─ Level scaling factor scales the threshold linearly from low
          (coarse = music content) to full (fine = noise).
        ─ prop_decrease further scales all thresholds proportionally.
        """
        import pywt

        max_level = pywt.dwt_max_level(len(audio), self.wavelet)
        levels = min(max_level, 6)
        coeffs = pywt.wavedec(audio, self.wavelet, level=levels)

        # Robust noise sigma estimate from finest level
        sigma_n = np.median(np.abs(coeffs[-1])) / 0.6745

        coeffs_thresh = [coeffs[0]]  # approximation untouched
        for level_idx, detail in enumerate(coeffs[1:], start=1):
            # BayesShrink threshold
            signal_var = max(0.0, float(np.mean(detail ** 2)) - sigma_n ** 2)
            if signal_var < 1e-12:
                thresh = float(np.max(np.abs(detail)))
            else:
                thresh = sigma_n ** 2 / np.sqrt(signal_var)

            # Scale: coarser levels get a gentler threshold
            level_factor = level_idx / levels   # 1/levels (coarse) → 1.0 (fine)
            thresh_scaled = thresh * level_factor * self.prop_decrease

            coeffs_thresh.append(
                pywt.threshold(detail, thresh_scaled, mode=self.wavelet_threshold_mode)
            )

        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        return self._match_length(denoised, len(audio)).astype(np.float32)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resample_np(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        from math import gcd
        g = gcd(orig_sr, target_sr)
        return signal.resample_poly(
            audio.astype(np.float64), target_sr // g, orig_sr // g
        ).astype(np.float32)

    @staticmethod
    def _match_length(audio: np.ndarray, target_len: int) -> np.ndarray:
        if len(audio) > target_len:
            return audio[:target_len]
        if len(audio) < target_len:
            return np.pad(audio, (0, target_len - len(audio)))
        return audio
        return audio
