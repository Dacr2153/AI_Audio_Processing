"""Denoiser — Phase 2 of the audio restoration pipeline.

Implements four denoising methods:

1. ``music`` (DEFAULT — recommended for music)
   Minimum-statistics spectral subtraction with a Wiener gain: the noise floor
   is estimated per frequency bin with a sliding minimum over ~1.5 s and the
   STFT gain is computed from the estimated SNR. Temporal smoothing prevents
   "musical noise" graininess.  Recommended ``prop_decrease``: 0.6–0.8.

2. ``noisereduce`` (speech / adaptive noise)
   Spectral subtraction using the quietest segment of the recording as the
   noise reference, with frequency- and time-domain smoothing.

3. ``deepfilternet`` (neural, speech only — NOT for music)
   RNN/attention model trained on the DNS5 speech dataset. Excellent for
   voice; treats instruments as noise, so avoid for music.  Requires the
   ``[neural]`` extra.

4. ``wavelet`` (fast fallback)
   Level-dependent BayesShrink soft thresholding (Daubechies-4).

Auto fallback order: ``music`` → ``noisereduce`` → ``wavelet``.
DeepFilterNet is intentionally excluded from the auto chain.

Stereo input is processed channel-by-channel; mono behaviour is identical to
the original single-channel implementation.
"""

from __future__ import annotations

import logging
import threading
import warnings
from typing import Literal

import numpy as np
from scipy.ndimage import uniform_filter1d

from ..constants import DEEPFILTER_SR, DEFAULT_HOP_LENGTH, DEFAULT_N_FFT
from ..exceptions import NeuralModelUnavailableError
from . import audio_utils

logger = logging.getLogger(__name__)

Method = Literal["auto", "music", "noisereduce", "deepfilternet", "wavelet"]

# ---------------------------------------------------------------------------
# Optional heavy imports — each wrapped for graceful fallback.
# ---------------------------------------------------------------------------
try:
    import torch
    import torchaudio  # noqa: F401 — deepfilternet imports it internally
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
    import importlib.util as _find_spec_util

    _PYWT_AVAILABLE = _find_spec_util.find_spec("pywt") is not None
    if not _PYWT_AVAILABLE:
        logger.warning("PyWavelets not available. Install with: pip install PyWavelets")
except Exception:  # noqa: BLE001
    _PYWT_AVAILABLE = False


def deepfilternet_available() -> bool:
    return _DEEPFILTER_AVAILABLE


class Denoiser:
    """Multi-method audio denoiser optimised for music preservation.

    Usage::

        denoiser = Denoiser(method="music", prop_decrease=0.6)
        clean = denoiser.denoise(audio, sample_rate=44100)
    """

    def __init__(
        self,
        method: str = "auto",
        prop_decrease: float = 0.5,
        stationary: bool = False,
        wavelet: str = "db4",
        wavelet_threshold_mode: str = "soft",
        n_std_thresh: float = 1.5,
        passes: int = 1,
    ):
        self.method = method
        self.prop_decrease = prop_decrease
        self.stationary = stationary
        self.wavelet = wavelet
        self.wavelet_threshold_mode = wavelet_threshold_mode
        self.n_std_thresh = float(n_std_thresh)
        self.passes = max(1, int(passes))

        self._df_model = None
        self._df_state = None
        self._df_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Denoise audio using the configured method, channel by channel.

        Args:
            audio: Float32 array with shape ``(N,)`` or ``(N, 2)``.
            sample_rate: Sample rate in Hz.

        Returns:
            Denoised array with the same shape and dtype as the input.
        """
        audio = np.asarray(audio, dtype=np.float32)
        return audio_utils.process_channels(audio, self._denoise_channel, sample_rate)

    def available_method(self) -> str:
        """Return the name of the best denoiser available in this environment."""
        # 'music' relies only on librosa + scipy.ndimage (hard dependencies).
        return "music"

    # ------------------------------------------------------------------
    # Internal: per-channel orchestration
    # ------------------------------------------------------------------

    def _denoise_channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        result = self._denoise_single_pass(audio, sample_rate)
        for _p in range(1, self.passes):
            logger.debug("Denoiser pass %d/%d…", _p + 1, self.passes)
            result = self._denoise_single_pass(result, sample_rate)
        return result

    def _denoise_single_pass(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Execute one denoising pass with the configured method."""
        if self.method == "music":
            return self._denoise_music(audio, sample_rate)

        if self.method == "deepfilternet":
            if not _DEEPFILTER_AVAILABLE:
                raise NeuralModelUnavailableError(
                    "DeepFilterNet is not installed. "
                    'Install with: pip install "audio-restoration[neural]" or: pip install deepfilternet'
                )
            return self._denoise_deepfilternet(audio, sample_rate)

        if self.method == "noisereduce":
            if not _NOISEREDUCE_AVAILABLE:
                raise NeuralModelUnavailableError(
                    "noisereduce is not installed. Install with: pip install noisereduce"
                )
            return self._denoise_noisereduce(audio, sample_rate)

        if self.method == "wavelet":
            if not _PYWT_AVAILABLE:
                raise NeuralModelUnavailableError(
                    "PyWavelets is not installed. Install with: pip install PyWavelets"
                )
            return self._denoise_wavelet(audio)

        # auto — music → noisereduce → wavelet (DeepFilterNet NOT in auto).
        try:
            return self._denoise_music(audio, sample_rate)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Music denoiser failed (%s); falling back.", exc)

        if _NOISEREDUCE_AVAILABLE:
            try:
                return self._denoise_noisereduce(audio, sample_rate)
            except Exception as exc:  # noqa: BLE001
                logger.warning("noisereduce failed (%s); falling back.", exc)

        if _PYWT_AVAILABLE:
            return self._denoise_wavelet(audio)

        raise NeuralModelUnavailableError(
            "No denoiser available. Install at least one of: librosa, noisereduce, PyWavelets."
        )

    # ------------------------------------------------------------------
    # Method 1: Music-preserving HPSS + Wiener filter  (RECOMMENDED)
    # ------------------------------------------------------------------

    def _denoise_music(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Music-preserving noise reduction via inter-harmonic noise estimation.

        Aliasing-safe STFT reconstruction: gain is applied to the complex
        STFT (phase untouched) and the signal is rebuilt with ISTFT.
        """
        import librosa
        from scipy.ndimage import minimum_filter1d as _minf

        n_fft = DEFAULT_N_FFT
        hop_length = DEFAULT_HOP_LENGTH

        audio_f32 = np.asarray(audio, dtype=np.float32)
        stft = librosa.stft(audio_f32, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)

        # Per-frame inter-harmonic noise estimate: local min along frequency.
        noise_floor_frames = _minf(magnitude.astype(np.float64), size=23, axis=0)
        noise_profile_1d = np.mean(noise_floor_frames, axis=1)

        # Frequency-dependent bias correction.
        n_bins = magnitude.shape[0]
        lo_bin = int(500 * n_fft / sample_rate)
        hi_bin = int(4000 * n_fft / sample_rate)
        correction = np.empty(n_bins, dtype=np.float64)
        correction[:lo_bin] = 4.80
        correction[hi_bin:] = 8.50
        correction[lo_bin:hi_bin] = np.linspace(
            4.80, 8.50, hi_bin - lo_bin, dtype=np.float64
        )
        noise_profile_1d = noise_profile_1d * correction
        noise_profile = noise_profile_1d[:, np.newaxis].astype(np.float32)

        # Wiener-style gain with a frequency-dependent floor.
        alpha = np.float32(self.prop_decrease)
        base_floor = float(max(1.0 - self.prop_decrease, 0.05))
        floor_arr = np.full(n_bins, base_floor, dtype=np.float32)
        floor_arr[hi_bin:] = np.float32(0.02)
        floor_arr[lo_bin:hi_bin] = np.linspace(
            base_floor, 0.02, hi_bin - lo_bin, dtype=np.float32
        )

        ratio = noise_profile / (magnitude + np.float32(1e-8))
        gain = np.float32(1.0) - alpha * ratio
        gain = np.clip(gain, floor_arr[:, np.newaxis], np.float32(1.0))
        gain = uniform_filter1d(gain, size=5, axis=1)

        stft_out = gain * stft
        denoised = librosa.istft(stft_out, hop_length=hop_length, length=len(audio_f32))
        return np.asarray(denoised, dtype=np.float32)

    # ------------------------------------------------------------------
    # Method 2: noisereduce (spectral subtraction)
    # ------------------------------------------------------------------

    def _denoise_noisereduce(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Spectral subtraction via noisereduce with music-friendly smoothing."""
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
                n_fft=DEFAULT_N_FFT,
                win_length=DEFAULT_N_FFT,
                hop_length=DEFAULT_HOP_LENGTH,
                freq_mask_smooth_hz=500,
                time_mask_smooth_ms=50,
                time_constant_s=0.5,
            )
        return np.asarray(denoised, dtype=np.float32)

    def _find_quietest_segment(
        self,
        audio: np.ndarray,
        sample_rate: int,
        segment_duration: float = 0.3,
        n_segments: int = 8,
    ) -> np.ndarray:
        """Return a noise clip from the quietest sections of the audio."""
        frame_len = max(int(segment_duration * sample_rate), 512)
        hop = frame_len // 2

        if len(audio) < frame_len * 4:
            return audio[:frame_len]

        n_frames = (len(audio) - frame_len) // hop + 1
        frames = np.array(
            [audio[i * hop : i * hop + frame_len] for i in range(n_frames)]
        )
        rms = np.sqrt(np.mean(frames**2, axis=1))

        n_use = min(n_segments, max(1, n_frames // 4))
        quietest_idx = np.argsort(rms)[:n_use]
        return np.concatenate([frames[i] for i in sorted(quietest_idx)]).astype(
            np.float32
        )

    # ------------------------------------------------------------------
    # Method 3: DeepFilterNet (speech — not recommended for music)
    # ------------------------------------------------------------------

    def _denoise_deepfilternet(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Neural denoising via DeepFilterNet (speech only)."""
        self._load_deepfilter_model()

        if sample_rate != DEEPFILTER_SR:
            audio_in = audio_utils.resample(audio, sample_rate, DEEPFILTER_SR)
        else:
            audio_in = audio

        tensor = torch.from_numpy(np.ascontiguousarray(audio_in)).unsqueeze(0)
        with torch.no_grad():
            enhanced = enhance(self._df_model, self._df_state, tensor)
        out = enhanced.squeeze(0).numpy()

        if sample_rate != DEEPFILTER_SR:
            out = audio_utils.resample(out, DEEPFILTER_SR, sample_rate)

        return audio_utils.match_length(out, len(audio)).astype(np.float32)

    def _load_deepfilter_model(self) -> None:
        if self._df_model is None:
            with self._df_lock:
                if self._df_model is None:
                    logger.info(
                        "Loading DeepFilterNet model (first run may download weights)…"
                    )
                    self._df_model, self._df_state, _ = init_df()
                    logger.info("DeepFilterNet model loaded.")

    # ------------------------------------------------------------------
    # Method 4: Wavelet BayesShrink (fast fallback)
    # ------------------------------------------------------------------

    def _denoise_wavelet(self, audio: np.ndarray) -> np.ndarray:
        """Level-dependent BayesShrink wavelet soft-thresholding."""
        import pywt

        max_level = pywt.dwt_max_level(len(audio), self.wavelet)
        levels = min(max_level, 6)
        coeffs = pywt.wavedec(audio, self.wavelet, level=levels)

        sigma_n = np.median(np.abs(coeffs[-1])) / 0.6745

        coeffs_thresh: list = [coeffs[0]]  # approximation untouched
        for level_idx, detail in enumerate(coeffs[1:], start=1):
            signal_var = max(0.0, float(np.mean(detail**2)) - sigma_n**2)
            if signal_var < 1e-12:
                thresh = float(np.max(np.abs(detail)))
            else:
                thresh = sigma_n**2 / np.sqrt(signal_var)

            level_factor = level_idx / levels
            thresh_scaled = thresh * level_factor * self.prop_decrease

            coeffs_thresh.append(
                pywt.threshold(detail, thresh_scaled, mode=self.wavelet_threshold_mode)
            )

        denoised = pywt.waverec(coeffs_thresh, self.wavelet)
        return audio_utils.match_length(denoised, len(audio)).astype(np.float32)
