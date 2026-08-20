"""
Pre-processing module for the Audio Restoration Pipeline.

Handles:
  - Audio loading with automatic format detection (WAV, MP3, FLAC, OGG, M4A)
  - Stereo → mono conversion
  - Peak normalization
  - Silence trimming
  - Target sample-rate resampling
  - Low-pass pre-filter for MP3 inputs before super-resolution (avoids artifacts)
"""

import os
import numpy as np
import librosa
import soundfile as sf
from scipy import signal


class AudioPreprocessor:
    """
    Prepares raw audio files for the restoration pipeline.

    Usage:
        prep = AudioPreprocessor(target_sr=44100, trim_silence=True)
        audio, sr = prep.load_and_prepare("song.mp3")
    """

    def __init__(
        self,
        target_sr: int = None,
        trim_silence: bool = True,
        trim_top_db: float = 30.0,
        normalize: bool = True,
    ):
        """
        Args:
            target_sr: If set, resample audio to this sample rate after loading.
                       None preserves the original sample rate.
            trim_silence: Remove leading/trailing silence.
            trim_top_db: Silence threshold in dB for trimming.
            normalize: Peak-normalize the audio to [-1, 1].
        """
        self.target_sr = target_sr
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_prepare(self, filepath: str):
        """
        Main entry point. Loads and pre-processes an audio file.

        Args:
            filepath: Path to the audio file.

        Returns:
            Tuple (audio: np.ndarray float32, sample_rate: int)
        """
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        audio, sr = self._load(filepath)
        audio = self._to_mono(audio)

        if self.trim_silence:
            audio = self._trim_silence(audio, sr)

        if self.target_sr is not None and sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        if self.normalize:
            audio = self.peak_normalize(audio)

        return audio.astype(np.float32), sr

    def prepare_for_super_resolution(self, audio: np.ndarray, sr: int):
        """
        Applies a low-pass filter at the Nyquist cutoff of MP3 encoding (typically
        ~16 kHz for 128 kbps, ~20 kHz for 320 kbps) before feeding audio into
        AudioSR. Without this, residual MP3 cutoff patterns cause artifacts in
        the super-resolution output.

        Args:
            audio: Audio data as float32 array.
            sr: Current sample rate.

        Returns:
            Low-pass-filtered audio (float32).
        """
        cutoff_hz = min(16000.0, sr / 2.0 * 0.9)
        nyquist = sr / 2.0
        normalized_cutoff = cutoff_hz / nyquist
        b, a = signal.butter(8, normalized_cutoff, btype='low', analog=False)
        filtered = signal.filtfilt(b, a, audio.astype(np.float64))
        return filtered.astype(np.float32)

    @staticmethod
    def peak_normalize(audio: np.ndarray) -> np.ndarray:
        """
        Normalizes audio so the peak absolute value is 1.0.
        Returns audio unchanged if it is silent.
        """
        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            return audio / peak
        return audio

    @staticmethod
    def rms_level(audio: np.ndarray) -> float:
        """Returns the RMS level of the audio in dB."""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-12:
            return -120.0
        return 20.0 * np.log10(rms)

    @staticmethod
    def get_info(filepath: str) -> dict:
        """
        Returns metadata about an audio file without fully loading it.

        Returns:
            Dict with keys: sample_rate, channels, frames, duration_s, format.
        """
        info = sf.info(filepath)
        return {
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "frames": info.frames,
            "duration_s": info.duration,
            "format": info.format,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(filepath: str):
        """Loads audio using librosa (supports WAV, MP3, FLAC, OGG, M4A, etc.)."""
        audio, sr = librosa.load(filepath, sr=None, mono=False)
        return audio, sr

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """Converts multi-channel audio to mono by averaging channels."""
        if audio.ndim == 1:
            return audio
        return librosa.to_mono(audio)

    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trims leading and trailing silence below trim_top_db."""
        trimmed, _ = librosa.effects.trim(audio, top_db=self.trim_top_db)
        return trimmed
