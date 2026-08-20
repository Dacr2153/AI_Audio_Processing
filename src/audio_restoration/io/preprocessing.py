"""Pre-processing — Phase 1 of the restoration pipeline.

Handles audio loading (with ffmpeg fallback), channel preservation/selection,
peak normalisation, optional silence trimming, target sample-rate resampling
and file metadata inspection.

Unlike earlier versions this module **preserves stereo** by default, so
down-stream stages (M/S processing, wow & flutter, multiband) operate on the
true channel count.
"""

from __future__ import annotations

import logging

import numpy as np
import soundfile as sf

from ..dsp import audio_utils
from .format_handler import FormatHandler

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Prepares raw audio files for the restoration pipeline.

    Usage::

        prep = AudioPreprocessor(target_sr=44100, trim_silence=True)
        audio, sr = prep.load_and_prepare("song.mp3")
    """

    def __init__(
        self,
        target_sr: int | None = None,
        trim_silence: bool = True,
        trim_top_db: float = 30.0,
        normalize: bool = True,
        mono: bool = False,
        ffmpeg_path: str = "ffmpeg",
    ):
        """Args are described inline; every option mirrors legacy behaviour
        except that ``mono`` now defaults to ``False`` (channels preserved).
        """
        self.target_sr = target_sr
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db
        self.normalize = normalize
        self.mono = mono
        self._handler = FormatHandler(ffmpeg_path=ffmpeg_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_and_prepare(self, filepath: str) -> tuple[np.ndarray, int]:
        """Load and pre-process an audio file.

        Returns ``(audio: float32, sample_rate: int)`` where ``audio`` is
        ``(N,)`` for mono and ``(N, 2)`` when stereo channels are preserved.
        """
        audio, sr = self._handler.read(filepath, sample_rate=None, mono=self.mono)

        if self.trim_silence and audio.ndim == 1:
            audio = self._trim_silence(audio, sr)

        if self.target_sr is not None and sr != self.target_sr:
            audio = audio_utils.resample(audio, sr, self.target_sr)
            sr = self.target_sr

        if self.normalize:
            audio = audio_utils.peak_normalize(audio)

        logger.debug("Prepared %r → %s @ %d Hz", filepath, audio.shape, sr)
        return np.asarray(audio, dtype=np.float32), sr

    @staticmethod
    def prepare_for_super_resolution(audio: np.ndarray, sr: int) -> np.ndarray:
        """Low-pass filter audio at ~90% of its Nyquist before super-resolution.

        Removes the hard MP3/AAC spectral cutoff pattern that can confuse
        diffusion-based super-resolution models.
        """
        from scipy import signal as sp_signal

        nyquist = sr / 2.0
        cutoff = min(nyquist * 0.90, nyquist - 500.0)
        if cutoff <= 0:
            return audio
        normalized = cutoff / nyquist
        b, a = sp_signal.butter(8, normalized, btype="low")
        filtered = sp_signal.filtfilt(b, a, np.asarray(audio, dtype=np.float64))
        return filtered.astype(np.float32)

    @staticmethod
    def get_info(filepath: str) -> dict:
        """Return metadata about an audio file without fully loading it."""
        info = sf.info(filepath)
        return {
            "sample_rate": int(info.samplerate),
            "channels": int(info.channels),
            "frames": int(info.frames),
            "duration_s": float(info.duration),
            "format": str(info.format),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        import librosa

        trimmed, _index = librosa.effects.trim(audio, top_db=self.trim_top_db)
        return trimmed
