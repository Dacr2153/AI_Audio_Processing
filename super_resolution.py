"""
Super-Resolution — Phase 4 of the Audio Restoration Pipeline.

Upsamples low-quality or bandwidth-limited audio to 48 kHz using two methods:

  1. AudioSR (best quality) — diffusion-based audio super-resolution model.
     Recovers high-frequency content (8-24 kHz) that was lost due to:
     - Low original sample rate (e.g., 8 kHz phone, 22 kHz old recording)
     - MP3/AAC compression (which cuts off frequencies)
     - Tape/vinyl bandwidth limitations (~12-14 kHz cutoff on old recordings)

     AudioSR is called via the system CLI since it requires Python 3.10 or older
     for its full dependencies. If audiosr CLI is available, it is used.
     Reference: https://github.com/haoheliu/versatile_audio_super_resolution

  2. scipy polyphase resampling (always available fallback).
     This is a high-quality sinc interpolation — it does NOT synthesize new
     high-frequency content, but does convert the sample rate cleanly.
     It is the industry-standard method when SR models are unavailable.

Strategy:
  - If `audiosr` CLI is on PATH → use AudioSR (best quality).
  - Else → use scipy polyphase resampler.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

logger = logging.getLogger(__name__)

TARGET_SR = 48000  # AudioSR and DeepFilterNet both work at 48 kHz


class SuperResolution:
    """
    Upsamples audio to 48 kHz using AudioSR (when available) or scipy resampling.

    Usage::

        sr_module = SuperResolution()
        hi_res_audio, new_sr = sr_module.upsample(audio, sample_rate=22050)
    """

    def __init__(self, target_sr: int = TARGET_SR, device: str = "auto"):
        """
        Args:
            target_sr: Target sample rate. Default 48000 Hz.
            device: ``"auto"`` selects GPU if available, ``"cpu"`` forces CPU.
                    Only used by AudioSR.
        """
        self.target_sr = target_sr

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        self._audiosr_available = self._check_audiosr()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_audiosr_available(self) -> bool:
        """Returns True if the AudioSR CLI is on the system PATH."""
        return self._audiosr_available

    def upsample(self, audio: np.ndarray, sample_rate: int) -> tuple:
        """
        Upsamples audio to target_sr.

        Args:
            audio: Input audio (float32, mono).
            sample_rate: Current sample rate.

        Returns:
            Tuple (upsampled_audio: np.ndarray float32, target_sr: int).
        """
        audio = audio.astype(np.float32)

        if sample_rate == self.target_sr:
            logger.info("Audio is already at %d Hz — no resampling needed.", self.target_sr)
            return audio, self.target_sr

        if self._audiosr_available:
            try:
                result = self._upsample_audiosr(audio, sample_rate)
                logger.info("AudioSR super-resolution complete.")
                return result, self.target_sr
            except Exception as exc:
                logger.warning("AudioSR failed (%s); using scipy resampler.", exc)

        result = self._upsample_scipy(audio, sample_rate)
        logger.info("scipy polyphase resampling to %d Hz complete.", self.target_sr)
        return result, self.target_sr

    # ------------------------------------------------------------------
    # AudioSR (CLI-based)
    # ------------------------------------------------------------------

    def _upsample_audiosr(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Calls the `audiosr` CLI tool to perform diffusion-based super-resolution.

        Steps:
          1. Pre-filter: apply low-pass at the original Nyquist to prevent MP3
             cutoff artifacts from confusing the diffusion model.
          2. Write audio to a temp WAV file.
          3. Call: audiosr -i <input.wav> -s <output_dir> --device <device>
          4. Read the output WAV back into memory.
          5. Resample to exact target_sr if AudioSR output differs.
        """
        # Step 1: Pre-filter (removes MP3 cutoff pattern above original Nyquist)
        audio_prefiltered = self._low_pass_prefilter(audio, sample_rate)

        with tempfile.TemporaryDirectory(prefix="audiosr_") as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.wav")
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            sf.write(input_path, audio_prefiltered, sample_rate, subtype='PCM_16')

            cmd = [
                "audiosr",
                "-i", input_path,
                "-s", output_dir,
                "--device", self.device,
            ]
            logger.info("Running AudioSR: %s", " ".join(cmd))

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes max
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"audiosr exited with code {result.returncode}.\n"
                    f"stderr: {result.stderr[-1000:]}"
                )

            # AudioSR writes to <output_dir>/<input_stem>_audiosr.wav
            stem = Path(input_path).stem
            candidates = [
                os.path.join(output_dir, f"{stem}_audiosr.wav"),
                os.path.join(output_dir, f"{stem}.wav"),
            ]
            out_path = None
            for c in candidates:
                if os.path.isfile(c):
                    out_path = c
                    break
            if out_path is None:
                # Fallback: take any .wav in the output dir
                wav_files = list(Path(output_dir).glob("*.wav"))
                if not wav_files:
                    raise RuntimeError("AudioSR produced no output WAV file.")
                out_path = str(wav_files[0])

            hi_res, out_sr = sf.read(out_path, dtype='float32')
            if hi_res.ndim > 1:
                hi_res = hi_res.mean(axis=1)

            # Ensure we're at exactly target_sr
            if out_sr != self.target_sr:
                hi_res = self._resample(hi_res, out_sr, self.target_sr)

        return hi_res.astype(np.float32)

    # ------------------------------------------------------------------
    # scipy polyphase resampling (fallback)
    # ------------------------------------------------------------------

    def _upsample_scipy(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Resamples audio from sample_rate to target_sr using scipy polyphase filter.

        This is a high-quality, band-limited sinc interpolation — it does NOT
        synthesize new high-frequency content, but avoids aliasing artifacts.
        """
        return self._resample(audio, sample_rate, self.target_sr)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """High-quality polyphase resampling with anti-aliasing."""
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g
        resampled = signal.resample_poly(audio.astype(np.float64), up, down)
        return resampled.astype(np.float32)

    @staticmethod
    def _low_pass_prefilter(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """
        Applies an 8th-order Butterworth low-pass filter just below the Nyquist
        of the current sample rate. This removes the MP3 "hard cutoff" spectral
        pattern that would confuse the AudioSR diffusion model.
        """
        nyquist = sample_rate / 2.0
        # Cut at 90% of Nyquist to leave a clean rolloff
        cutoff = min(nyquist * 0.90, nyquist - 500.0)
        if cutoff <= 0:
            return audio
        normalized = cutoff / nyquist
        b, a = signal.butter(8, normalized, btype='low', analog=False)
        filtered = signal.filtfilt(b, a, audio.astype(np.float64))
        return filtered.astype(np.float32)

    @staticmethod
    def _check_audiosr() -> bool:
        """Returns True if the `audiosr` command is found on PATH."""
        found = shutil.which("audiosr") is not None
        if found:
            logger.info("AudioSR CLI found — will use for super-resolution.")
        else:
            logger.info(
                "AudioSR CLI not found. Using scipy resampler. "
                "To enable AudioSR: pip install audiosr (requires Python 3.10 or older)."
            )
        return found
