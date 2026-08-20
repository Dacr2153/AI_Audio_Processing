"""Super-resolution — Phase 4 of the restoration pipeline.

Upsamples bandwidth-limited audio towards ``target_sr`` (default 48 kHz):

1. **AudioSR** (best quality) — diffusion-based model invoked via its system
   CLI. Recovers 8–24 kHz content lost to MP3/AAC/tape/vinyl rolloff. The CLI
   is used because AudioSR requires Python <= 3.10 while the rest of the
   package supports 3.14.
2. **scipy polyphase resampling** (always available) — clean sinc
   interpolation that does not synthesize new content.

Stereo input is processed channel-by-channel through AudioSR and recombined;
the scipy fallback resamples the full frame in one pass.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..constants import TARGET_SR
from ..dsp import audio_utils
from ..io.preprocessing import AudioPreprocessor
from .devices import resolve_device

logger = logging.getLogger(__name__)


class SuperResolution:
    """Upsample audio to a target sample rate using AudioSR or scipy.

    Usage::

        sr_module = SuperResolution()
        hi_res_audio, new_sr = sr_module.upsample(audio, sample_rate=22050)
    """

    def __init__(self, target_sr: int = TARGET_SR, device: str = "auto"):
        self.target_sr = target_sr
        self.device = resolve_device(device)
        self._audiosr_available = self._check_audiosr()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_audiosr_available(self) -> bool:
        """True when the AudioSR CLI is on the system PATH."""
        return self._audiosr_available

    def upsample(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
        """Upsample *audio* (``(N,)`` or ``(N, 2)``) to ``target_sr``.

        Returns ``(upsampled_audio, target_sr)``.
        """
        audio = audio_utils.validate_audio(audio)

        if sample_rate == self.target_sr:
            logger.info(
                "Audio is already at %d Hz — no resampling needed.", self.target_sr
            )
            return audio, self.target_sr

        if self._audiosr_available:
            try:
                result = self._upsample_audiosr(audio, sample_rate)
                logger.info("AudioSR super-resolution complete.")
                return result, self.target_sr
            except Exception as exc:  # noqa: BLE001
                logger.warning("AudioSR failed (%s); using scipy resampler.", exc)

        result = audio_utils.resample(audio, sample_rate, self.target_sr)
        logger.info("scipy polyphase resampling to %d Hz complete.", self.target_sr)
        return result, self.target_sr

    # ------------------------------------------------------------------
    # AudioSR (CLI-based)
    # ------------------------------------------------------------------

    def _upsample_audiosr(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Diffusion-based super-resolution, channel by channel for stereo."""
        if audio.ndim == 2:
            channels = [
                self._audiosr_channel(audio[:, c], sample_rate)
                for c in range(audio.shape[1])
            ]
            return np.stack(channels, axis=1)
        return self._audiosr_channel(audio, sample_rate)

    def _audiosr_channel(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Run AudioSR on a single (mono) channel."""
        # Pre-filter at ~90% of Nyquist to hide the hard codec cutoff.
        pre = AudioPreprocessor.prepare_for_super_resolution(audio, sample_rate)

        with tempfile.TemporaryDirectory(prefix="audiosr_") as tmp_dir:
            input_path = os.path.join(tmp_dir, "input.wav")
            output_dir = os.path.join(tmp_dir, "output")
            os.makedirs(output_dir, exist_ok=True)

            sf.write(input_path, pre, sample_rate, subtype="PCM_16")

            cmd = [
                "audiosr",
                "-i",
                input_path,
                "-s",
                output_dir,
                "--device",
                self.device,
            ]
            logger.info("Running AudioSR: %s", " ".join(cmd))

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300, check=False
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"audiosr exited with code {result.returncode}.\n"
                    f"stderr: {result.stderr[-1000:]}"
                )

            stem = Path(input_path).stem
            candidates = [
                os.path.join(output_dir, f"{stem}_audiosr.wav"),
                os.path.join(output_dir, f"{stem}.wav"),
            ]
            out_path = next((c for c in candidates if os.path.isfile(c)), None)
            if out_path is None:
                wav_files = list(Path(output_dir).glob("*.wav"))
                if not wav_files:
                    raise RuntimeError("AudioSR produced no output WAV file.")
                out_path = str(wav_files[0])

            hi_res, out_sr = sf.read(out_path, dtype="float32")
            if hi_res.ndim > 1:
                hi_res = hi_res.mean(axis=1)
            if out_sr != self.target_sr:
                hi_res = audio_utils.resample(hi_res, out_sr, self.target_sr)

        return hi_res.astype(np.float32)

    @staticmethod
    def _check_audiosr() -> bool:
        """True when the ``audiosr`` command is found on PATH."""
        found = shutil.which("audiosr") is not None
        if found:
            logger.info("AudioSR CLI found — will use for super-resolution.")
        else:
            logger.info(
                "AudioSR CLI not found. Using scipy resampler. "
                "To enable AudioSR: pip install audiosr (requires Python 3.10 or older)."
            )
        return found
