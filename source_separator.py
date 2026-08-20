"""
Source Separator — Phase 3 of the Audio Restoration Pipeline.

Uses Demucs v4 (htdemucs_ft) to separate audio into four stems:
  drums, bass, other (instruments), vocals.

Demucs htdemucs_ft is a fine-tuned hybrid transformer/recurrent model
trained on MusDB-HQ + additional data. It is the highest quality model
in the Demucs family.

References:
  - Paper: "Hybrid Transformers for Music Source Separation" (Rouard et al., 2022)
  - Repo:  https://github.com/adefossez/demucs
  - Model: htdemucs_ft  (fine-tuned, 4x slower, best quality)
           htdemucs     (default, faster)
           mdx_extra    (MDX-Net, alternative architecture)
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

try:
    import demucs.separate
    _DEMUCS_AVAILABLE = True
except ImportError:
    _DEMUCS_AVAILABLE = False
    logger.warning(
        "Demucs not available. Install with: pip install demucs\n"
        "Source separation will be skipped."
    )

DEMUCS_STEMS = ("drums", "bass", "other", "vocals")


class SourceSeparator:
    """
    Separates an audio file into stems using Demucs v4.

    Usage::

        sep = SourceSeparator(model="htdemucs_ft")
        stems = sep.separate("song.wav")
        # stems = {"vocals": np.ndarray, "drums": np.ndarray, ...}

    Or just get the instrumental (all stems except vocals)::

        instrumental = sep.get_instrumental("song.wav", sample_rate)
    """

    def __init__(
        self,
        model: str = "htdemucs_ft",
        device: str = "auto",
        segment: Optional[int] = None,
        shifts: int = 1,
    ):
        """
        Args:
            model: Demucs model name.
                   ``"htdemucs_ft"`` — best quality, slowest (recommended).
                   ``"htdemucs"``    — default, balanced quality/speed.
                   ``"mdx_extra"``   — alternative architecture.
            device: ``"auto"`` selects GPU if available, else CPU.
                    ``"cpu"`` forces CPU.
                    ``"cuda"`` forces CUDA GPU.
            segment: Override processing segment length in seconds. Use 8 if
                     you have less than 3 GB VRAM. None uses model defaults.
            shifts: Number of random shifts for test-time augmentation.
                    1 = fastest, 5 = better quality (slower). Default: 1.
        """
        self.model = model
        self.shifts = shifts
        self.segment = segment

        if device == "auto":
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = device

        if not _DEMUCS_AVAILABLE:
            logger.warning("SourceSeparator created but Demucs is not installed.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """Returns True if Demucs is installed and usable."""
        return _DEMUCS_AVAILABLE

    def separate(self, input_path: str, output_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Separates the audio file into stems.

        Args:
            input_path: Path to the input audio file (WAV recommended).
            output_dir: Where to write stem WAV files. If None, uses a temp dir
                        that is cleaned up after loading the stems into memory.

        Returns:
            Dict mapping stem name → audio array (float32, shape (N,) mono).
            Keys: "vocals", "drums", "bass", "other".
            Sample rate is always 44100 Hz (Demucs native output).
        """
        if not _DEMUCS_AVAILABLE:
            raise RuntimeError(
                "Demucs is not installed. Run: pip install demucs"
            )

        input_path = os.path.abspath(input_path)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        use_temp = output_dir is None
        if use_temp:
            output_dir = tempfile.mkdtemp(prefix="demucs_")

        try:
            logger.info(
                "Running Demucs (%s) on: %s  [device=%s]",
                self.model, input_path, self.device,
            )
            self._run_demucs(input_path, output_dir)
            stems = self._load_stems(input_path, output_dir)
            logger.info("Source separation complete. Stems: %s", list(stems.keys()))
            return stems
        finally:
            if use_temp and os.path.isdir(output_dir):
                shutil.rmtree(output_dir, ignore_errors=True)

    def get_instrumental(
        self,
        input_path: str,
        stems_to_exclude: Optional[List[str]] = None,
    ) -> tuple:
        """
        Returns a mono instrumental mix (all stems except excluded ones).

        By default excludes "vocals" to produce a karaoke/instrumental track.
        You can also exclude "drums" for a backing track, etc.

        Args:
            input_path: Path to the input audio file.
            stems_to_exclude: List of stem names to exclude from the mix.
                              Default: ["vocals"].

        Returns:
            Tuple (instrumental: np.ndarray float32, sample_rate: int).
        """
        if stems_to_exclude is None:
            stems_to_exclude = ["vocals"]

        stems = self.separate(input_path)
        included = {
            name: audio
            for name, audio in stems.items()
            if name not in stems_to_exclude
        }

        if not included:
            raise ValueError("No stems left after exclusions.")

        # Mix by summing (equal weights) and clipping
        mixed = np.sum(list(included.values()), axis=0)
        mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
        return mixed, 44100  # Demucs always outputs 44100 Hz

    def save_stems(self, stems: Dict[str, np.ndarray], output_dir: str, sample_rate: int = 44100):
        """
        Saves separated stems to WAV files in output_dir.

        Args:
            stems: Dict from separate().
            output_dir: Directory to write <stem>.wav files.
            sample_rate: Sample rate (Demucs outputs 44100 Hz by default).
        """
        os.makedirs(output_dir, exist_ok=True)
        for name, audio in stems.items():
            out_path = os.path.join(output_dir, f"{name}.wav")
            sf.write(out_path, audio.astype(np.float32), sample_rate, subtype='PCM_16')
            logger.info("  Saved stem: %s", out_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_demucs(self, input_path: str, output_dir: str):
        """Calls demucs.separate.main() with the configured arguments."""
        args = [
            "--out", output_dir,
            "-n", self.model,
            "-d", self.device,
            "--shifts", str(self.shifts),
        ]
        if self.segment is not None:
            args += ["--segment", str(self.segment)]

        # mp3 output can cause issues — always request WAV
        args += ["--mp3"]  # demucs default is float32 wav, this is just to be safe

        # Remove --mp3 if it was accidentally added; we want WAV
        args = [a for a in args if a != "--mp3"]

        args.append(input_path)
        logger.debug("Demucs args: %s", args)
        demucs.separate.main(args)

    def _load_stems(self, input_path: str, output_dir: str) -> Dict[str, np.ndarray]:
        """
        After demucs.separate.main() writes stems, this reads them back into memory.

        Demucs writes stems to:
          <output_dir>/<model_name>/<track_name>/<stem>.wav
        """
        track_name = Path(input_path).stem
        stems_dir = Path(output_dir) / self.model / track_name

        if not stems_dir.is_dir():
            raise RuntimeError(
                f"Demucs output directory not found: {stems_dir}\n"
                f"Expected structure: {output_dir}/{self.model}/{track_name}/"
            )

        stems: Dict[str, np.ndarray] = {}
        for stem_name in DEMUCS_STEMS:
            stem_path = stems_dir / f"{stem_name}.wav"
            if not stem_path.is_file():
                logger.warning("Stem file not found: %s", stem_path)
                continue

            audio, sr = sf.read(str(stem_path), dtype='float32')
            # Convert stereo to mono by averaging channels
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            stems[stem_name] = audio

        return stems
