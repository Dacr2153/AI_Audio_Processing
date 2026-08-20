"""Source separation — Phase 3 of the restoration pipeline.

Uses Demucs v4 (``htdemucs_ft``) to split audio into four stems:
``drums``, ``bass``, ``other``, ``vocals``.

Stems are returned in memory as stereo float32 arrays at 44100 Hz (Demucs
native output).  Down-stream mixing preserves the channel count.

References:
    - "Hybrid Transformers for Music Source Separation" (Rouard et al., 2022)
    - https://github.com/adefossez/demucs
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from ..constants import DEMUCS_MODELS, DEMUCS_STEMS
from ..exceptions import NeuralModelUnavailableError
from .devices import resolve_device

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


class SourceSeparator:
    """Separates an audio file into stems using Demucs v4.

    Usage::

        sep = SourceSeparator(model="htdemucs_ft")
        stems = sep.separate("song.wav")
        # stems = {"vocals": np.ndarray ((N, 2) stereo), "drums": ...}
    """

    def __init__(
        self,
        model: str = "htdemucs_ft",
        device: str = "auto",
        segment: int | None = None,
        shifts: int = 1,
    ):
        if model not in DEMUCS_MODELS:
            raise ValueError(
                f"Unsupported Demucs model {model!r}. Choices: {DEMUCS_MODELS}"
            )
        self.model = model
        self.shifts = shifts
        self.segment = segment
        self.device = resolve_device(device)

        if not _DEMUCS_AVAILABLE:
            logger.warning("SourceSeparator created but Demucs is not installed.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True when Demucs is installed and usable."""
        return _DEMUCS_AVAILABLE

    def separate_from_array(
        self, audio: np.ndarray, sample_rate: int
    ) -> dict[str, np.ndarray]:
        """Separate an in-memory array by writing it to a temp WAV first."""
        if not _DEMUCS_AVAILABLE:
            raise NeuralModelUnavailableError(
                "Demucs is not installed. Run: pip install demucs"
            )
        audio = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio, sample_rate, subtype="PCM_16")
            return self.separate(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def separate(
        self, input_path: str, output_dir: str | None = None
    ) -> dict[str, np.ndarray]:
        """Separate the file at ``input_path`` into stems.

        Returns a dict mapping stem name → stereo float32 array ``(N, 2)``.
        Sample rate is always 44100 Hz. When ``output_dir`` is None a temporary
        directory is used and cleaned up afterwards.
        """
        if not _DEMUCS_AVAILABLE:
            raise NeuralModelUnavailableError(
                "Demucs is not installed. Run: pip install demucs"
            )

        input_path = os.path.abspath(input_path)
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        use_temp = output_dir is None
        if use_temp:
            output_dir = tempfile.mkdtemp(prefix="demucs_")

        assert output_dir is not None  # narrowed by the branch above
        try:
            logger.info(
                "Running Demucs (%s) on: %s  [device=%s]",
                self.model,
                input_path,
                self.device,
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
        stems_to_exclude: list[str] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Return a mix of all stems except the excluded ones (default: vocals)."""
        if stems_to_exclude is None:
            stems_to_exclude = ["vocals"]

        stems = self.separate(input_path)
        included = {
            name: audio for name, audio in stems.items() if name not in stems_to_exclude
        }

        if not included:
            raise ValueError("No stems left after exclusions.")

        arrays = [np.asarray(audio, dtype=np.float32) for audio in included.values()]
        min_len = min(len(a) for a in arrays)
        mixed = np.sum([a[:min_len] for a in arrays], axis=0)
        mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)
        return mixed, 44_100  # Demucs always outputs 44100 Hz

    def save_stems(
        self,
        stems: dict[str, np.ndarray],
        output_dir: str,
        sample_rate: int = 44_100,
    ) -> None:
        """Persist separated stems as WAV files in ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)
        for name, audio in stems.items():
            out_path = os.path.join(output_dir, f"{name}.wav")
            sf.write(
                out_path,
                np.asarray(audio, dtype=np.float32),
                sample_rate,
                subtype="PCM_16",
            )
            logger.info("  Saved stem: %s", out_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_demucs(self, input_path: str, output_dir: str) -> None:
        args = [
            "--out",
            output_dir,
            "-n",
            self.model,
            "-d",
            self.device,
            "--shifts",
            str(self.shifts),
        ]
        if self.segment is not None:
            args += ["--segment", str(self.segment)]
        args.append(input_path)

        logger.debug("Demucs args: %s", args)
        demucs.separate.main(args)

    def _load_stems(self, input_path: str, output_dir: str) -> dict[str, np.ndarray]:
        """Read the stem WAVs Demucs wrote under ``output_dir/<model>/<track>/``."""
        track_name = Path(input_path).stem
        stems_dir = Path(output_dir) / self.model / track_name

        if not stems_dir.is_dir():
            raise RuntimeError(
                f"Demucs output directory not found: {stems_dir}\n"
                f"Expected structure: {output_dir}/{self.model}/{track_name}/"
            )

        stems: dict[str, np.ndarray] = {}
        for stem_name in DEMUCS_STEMS:
            stem_path = stems_dir / f"{stem_name}.wav"
            if not stem_path.is_file():
                logger.warning("Stem file not found: %s", stem_path)
                continue

            audio, _sr = sf.read(str(stem_path), dtype="float32")
            stems[stem_name] = audio

        return stems
