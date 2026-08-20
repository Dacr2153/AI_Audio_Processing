"""Multi-format audio I/O for the Restoration Pipeline.

Supported INPUT formats (read via librosa + soundfile + ffmpeg fallback):

* WAV, MP3, FLAC, OGG/Vorbis, AIFF, M4A/AAC, OPUS, WMA, WV (WavPack),
  AU, CAF, W64, RF64, APE, and any other format ffmpeg can decode.

Supported OUTPUT formats:

* Lossless (via soundfile): ``.wav``, ``.flac``, ``.aiff``
* Lossy (via ffmpeg): ``.mp3``, ``.m4a``, ``.aac``, ``.ogg``, ``.opus``, ``.wma``

Usage::

    from audio_restoration.io.format_handler import FormatHandler

    handler = FormatHandler()
    handler.write(audio_float32, sample_rate, "output.mp3", bitrate="320k")
    handler.write(audio_float32, sample_rate, "output.flac")
    audio, sr = handler.read("song.mp3", mono=False)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf

from ..exceptions import (
    AudioLoadError,
    EncodingError,
    UnsupportedFormatError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

#: Maps file extension → (category, description, default_bitrate_or_None)
FORMAT_REGISTRY: dict[str, tuple[str, str, str | None]] = {
    # Lossless
    ".wav": ("lossless", "WAV PCM", None),
    ".flac": ("lossless", "FLAC (lossless)", None),
    ".aiff": ("lossless", "AIFF PCM", None),
    ".aif": ("lossless", "AIFF PCM", None),
    # Lossy — encoded via ffmpeg
    ".mp3": ("lossy", "MP3 (libmp3lame)", "V0"),  # VBR V0 ≈ 245 kbps
    ".m4a": ("lossy", "AAC/M4A", "256k"),
    ".aac": ("lossy", "AAC", "256k"),
    ".ogg": ("lossy", "OGG Vorbis", "8"),  # quality scale 0–10
    ".opus": ("lossy", "Opus", "192k"),
    ".wma": ("lossy", "WMA", "192k"),
}

#: Default bit depth for lossless writes.
DEFAULT_BIT_DEPTH = 24

_SUBTYPE_MAP: dict[int, str] = {16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}
_FORMAT_MAP: dict[str, str] = {
    ".wav": "WAV",
    ".flac": "FLAC",
    ".aiff": "AIFF",
    ".aif": "AIFF",
}


class FormatHandler:
    """High-quality multi-format audio I/O handler.

    Reads any format via librosa (falling back to ffmpeg/audioread).
    Writes lossless via soundfile and lossy via an ffmpeg subprocess.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self._ffmpeg_available: bool | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_format(self, filepath: str) -> str:
        """Return the lowercase file extension without the dot ("song.mp3" → "mp3")."""
        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            raise UnsupportedFormatError(
                f"Cannot determine format: {filepath!r} has no file extension. "
                "Use an extension like .wav, .flac, .mp3, .m4a, .ogg, or .opus."
            )
        return ext[1:]

    def is_lossless(self, filepath: str) -> bool:
        ext = "." + self.detect_format(filepath)
        entry = FORMAT_REGISTRY.get(ext)
        return entry is not None and entry[0] == "lossless"

    def is_lossy(self, filepath: str) -> bool:
        ext = "." + self.detect_format(filepath)
        entry = FORMAT_REGISTRY.get(ext)
        return entry is not None and entry[0] == "lossy"

    def supported_output_formats(self) -> list[str]:
        """Return the list of supported output extensions (with dot)."""
        return list(FORMAT_REGISTRY.keys())

    def read(
        self,
        filepath: str,
        sample_rate: int | None = None,
        mono: bool = True,
    ) -> tuple[np.ndarray, int]:
        """Read any audio file into a float32 numpy array.

        Args:
            filepath:    Path to the audio file.
            sample_rate: If given, resample to this rate; None keeps the native SR.
            mono:        Down-mix to mono when True (default).

        Returns:
            ``(audio: float32 ndarray, sample_rate: int)``. ``audio`` has shape
            ``(N,)`` when mono, ``(N, C)`` when ``mono=False``.
        """
        import librosa

        if not os.path.isfile(filepath):
            raise AudioLoadError(f"Audio file not found: {filepath}")

        try:
            audio, sr = librosa.load(filepath, sr=sample_rate, mono=mono)
            logger.debug(
                "Loaded %r via librosa/soundfile [%d Hz]",
                os.path.basename(filepath),
                sr,
            )
        except Exception as exc:  # noqa: BLE001 — any decoder failure warrants the fallback
            logger.warning(
                "librosa failed (%s) — trying ffmpeg decode for %r",
                exc,
                os.path.basename(filepath),
            )
            audio, sr = self._read_via_ffmpeg(
                filepath, sample_rate=sample_rate, mono=mono
            )

        # librosa returns (channels, samples); our convention is (samples, channels).
        if not mono and np.ndim(audio) == 2 and audio.shape[0] < audio.shape[1]:
            audio = audio.T
        return np.asarray(audio, dtype=np.float32), int(sr)

    def write(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        bitrate: str | None = None,
        bit_depth: int = DEFAULT_BIT_DEPTH,
        vbr: bool = True,
    ) -> None:
        """Write audio to a file; the format is inferred from the extension.

        Args:
            audio:       Float32 array with values in [-1, 1].
            sample_rate: Sample rate of the audio.
            output_path: Destination path (extension determines format).
            bitrate:     For lossy formats ('320k', '256k', 'V0', quality 0–10).
                         None = format default (see FORMAT_REGISTRY).
            bit_depth:   For lossless formats: 16, 24, or 32.
            vbr:         MP3: True = VBR, False = CBR at the given bitrate.
        """
        ext = "." + self.detect_format(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        entry = FORMAT_REGISTRY.get(ext)
        if entry is None:
            raise UnsupportedFormatError(
                f"Unsupported output format {ext!r}. "
                f"Supported: {', '.join(sorted(FORMAT_REGISTRY))}"
            )

        category, _desc, default_br = entry
        effective_bitrate = bitrate or default_br

        # Safety: clip to [-1, 1] before encoding.
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        if category == "lossless":
            self._write_lossless(audio, sample_rate, output_path, ext, bit_depth)
        else:
            self._write_lossy_ffmpeg(
                audio, sample_rate, output_path, ext, effective_bitrate, vbr
            )

    def format_info(self, filepath: str) -> str:
        """Return a human-readable description of the target format."""
        ext = "." + self.detect_format(filepath)
        entry = FORMAT_REGISTRY.get(ext)
        if entry is None:
            return f"Unknown format ({ext})"
        category, desc, default_br = entry
        if category == "lossless":
            return f"{desc} (lossless)"
        return f"{desc} (lossy, default quality: {default_br})"

    # ------------------------------------------------------------------
    # Private: lossless write (soundfile)
    # ------------------------------------------------------------------

    def _write_lossless(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        ext: str,
        bit_depth: int,
    ) -> None:
        subtype = _SUBTYPE_MAP.get(bit_depth, "PCM_24")
        if ext == ".flac" and bit_depth == 32:
            subtype = "PCM_24"  # FLAC maximum is 24-bit

        try:
            sf.write(
                output_path,
                audio,
                sample_rate,
                subtype=subtype,
                format=_FORMAT_MAP.get(ext, "WAV"),
            )
        except Exception as exc:
            raise EncodingError(
                f"soundfile failed to write {output_path!r}: {exc}"
            ) from exc

        logger.debug(
            "Written lossless %s %d-bit @ %d Hz → %s",
            _FORMAT_MAP.get(ext, "WAV"),
            bit_depth,
            sample_rate,
            output_path,
        )

    # ------------------------------------------------------------------
    # Private: lossy write (ffmpeg)
    # ------------------------------------------------------------------

    def _write_lossy_ffmpeg(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        ext: str,
        bitrate: str | None,
        vbr: bool,
    ) -> None:
        if not self._check_ffmpeg():
            raise EncodingError(
                "ffmpeg not found. Install ffmpeg to encode lossy formats "
                "(MP3, M4A, OGG, OPUS). Alternatively use .wav or .flac output."
            )

        # Write a temp PCM WAV → pipe through ffmpeg → final output.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            sf.write(tmp_wav, audio, sample_rate, subtype="PCM_32", format="WAV")
            cmd = self._build_ffmpeg_cmd(
                tmp_wav, output_path, ext, bitrate, vbr, sample_rate
            )
            logger.debug("ffmpeg cmd: %s", " ".join(cmd))
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise EncodingError(
                    f"ffmpeg encoding failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
                )
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        logger.info(
            "Written %s [%s] → %s", ext.upper(), bitrate or "default", output_path
        )

    def _build_ffmpeg_cmd(
        self,
        input_wav: str,
        output_path: str,
        ext: str,
        bitrate: str | None,
        vbr: bool,
        sample_rate: int,
    ) -> list[str]:
        cmd = [self.ffmpeg_path, "-y", "-i", input_wav]

        if ext == ".mp3":
            cmd += ["-codec:a", "libmp3lame"]
            if vbr and (bitrate is None or bitrate.startswith("V")):
                q = (
                    bitrate.replace("V", "")
                    if bitrate and bitrate.startswith("V")
                    else "0"
                )
                cmd += ["-q:a", q]
            else:
                br = bitrate if bitrate and not bitrate.startswith("V") else "320k"
                cmd += ["-b:a", br]
            cmd += ["-id3v2_version", "3"]

        elif ext in (".m4a", ".aac"):
            if self._has_codec("libfdk_aac"):
                cmd += ["-codec:a", "libfdk_aac", "-vbr", "5"]
            else:
                cmd += ["-codec:a", "aac", "-b:a", bitrate or "256k"]
            if ext == ".m4a":
                cmd += ["-movflags", "+faststart"]

        elif ext == ".ogg":
            cmd += ["-codec:a", "libvorbis"]
            q = bitrate if bitrate and not bitrate.endswith("k") else "8"
            cmd += ["-q:a", q]

        elif ext == ".opus":
            cmd += ["-codec:a", "libopus", "-b:a", bitrate or "192k", "-vbr", "on"]

        elif ext == ".wma":
            cmd += ["-codec:a", "wmav2", "-b:a", bitrate or "192k"]

        else:  # generic: let ffmpeg pick a codec
            if bitrate:
                cmd += ["-b:a", bitrate]

        cmd.append(output_path)
        return cmd

    # ------------------------------------------------------------------
    # Private: ffmpeg read fallback
    # ------------------------------------------------------------------

    def _read_via_ffmpeg(
        self,
        filepath: str,
        sample_rate: int | None,
        mono: bool,
    ) -> tuple[np.ndarray, int]:
        """Decode any file to a temp WAV via ffmpeg, then load with librosa."""
        import librosa

        if not self._check_ffmpeg():
            raise AudioLoadError(
                f"Cannot read {filepath!r}: neither librosa nor ffmpeg could decode it. "
                "Install ffmpeg for broader format support."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            cmd = [self.ffmpeg_path, "-y", "-i", filepath, "-f", "wav", tmp_wav]
            result = subprocess.run(cmd, capture_output=True, check=False)
            if result.returncode != 0:
                raise AudioLoadError(f"ffmpeg could not decode {filepath!r}")
            audio, sr = librosa.load(tmp_wav, sr=sample_rate, mono=mono)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        return np.asarray(audio, dtype=np.float32), int(sr)

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _check_ffmpeg(self) -> bool:
        if self._ffmpeg_available is None:
            try:
                result = subprocess.run(
                    [self.ffmpeg_path, "-version"], capture_output=True, check=False
                )
                self._ffmpeg_available = result.returncode == 0
            except FileNotFoundError:
                self._ffmpeg_available = False
        return bool(self._ffmpeg_available)

    def _has_codec(self, codec_name: str) -> bool:
        """Check whether local ffmpeg was compiled with a specific codec encoder."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-codecs"],
                capture_output=True,
                text=True,
                check=False,
            )
            return codec_name in result.stdout
        except Exception:  # noqa: BLE001
            return False
