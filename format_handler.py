"""
format_handler.py — Multi-format audio I/O for the Restoration Pipeline.

Supported INPUT formats (read via librosa + soundfile + ffmpeg fallback):
  WAV, MP3, FLAC, OGG/Vorbis, AIFF, M4A/AAC, OPUS, WMA, WV (WavPack),
  AU, CAF, W64, RF64, APE, and any other format ffmpeg can decode.

Supported OUTPUT formats:
  Lossless (via soundfile):
    .wav   — PCM 16/24/32-bit or float32
    .flac  — FLAC PCM 24-bit (lossless)
    .aiff  — AIFF PCM 24-bit

  Lossy (via ffmpeg — full bitrate/quality control):
    .mp3   — MP3 via libmp3lame (VBR V0 ~245kbps, or CBR up to 320kbps)
    .m4a   — AAC via libfdk_aac or native aac (256kbps default)
    .aac   — AAC raw (256kbps)
    .ogg   — OGG Vorbis (quality 8 / ~256kbps)
    .opus  — Opus (192kbps transparent quality)

Usage:
    from format_handler import FormatHandler

    # Detect output format
    handler = FormatHandler()
    fmt = handler.detect_format("output.mp3")    # → 'mp3'
    fmt = handler.detect_format("output.flac")   # → 'flac'

    # Write audio
    handler.write(audio_float32, sample_rate, "output.mp3", bitrate="320k")
    handler.write(audio_float32, sample_rate, "output.flac")
    handler.write(audio_float32, sample_rate, "output.wav", bit_depth=24)

    # Read audio (thin wrapper around librosa)
    audio, sr = handler.read("song.mp3", sample_rate=None)
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

logger = logging.getLogger("format_handler")

# ---------------------------------------------------------------------------
# Format registry
# ---------------------------------------------------------------------------

#: Maps file extension → (category, description, default_bitrate_or_None)
FORMAT_REGISTRY: dict[str, tuple[str, str, Optional[str]]] = {
    # Lossless
    ".wav":  ("lossless", "WAV PCM",          None),
    ".flac": ("lossless", "FLAC (lossless)",  None),
    ".aiff": ("lossless", "AIFF PCM",         None),
    ".aif":  ("lossless", "AIFF PCM",         None),
    # Lossy — encoded via ffmpeg
    ".mp3":  ("lossy",    "MP3 (libmp3lame)", "V0"),    # VBR V0 ≈ 245 kbps
    ".m4a":  ("lossy",    "AAC/M4A",          "256k"),
    ".aac":  ("lossy",    "AAC",              "256k"),
    ".ogg":  ("lossy",    "OGG Vorbis",       "8"),     # quality scale 0–10
    ".opus": ("lossy",    "Opus",             "192k"),
    ".wma":  ("lossy",    "WMA",              "192k"),
}

#: All extensions that can be READ by librosa without ffmpeg
LIBROSA_NATIVE_READ = {".wav", ".flac", ".ogg", ".aiff", ".aif", ".au", ".caf", ".w64", ".rf64", ".mp3"}

#: Default bit depth for lossless writes
DEFAULT_BIT_DEPTH = 24

#: Maximum value for clipping check
CLIP_THRESHOLD = 1.0


class FormatHandler:
    """
    High-quality multi-format audio I/O handler.

    Reads any format via librosa (falls back to ffmpeg/audioread).
    Writes lossless via soundfile, lossy via ffmpeg subprocess.
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path
        self._ffmpeg_available: Optional[bool] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_format(self, filepath: str) -> str:
        """
        Returns the lowercase file extension without the dot.
        e.g. "song.mp3" → "mp3", "track.FLAC" → "flac"
        """
        ext = os.path.splitext(filepath)[1].lower()
        if not ext:
            raise ValueError(
                f"Cannot determine format: '{filepath}' has no file extension. "
                "Please use an extension like .wav, .flac, .mp3, .m4a, .ogg, .opus."
            )
        return ext[1:]  # strip the dot

    def is_lossless(self, filepath: str) -> bool:
        ext = "." + self.detect_format(filepath)
        entry = FORMAT_REGISTRY.get(ext)
        return entry is not None and entry[0] == "lossless"

    def is_lossy(self, filepath: str) -> bool:
        ext = "." + self.detect_format(filepath)
        entry = FORMAT_REGISTRY.get(ext)
        return entry is not None and entry[0] == "lossy"

    def supported_output_formats(self) -> list[str]:
        """Returns a list of supported output extensions (with dot)."""
        return list(FORMAT_REGISTRY.keys())

    def read(
        self,
        filepath: str,
        sample_rate: Optional[int] = None,
        mono: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        Read any audio file into a float32 numpy array.

        Args:
            filepath:    Path to the audio file.
            sample_rate: If given, resample to this rate. None = native SR.
            mono:        Convert to mono if True.

        Returns:
            (audio: np.ndarray float32, sample_rate: int)
        """
        import librosa

        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        ext = os.path.splitext(filepath)[1].lower()

        try:
            audio, sr = librosa.load(
                filepath,
                sr=sample_rate,
                mono=mono,
            )
            logger.debug("Loaded '%s' via librosa/soundfile [%d Hz]", os.path.basename(filepath), sr)
        except Exception as e_librosa:
            # Fallback: decode via ffmpeg to a temp WAV then load
            logger.warning(
                "librosa failed (%s) — trying ffmpeg decode for '%s'",
                e_librosa, os.path.basename(filepath),
            )
            audio, sr = self._read_via_ffmpeg(filepath, sample_rate=sample_rate, mono=mono)

        return audio.astype(np.float32), sr

    def write(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        bitrate: Optional[str] = None,
        bit_depth: int = DEFAULT_BIT_DEPTH,
        vbr: bool = True,
    ) -> None:
        """
        Write audio to a file. Format is inferred from the file extension.

        Args:
            audio:       Float32 array, values in [-1, 1].
            sample_rate: Sample rate of the audio.
            output_path: Destination path (extension determines format).
            bitrate:     For lossy formats. Examples: '320k', '256k', 'V0'.
                         None = use format default (see FORMAT_REGISTRY).
            bit_depth:   For lossless formats: 16, 24, or 32.
            vbr:         For MP3: if True use VBR (best quality/size ratio),
                         if False use CBR at the specified bitrate.
        """
        ext = "." + self.detect_format(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if ext not in FORMAT_REGISTRY:
            raise ValueError(
                f"Unsupported output format '{ext}'. "
                f"Supported: {', '.join(sorted(FORMAT_REGISTRY.keys()))}"
            )

        category = FORMAT_REGISTRY[ext][0]
        default_br = FORMAT_REGISTRY[ext][2]
        effective_bitrate = bitrate or default_br

        # Safety: clip to [-1, 1] before encoding
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        if category == "lossless":
            self._write_lossless(audio, sample_rate, output_path, ext, bit_depth)
        else:
            self._write_lossy_ffmpeg(
                audio, sample_rate, output_path, ext, effective_bitrate, vbr
            )

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
        subtype_map = {
            16: "PCM_16",
            24: "PCM_24",
            32: "PCM_32",
        }
        subtype = subtype_map.get(bit_depth, "PCM_24")

        if ext == ".flac" and bit_depth == 32:
            subtype = "PCM_24"  # FLAC max is 24-bit

        format_map = {
            ".wav":  "WAV",
            ".flac": "FLAC",
            ".aiff": "AIFF",
            ".aif":  "AIFF",
        }
        sf_format = format_map.get(ext, "WAV")

        logger.debug("Writing %s %s %d-bit @ %d Hz → %s",
                     sf_format, subtype, bit_depth, sample_rate, output_path)
        sf.write(output_path, audio, sample_rate, subtype=subtype, format=sf_format)

    # ------------------------------------------------------------------
    # Private: lossy write (ffmpeg)
    # ------------------------------------------------------------------

    def _write_lossy_ffmpeg(
        self,
        audio: np.ndarray,
        sample_rate: int,
        output_path: str,
        ext: str,
        bitrate: Optional[str],
        vbr: bool,
    ) -> None:
        if not self._check_ffmpeg():
            raise RuntimeError(
                "ffmpeg not found. Install ffmpeg to encode lossy formats "
                f"(MP3, M4A, OGG, OPUS). Alternatively, use .wav or .flac output."
            )

        # Write a temp PCM WAV → pipe through ffmpeg → final output
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            sf.write(tmp_wav, audio, sample_rate, subtype="PCM_32", format="WAV")
            cmd = self._build_ffmpeg_cmd(tmp_wav, output_path, ext, bitrate, vbr, sample_rate)
            logger.debug("ffmpeg cmd: %s", " ".join(cmd))
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg encoding failed (exit {result.returncode}):\n{result.stderr[-2000:]}"
                )
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)

        logger.info("Written %s [%s] → %s", ext.upper(), bitrate or "default", output_path)

    def _build_ffmpeg_cmd(
        self,
        input_wav: str,
        output_path: str,
        ext: str,
        bitrate: Optional[str],
        vbr: bool,
        sample_rate: int,
    ) -> list[str]:
        cmd = [
            self.ffmpeg_path,
            "-y",               # overwrite without asking
            "-i", input_wav,
        ]

        if ext == ".mp3":
            cmd += ["-codec:a", "libmp3lame"]
            if vbr and (bitrate is None or bitrate.startswith("V")):
                # VBR quality: V0 (best) … V9 (worst)
                q = bitrate.replace("V", "") if bitrate and bitrate.startswith("V") else "0"
                cmd += ["-q:a", q]
            else:
                br = bitrate if bitrate and not bitrate.startswith("V") else "320k"
                cmd += ["-b:a", br]
            # ID3v2.3 tags for broad player compatibility
            cmd += ["-id3v2_version", "3"]

        elif ext in (".m4a", ".aac"):
            # Prefer libfdk_aac if available, else native aac
            if self._has_codec("libfdk_aac"):
                cmd += ["-codec:a", "libfdk_aac", "-vbr", "5"]
            else:
                cmd += ["-codec:a", "aac", "-b:a", bitrate or "256k"]
            if ext == ".m4a":
                cmd += ["-movflags", "+faststart"]

        elif ext == ".ogg":
            cmd += ["-codec:a", "libvorbis"]
            # OGG quality scale: -1 to 10 (10 = best, ~500kbps)
            q = bitrate if bitrate and not bitrate.endswith("k") else "8"
            cmd += ["-q:a", q]

        elif ext == ".opus":
            cmd += ["-codec:a", "libopus", "-b:a", bitrate or "192k", "-vbr", "on"]

        elif ext == ".wma":
            cmd += ["-codec:a", "wmav2", "-b:a", bitrate or "192k"]

        else:
            # Generic: let ffmpeg pick codec
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
        sample_rate: Optional[int],
        mono: bool,
    ) -> Tuple[np.ndarray, int]:
        """Decode any file to a temp WAV via ffmpeg, then load with soundfile."""
        import librosa

        if not self._check_ffmpeg():
            raise RuntimeError(
                f"Cannot read '{filepath}': neither librosa nor ffmpeg could decode it. "
                "Install ffmpeg for broader format support."
            )

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_wav = tmp.name
        try:
            cmd = [self.ffmpeg_path, "-y", "-i", filepath, "-f", "wav", tmp_wav]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"ffmpeg could not decode '{filepath}'"
                )
            audio, sr = librosa.load(tmp_wav, sr=sample_rate, mono=mono)
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        return audio, sr

    # ------------------------------------------------------------------
    # Private: helpers
    # ------------------------------------------------------------------

    def _check_ffmpeg(self) -> bool:
        if self._ffmpeg_available is None:
            try:
                result = subprocess.run(
                    [self.ffmpeg_path, "-version"],
                    capture_output=True,
                )
                self._ffmpeg_available = result.returncode == 0
            except FileNotFoundError:
                self._ffmpeg_available = False
        return self._ffmpeg_available

    def _has_codec(self, codec_name: str) -> bool:
        """Check if ffmpeg was compiled with a specific codec encoder."""
        try:
            result = subprocess.run(
                [self.ffmpeg_path, "-codecs"],
                capture_output=True,
                text=True,
            )
            return codec_name in result.stdout
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Convenience: format info string
    # ------------------------------------------------------------------

    def format_info(self, filepath: str) -> str:
        """Returns a human-readable description of the format."""
        ext = "." + self.detect_format(filepath)
        entry = FORMAT_REGISTRY.get(ext)
        if entry:
            category, desc, default_br = entry
            if category == "lossless":
                return f"{desc} (lossless)"
            else:
                return f"{desc} (lossy, default quality: {default_br})"
        return f"Unknown format ({ext})"
