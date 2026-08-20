"""Global constants shared across the audio restoration package."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Sample rates
# ---------------------------------------------------------------------------
#: DeepFilterNet operates at exactly 48 kHz.
DEEPFILTER_SR: int = 48_000
#: AudioSR / DeepFilterNet target sample rate.
TARGET_SR: int = 48_000
#: Native output sample rate of Demucs.
DEMUCS_SR: int = 44_100

# ---------------------------------------------------------------------------
# Default block sizes (kept consistent across STFT-based processors)
# ---------------------------------------------------------------------------
DEFAULT_N_FFT: int = 2048
DEFAULT_HOP_LENGTH: int = 512

# ---------------------------------------------------------------------------
# Demucs source separation
# ---------------------------------------------------------------------------
DEMUCS_STEMS: tuple[str, ...] = ("drums", "bass", "other", "vocals")
DEMUCS_MODELS: tuple[str, ...] = ("htdemucs_ft", "htdemucs", "mdx_extra")

# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------
DENOISE_METHODS: tuple[str, ...] = (
    "auto",
    "music",
    "noisereduce",
    "deepfilternet",
    "wavelet",
)

# ---------------------------------------------------------------------------
# Audio extensions recognised for batch scanning / encoding
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".wav",
        ".flac",
        ".aiff",
        ".aif",
        ".mp3",
        ".m4a",
        ".aac",
        ".ogg",
        ".opus",
        ".wma",
        ".mp4",
    }
)
