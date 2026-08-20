"""Digital signal processing stages for the audio restoration pipeline."""

from __future__ import annotations

from .declicker import Declicker
from .dehum import Dehummer
from .denoiser import Denoiser
from .equalizer import AudioEqualizer
from .ms_processor import MSProcessor
from .multiband_compressor import BandSettings, MultibandCompressor
from .wow_flutter import WowFlutterCorrector

__all__ = [
    "AudioEqualizer",
    "BandSettings",
    "Declicker",
    "Dehummer",
    "Denoiser",
    "MSProcessor",
    "MultibandCompressor",
    "WowFlutterCorrector",
]
