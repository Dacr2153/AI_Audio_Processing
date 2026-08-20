"""audio-restoration — AI-powered audio restoration toolkit.

Public API::

    from audio_restoration import RestorationPipeline, PipelineConfig
    from audio_restoration.dsp import Denoiser, Declicker
    from audio_restoration.io import FormatHandler, AudioPreprocessor
"""

from __future__ import annotations

__version__ = "2.0.0"

from .config import (
    GENRE_PRESETS,
    DeclickConfig,
    DehumConfig,
    DenoiseConfig,
    EQConfig,
    LoudnessConfig,
    MSConfig,
    MultibandConfig,
    OutputConfig,
    PipelineConfig,
    ReportConfig,
    SourceSeparationConfig,
    SuperResolutionConfig,
    WowFlutterConfig,
)
from .exceptions import (
    AudioLoadError,
    AudioRestorationError,
    EncodingError,
    NeuralModelUnavailableError,
    PipelineProcessingError,
    UnsupportedFormatError,
    ValidationError,
)
from .pipeline import RestorationPipeline

__all__ = [
    "GENRE_PRESETS",
    "AudioLoadError",
    "AudioRestorationError",
    "DeclickConfig",
    "DehumConfig",
    "DenoiseConfig",
    "EQConfig",
    "EncodingError",
    "LoudnessConfig",
    "MSConfig",
    "MultibandConfig",
    "NeuralModelUnavailableError",
    "OutputConfig",
    "PipelineConfig",
    "PipelineProcessingError",
    "ReportConfig",
    "RestorationPipeline",
    "SourceSeparationConfig",
    "SuperResolutionConfig",
    "UnsupportedFormatError",
    "ValidationError",
    "WowFlutterConfig",
]
