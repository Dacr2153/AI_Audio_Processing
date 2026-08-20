"""Exception hierarchy for the audio restoration package.

Every error surfaced by the package derives from :class:`AudioRestorationError`
so that callers (and the CLI) can handle failures uniformly and map them to
stable exit codes.
"""

from __future__ import annotations


class AudioRestorationError(Exception):
    """Base class for all package-specific errors."""


class AudioLoadError(AudioRestorationError):
    """Raised when an input audio file cannot be decoded/loaded."""


class UnsupportedFormatError(AudioRestorationError):
    """Raised when an output path has no usable / supported extension."""


class EncodingError(AudioRestorationError):
    """Raised when an output file cannot be written or encoded."""


class NeuralModelUnavailableError(AudioRestorationError):
    """Raised when a required neural model/dependency is not installed."""


class ValidationError(AudioRestorationError):
    """Raised when configuration or input values are invalid."""


class PipelineProcessingError(AudioRestorationError):
    """Raised when a stage in the pipeline fails irrecoverably."""
