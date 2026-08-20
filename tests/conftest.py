"""Shared pytest fixtures for the audio-restoration test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


@pytest.fixture
def sample_rate() -> int:
    return 22050


@pytest.fixture
def mono_signal(sample_rate: int) -> np.ndarray:
    """A clean 1-second 440 Hz tone."""
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    return (0.5 * np.sin(2.0 * np.pi * 440.0 * t)).astype(np.float32)


@pytest.fixture
def stereo_signal(sample_rate: int) -> np.ndarray:
    """A 1-second stereo signal with distinct channels."""
    t = np.arange(sample_rate, dtype=np.float64) / sample_rate
    left = 0.5 * np.sin(2.0 * np.pi * 440.0 * t)
    right = 0.4 * np.sin(2.0 * np.pi * 660.0 * t)
    return np.column_stack([left, right]).astype(np.float32)


@pytest.fixture
def noisy_signal(mono_signal) -> np.ndarray:
    """A clean tone polluted with broadband Gaussian noise."""
    rng = np.random.default_rng(42)
    noise = rng.normal(0.0, 0.05, size=len(mono_signal)).astype(np.float32)
    return np.clip(mono_signal + noise, -1.0, 1.0).astype(np.float32)


@pytest.fixture
def wav_file(tmp_path: Path, stereo_signal: np.ndarray, sample_rate: int) -> str:
    """A real stereo WAV file on disk."""
    path = tmp_path / "stereo.wav"
    sf.write(path, stereo_signal, sample_rate, subtype="PCM_16")
    return str(path)
