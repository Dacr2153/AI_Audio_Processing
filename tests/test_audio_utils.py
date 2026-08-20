"""Tests for audio_utils low-level helpers."""

from __future__ import annotations

import numpy as np
import pytest

from audio_restoration.dsp import audio_utils


def test_as_float_matches_dtype():
    x = np.array([1, 2, 3], dtype=np.int16)
    assert audio_utils.as_float(x).dtype == np.float64


def test_match_length_trim_and_pad():
    a = np.arange(10.0)
    assert len(audio_utils.match_length(a, 5)) == 5
    padded = audio_utils.match_length(a, 20)
    assert len(padded) == 20
    assert np.all(padded[10:] == 0.0)


def test_resample_lengths():
    x = np.zeros(2205, dtype=np.float64)
    out = audio_utils.resample(x, 22050, 44100)
    assert len(out) == 4410
    assert out.dtype == np.float64


def test_resample_preserves_dtype():
    x = np.zeros(2205, dtype=np.float32)
    out = audio_utils.resample(x, 22050, 44100, dtype=np.float32)
    assert out.dtype == np.float32


def test_resample_noop():
    x = np.arange(100.0)
    assert audio_utils.resample(x, 44100, 44100) is x


def test_resample_stereo_shape():
    x = np.zeros((2205, 2))
    out = audio_utils.resample(x, 22050, 44100)
    assert out.shape == (4410, 2)


@pytest.mark.parametrize("shape", [(100,), (100, 2)])
def test_process_channels_mono_and_stereo(shape):
    x = np.ones(shape, dtype=np.float32)

    def halve(ch, _sr):
        return ch * 0.5

    out = audio_utils.process_channels(x, halve, 44100)
    assert out.shape == shape
    assert out.dtype == np.float32
    assert np.all(out == 0.5)


def test_process_channels_passes_sample_rate():
    x = np.ones(8, dtype=np.float64)
    seen = []

    def record(ch, sr):
        seen.append(sr)
        return ch

    audio_utils.process_channels(x, record, 1234)
    assert seen == [1234]


def test_process_channels_rejects_bad_shape():
    with pytest.raises(ValueError, match="Unsupported audio shape"):
        audio_utils.process_channels(np.zeros((2, 3, 4)), lambda c: c)


def test_to_mono_averages_stereo():
    x = np.column_stack([np.ones(10), np.ones(10) * 3.0])
    assert np.all(audio_utils.to_mono(x) == 2.0)


def test_to_mono_identity_for_1d():
    x = np.arange(10.0)
    assert audio_utils.to_mono(x) is x


def test_validate_audio_rejects_empty():
    with pytest.raises(ValueError, match="empty"):
        audio_utils.validate_audio(np.array([]))


def test_validate_audio_rejects_nan():
    with pytest.raises(ValueError, match="non-finite"):
        audio_utils.validate_audio(np.array([0.0, np.nan, 1.0]))


def test_validate_audio_converts_to_float32():
    x = np.arange(10, dtype=np.int16)
    out = audio_utils.validate_audio(x)
    assert out.dtype == np.float32


def test_peak_normalize_scales_to_unity():
    x = np.array([0.5, -0.25], dtype=np.float32)
    out = audio_utils.peak_normalize(x)
    assert np.max(np.abs(out)) == pytest.approx(1.0)


def test_peak_normalize_silent_is_noop():
    x = np.zeros(10, dtype=np.float32)
    assert audio_utils.peak_normalize(x) is x


def test_rms_db_silence():
    assert audio_utils.rms_db(np.zeros(16)) == pytest.approx(-120.0)


def test_rms_db_fullscale():
    assert audio_utils.rms_db(np.ones(16)) == pytest.approx(0.0, abs=1e-6)


def test_peak_db_silence():
    assert audio_utils.peak_db(np.zeros(16)) == pytest.approx(-120.0)


def test_peak_db_known():
    assert audio_utils.peak_db(np.array([0.5, -0.25])) == pytest.approx(
        -6.0206, abs=1e-3
    )
