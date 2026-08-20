"""Tests for FormatHandler I/O and AudioPreprocessor."""

from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from audio_restoration.exceptions import AudioLoadError, UnsupportedFormatError
from audio_restoration.io.format_handler import FormatHandler
from audio_restoration.io.preprocessing import AudioPreprocessor


def test_detect_format_lowercases():
    h = FormatHandler()
    assert h.detect_format("song.MP3") == "mp3"
    assert h.detect_format("track.flac") == "flac"


def test_detect_format_no_extension_raises():
    h = FormatHandler()
    with pytest.raises(UnsupportedFormatError):
        h.detect_format("no_extension")


def test_lossless_write_read_roundtrip(tmp_path, stereo_signal, sample_rate):
    h = FormatHandler()
    out = str(tmp_path / "roundtrip.wav")
    h.write(stereo_signal, sample_rate, out, bit_depth=24)
    audio, sr = h.read(out, mono=False)
    assert sr == sample_rate
    assert audio.shape == stereo_signal.shape
    assert audio.dtype == np.float32
    # PCM 24-bit quantisation → small tolerance.
    assert np.allclose(audio, stereo_signal, atol=2e-7)


def test_lossy_write_via_ffmpeg(tmp_path, stereo_signal, sample_rate):
    h = FormatHandler()
    out = str(tmp_path / "lossy.mp3")
    h.write(stereo_signal, sample_rate, out, bitrate="128k")
    assert sf.info(out).format == "MP3"


def test_unsupported_output_raises(tmp_path, stereo_signal, sample_rate):
    h = FormatHandler()
    with pytest.raises(UnsupportedFormatError):
        h.write(stereo_signal, sample_rate, str(tmp_path / "out.weird"))


def test_read_missing_file_raises(tmp_path):
    h = FormatHandler()
    with pytest.raises(AudioLoadError):
        h.read(str(tmp_path / "missing.wav"))


def test_read_keeps_stereo_layout(wav_file):
    h = FormatHandler()
    audio, _sr = h.read(wav_file, mono=False)
    assert audio.ndim == 2
    assert audio.shape[1] == 2


def test_format_info():
    h = FormatHandler()
    assert "lossless" in h.format_info("out.wav")
    assert "lossy" in h.format_info("out.mp3")
    assert "Unknown" in h.format_info("out.xyz")


def test_preprocessor_preserves_channels(wav_file, sample_rate):
    prep = AudioPreprocessor(normalize=True)
    audio, _sr = prep.load_and_prepare(wav_file)
    assert audio.ndim == 2
    assert audio.shape[1] == 2


def test_preprocessor_mono_downmix(wav_file, sample_rate):
    prep = AudioPreprocessor(mono=True)
    audio, _sr = prep.load_and_prepare(wav_file)
    assert audio.ndim == 1


def test_prepare_for_super_resolution_shape(mono_signal, sample_rate):
    out = AudioPreprocessor.prepare_for_super_resolution(mono_signal, sample_rate)
    assert out.shape == mono_signal.shape
