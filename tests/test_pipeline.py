"""Tests for QualityMetrics and BatchReport."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from audio_restoration.config import PipelineConfig
from audio_restoration.pipeline import RestorationPipeline
from audio_restoration.reporting.metrics import QualityMetrics


def test_snr_identical_is_capped():
    x = np.ones(64)
    assert QualityMetrics.snr(x, x) == pytest.approx(120.0)


def test_snr_degraded_lower():
    signal = np.ones(64)
    noisy = signal + np.full(64, 0.5)
    assert QualityMetrics.snr(signal, noisy) < 20.0


def test_psnr_caps_at_identical():
    x = np.ones(64)
    assert QualityMetrics.psnr(x, x) == pytest.approx(120.0)


def test_rms_and_peak_zero_db():
    assert QualityMetrics.rms_db(np.ones(64)) == pytest.approx(0.0, abs=1e-6)
    assert QualityMetrics.peak_db(np.ones(64)) == pytest.approx(0.0, abs=1e-6)


def test_compare_returns_expected_keys(mono_signal, sample_rate):
    report = QualityMetrics().compare(mono_signal, mono_signal * 0.9, sample_rate)
    assert "snr_db" in report
    assert "psnr_db" in report
    assert "original_rms_db" in report
    assert "restored_spectral_centroid_hz" in report
    assert "restored_hf_energy_ratio" in report


def test_format_report_renders(mono_signal, sample_rate):
    report = QualityMetrics().compare(mono_signal, mono_signal, sample_rate)
    text = QualityMetrics.format_report(report)
    assert "AUDIO RESTORATION QUALITY REPORT" in text
    assert "SNR" in text


@pytest.mark.slow
def test_pipeline_end_to_end_stereo(tmp_path, wav_file, sample_rate, stereo_signal):
    """Run the whole pipeline on a real stereo file and check stereo survives."""
    out = str(tmp_path / "result.wav")
    cfg = PipelineConfig(
        denoise_method="wavelet",
        save_comparison_plot=False,
        print_metrics=False,
        enable_ms=True,
        eq_rumble_filter=False,
    )
    report = RestorationPipeline(cfg).restore(wav_file, out)
    import soundfile as sf

    result, sr = sf.read(out)
    assert sr == sample_rate
    assert result.ndim == 2
    assert result.shape[1] == 2
    assert report["snr_db"] >= -60.0  # sanity bound


# ---------------------------------------------------------------------------
# Pipeline branching: source separation, super-resolution, errors
# ---------------------------------------------------------------------------


def _config(**overrides) -> PipelineConfig:
    base = {
        "denoise_method": "wavelet",
        "save_comparison_plot": False,
        "print_metrics": False,
        "lufs_target": None,
        "normalize_output": True,
    }
    return PipelineConfig(**{**base, **overrides})


def test_pipeline_source_separation_remix(tmp_path, wav_file, sample_rate):
    stems = {
        name: np.zeros((5000, 2), dtype=np.float32)
        for name in ("drums", "bass", "other", "vocals")
    }
    cfg = _config(enable_source_separation=True)
    with patch("audio_restoration.pipeline.SourceSeparator") as mock_cls:
        inst = mock_cls.return_value
        inst.is_available = True
        inst.separate_from_array.return_value = stems
        report = RestorationPipeline(cfg).restore(wav_file, str(tmp_path / "out.wav"))
    assert "snr_db" in report


def test_pipeline_source_separation_unavailable(
    tmp_path, wav_file, sample_rate, monkeypatch
):
    cfg = _config(enable_source_separation=True)
    with patch("audio_restoration.pipeline.SourceSeparator") as mock_cls:
        inst = mock_cls.return_value
        inst.is_available = False
        report = RestorationPipeline(cfg).restore(wav_file, str(tmp_path / "out.wav"))
    assert "snr_db" in report


def test_pipeline_source_separation_empty_stems(tmp_path, wav_file):
    cfg = _config(enable_source_separation=True)
    with patch("audio_restoration.pipeline.SourceSeparator") as mock_cls:
        inst = mock_cls.return_value
        inst.is_available = True
        inst.separate_from_array.return_value = {}
        report = RestorationPipeline(cfg).restore(wav_file, str(tmp_path / "out.wav"))
    assert "snr_db" in report


def test_pipeline_super_resolution(tmp_path, wav_file, sample_rate):
    cfg = _config(enable_super_resolution=True)
    with patch("audio_restoration.pipeline.SuperResolution") as mock_cls:
        inst = mock_cls.return_value
        inst.upsample.return_value = (
            np.zeros((10000, 2), dtype=np.float32),
            48_000,
        )
        report = RestorationPipeline(cfg).restore(wav_file, str(tmp_path / "out.wav"))
    assert "snr_db" in report


def test_pipeline_missing_input(tmp_path):
    from audio_restoration.exceptions import AudioLoadError

    cfg = _config()
    with pytest.raises(AudioLoadError):
        RestorationPipeline(cfg).restore(
            str(tmp_path / "nope.wav"), str(tmp_path / "o.wav")
        )
