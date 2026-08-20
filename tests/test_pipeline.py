"""Tests for QualityMetrics and BatchReport."""

from __future__ import annotations

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
