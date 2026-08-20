"""Tests for PipelineConfig construction, legacy mapping and validation."""

from __future__ import annotations

import pytest

from audio_restoration.config import (
    GENRE_PRESETS,
    EQConfig,
    MSConfig,
    PipelineConfig,
)
from audio_restoration.exceptions import ValidationError


def test_default_config():
    cfg = PipelineConfig()
    assert cfg.denoise.method == "music"
    assert cfg.eq.bass_gain_db == pytest.approx(2.5)
    assert cfg.output.bit_depth == 24
    assert cfg.loudness.target_lufs == pytest.approx(-14.0)


def test_legacy_flat_kwargs():
    cfg = PipelineConfig(
        denoise_method="auto",
        enable_ms=True,
        ms_side_gain=1.4,
        bass_gain_db=4.0,
        lufs_target=None,
    )
    assert cfg.denoise.method == "auto"
    assert cfg.ms.enabled is True
    assert cfg.ms.side_gain == pytest.approx(1.4)
    assert cfg.eq.bass_gain_db == pytest.approx(4.0)
    assert cfg.loudness.target_lufs is None


def test_unknown_legacy_kwarg_raises():
    with pytest.raises(TypeError, match="not_a_real_option"):
        PipelineConfig(not_a_real_option=1)


def test_grouped_config_wins_over_default():
    eq = EQConfig(bass_gain_db=-2.0)
    cfg = PipelineConfig(eq=eq)
    assert cfg.eq.bass_gain_db == pytest.approx(-2.0)


def test_genre_preset_applies_and_is_overridable():
    cfg = PipelineConfig(genre="jazz")
    assert cfg.eq.bass_gain_db == pytest.approx(GENRE_PRESETS["jazz"]["bass_gain_db"])

    # Explicit CLI-style values are never overwritten by the preset.
    cfg2 = PipelineConfig(genre="jazz", bass_gain_db=9.5)
    assert cfg2.eq.bass_gain_db == pytest.approx(9.5)


def test_invalid_denoise_method_raises():
    with pytest.raises(ValidationError, match="denoise.method"):
        PipelineConfig(denoise_method="bogus")


def test_invalid_prop_decrease_raises():
    with pytest.raises(ValidationError, match="prop_decrease"):
        PipelineConfig(denoise_prop_decrease=1.5)


def test_invalid_multiband_crossovers_raise():
    with pytest.raises(ValidationError, match="xover_low"):
        PipelineConfig(eq_crossover_low=5000.0, eq_crossover_high=100.0)


def test_invalid_demucs_model_raises():
    with pytest.raises(ValidationError, match="separate.model"):
        PipelineConfig(demucs_model="not_a_model")


def test_ms_config_property_shorthand():
    assert MSConfig().side_prop_decrease == pytest.approx(0.6)
