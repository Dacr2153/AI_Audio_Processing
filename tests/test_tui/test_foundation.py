"""Tests for the TUI foundation: i18n, themes, state, serde and app shell."""

from __future__ import annotations

import pytest

from audio_restoration.config import PipelineConfig
from audio_restoration.tui import i18n
from audio_restoration.tui.config_serde import config_to_flat, flat_to_config
from audio_restoration.tui.state import TuiState

# ---------------------------------------------------------------------------
# i18n
# ---------------------------------------------------------------------------


def test_i18n_defaults_to_english():
    i18n.set_language("en")
    assert i18n.t("nav.home") == "Home"
    assert i18n.t("nav.batch") == "Batch"


def test_i18n_spanish_translation():
    i18n.set_language("es")
    assert i18n.t("nav.home") == "Inicio"
    assert i18n.t("nav.batch") == "Lote"


def test_i18n_fallback_to_key_when_missing():
    i18n.set_language("en")
    assert i18n.t("no.such.key") == "no.such.key"


def test_i18n_interpolation():
    assert i18n.t("ok.saved") == "Saved"


def test_i18n_toggle_cycles():
    i18n.set_language("en")
    assert i18n.toggle_language() == "es"
    assert i18n.toggle_language() == "en"


def test_i18n_set_language_rejects_unknown():
    with pytest.raises(ValueError):
        i18n.set_language("fr")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# State / profiles
# ---------------------------------------------------------------------------


def test_state_save_load_delete_profile(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    state = TuiState(config=PipelineConfig(denoise_method="music"))
    state.save_profile("vinyl")
    assert state.list_profiles() == ["vinyl"]
    state.delete_profile("vinyl")
    assert state.list_profiles() == []


def test_state_apply_profile_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    original = PipelineConfig(denoise_method="music", bass_gain_db=3.5)
    state = TuiState(config=original)
    state.save_profile("jazz")
    fresh = TuiState()
    fresh.apply_profile("jazz")
    assert fresh.config.denoise.method == "music"
    assert fresh.config.eq.bass_gain_db == 3.5


def test_state_history(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    state = TuiState()
    state.add_history_entry({"input": "a.wav"})
    state.add_history_entry({"input": "b.wav"})
    history = state.load_history()
    assert len(history) == 2
    assert history[0]["input"] == "b.wav"
    state.clear_history()
    assert state.load_history() == []


def test_state_listeners():
    state = TuiState()
    calls = []
    state.add_listener("refresh", lambda: calls.append(1))
    state.emit("refresh")
    assert calls == [1]


# ---------------------------------------------------------------------------
# config serde
# ---------------------------------------------------------------------------


def test_config_roundtrip_flat():
    original = PipelineConfig(
        enable_wow_flutter=True,
        wow_flutter_max_cents=40.0,
        enable_source_separation=True,
        demucs_model="mdx_extra",
        enable_super_resolution=True,
        sr_target_sr=48_000,
        multiband_low_threshold_db=-15.0,  # legacy key
    )
    flat = config_to_flat(original)
    rebuilt = flat_to_config(flat)
    assert rebuilt.separate.model == "mdx_extra"
    assert rebuilt.separate.enabled is True
    assert rebuilt.wow_flutter.max_cents == 40.0
    assert rebuilt.sr.target_sr == 48_000
    assert rebuilt.multiband.low_threshold_db == -15.0


def test_flat_to_config_ignores_unknown_keys():
    rebuilt = flat_to_config({"bogus_key": 123, "denoise_method": "wavelet"})
    assert rebuilt.denoise.method == "wavelet"