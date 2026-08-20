"""Tests for BatchReport CSV export and device resolution."""

from __future__ import annotations

import os

import pytest

from audio_restoration.neural.devices import resolve_device, seed_all
from audio_restoration.reporting.batch_report import BatchReport


def test_batch_report_empty_returns_message():
    assert BatchReport().summary_text() == "No files recorded."


def test_batch_report_save_requires_rows(tmp_path):
    with pytest.raises(ValueError, match="No files recorded"):
        BatchReport().save(str(tmp_path))


def test_batch_report_csv(tmp_path, mono_signal, sample_rate):
    report = BatchReport()
    metrics = report.measure(mono_signal, mono_signal * 0.9, sample_rate)
    report.add_file("song.wav", metrics)

    path = report.save(str(tmp_path))
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    assert "file" in content
    assert "snr_db" in content
    assert "song.wav" in content


def test_batch_report_summary_text_renders(mono_signal, sample_rate):
    report = BatchReport()
    report.add_file("a.wav", report.measure(mono_signal, mono_signal, sample_rate))
    text = report.summary_text()
    assert "a.wav" in text
    assert "Mean" in text


def test_resolve_device_prefers_cpu_without_torch(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "torch":
            raise ImportError("no torch")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_unavailable(monkeypatch):
    # Simulate torch present but no CUDA.
    import types

    torch_mod = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", torch_mod)
    assert resolve_device("cuda") == "cpu"
    assert resolve_device("auto") == "cpu"


def test_seed_all_runs():
    seed_all(7)  # must not raise
