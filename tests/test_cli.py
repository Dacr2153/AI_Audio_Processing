"""Tests for the CLI (parser construction + main entry points)."""

from __future__ import annotations

from pathlib import Path

from audio_restoration.cli import _build_config, build_argument_parser, main
from audio_restoration.config import PipelineConfig


def test_parser_covers_legacy_flags():
    p = build_argument_parser()
    args = p.parse_args(["-i", "in.mp3", "-o", "out.flac", "--preset", "jazz"])
    assert args.input == "in.mp3"
    assert args.output == "out.flac"
    assert args.preset == "jazz"


def test_build_config_maps_flags():
    p = build_argument_parser()
    args = p.parse_args(
        [
            "-i",
            "in.mp3",
            "-o",
            "out.flac",
            "--denoise-method",
            "auto",
            "--bass",
            "4.0",
            "--declick",
            "--ms",
            "--lufs-target",
            "-16",
            "--no-plot",
        ]
    )
    cfg: PipelineConfig = _build_config(args)
    assert cfg.denoise.method == "auto"
    assert cfg.eq.bass_gain_db == 4.0
    assert cfg.declick.enabled is True
    assert cfg.ms.enabled is True
    assert cfg.loudness.target_lufs == -16.0
    assert cfg.report.save_comparison_plot is False


def test_build_config_preset_plus_cli_override():
    p = build_argument_parser()
    args = p.parse_args(
        ["-i", "in.mp3", "-o", "out.mp3", "--preset", "hiphop", "--mid", "0"]
    )
    cfg = _build_config(args)
    # hiphop preset bass +5.0 left intact, explicit --mid wins.
    assert cfg.eq.bass_gain_db == 5.0
    assert cfg.eq.mid_gain_db == 0.0


def test_main_single_file_roundtrip(tmp_path, wav_file):

    out = str(tmp_path / "out.wav")
    rc = main(
        [
            "-i",
            wav_file,
            "-o",
            out,
            "--denoise-method",
            "wavelet",
            "--no-plot",
            "--no-metrics",
            "--no-lufs",
        ]
    )
    assert rc == 0
    import os

    import soundfile as sf

    assert os.path.isfile(out)
    audio, _sr = sf.read(out)
    assert audio.ndim == 2  # stereo preserved


def test_main_single_file_metrics_csv(tmp_path, wav_file):
    out = str(tmp_path / "out.wav")
    csv_path = str(tmp_path / "summary.csv")
    rc = main(
        [
            "-i",
            wav_file,
            "-o",
            out,
            "--denoise-method",
            "wavelet",
            "--no-plot",
            "--no-metrics",
            "--no-lufs",
            "--metrics-csv",
            csv_path,
        ]
    )
    assert rc == 0
    assert Path(csv_path).is_file()
    content = Path(csv_path).read_text()
    assert "out.wav" in content
    assert "restored_rms_db" in content


def test_main_batch_metrics_csv(tmp_path, wav_file):
    import shutil

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    shutil.copy(wav_file, batch_dir / "track1.wav")
    shutil.copy(wav_file, batch_dir / "track2.wav")

    out_dir = tmp_path / "out"
    csv_path = str(tmp_path / "batch.csv")
    rc = main(
        [
            "--batch",
            str(batch_dir),
            "--output-dir",
            str(out_dir),
            "--denoise-method",
            "wavelet",
            "--no-plot",
            "--no-metrics",
            "--no-lufs",
            "--metrics-csv",
            csv_path,
        ]
    )
    assert rc == 0
    assert (out_dir / "track1.wav").is_file()
    assert (out_dir / "track2.wav").is_file()
    assert Path(csv_path).is_file()
    content = Path(csv_path).read_text()
    assert content.count("track") >= 2


def test_main_missing_input_returns_1(tmp_path):
    rc = main(["-i", str(tmp_path / "nope.wav"), "-o", str(tmp_path / "x.wav")])
    assert rc == 1


def test_main_both_batch_and_input_returns_1(tmp_path, wav_file):
    rc = main(["-i", wav_file, "--batch", str(tmp_path)])
    assert rc == 1


def test_main_neither_input_nor_batch_returns_1():
    rc = main(["-o", "x.wav"])
    assert rc == 1


def test_main_batch_mode(tmp_path, wav_file):
    import shutil

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    shutil.copy(wav_file, batch_dir / "track.wav")

    rc = main(
        [
            "--batch",
            str(batch_dir),
            "--denoise-method",
            "wavelet",
            "--no-plot",
            "--no-metrics",
            "--no-lufs",
            "--output-ext",
            "flac",
        ]
    )
    assert rc == 0
    assert (batch_dir / "restored" / "track.flac").is_file()
