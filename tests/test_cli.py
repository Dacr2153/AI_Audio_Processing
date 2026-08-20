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


def test_main_batch_folder_not_found(tmp_path):
    rc = main(["--batch", str(tmp_path / "nope")])
    assert rc == 1


def test_main_batch_unsupported_output_ext(tmp_path, wav_file):
    import shutil

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    shutil.copy(wav_file, batch_dir / "track.wav")
    rc = main(["--batch", str(batch_dir), "--output-ext", "xyz"])
    assert rc == 1


def test_main_batch_no_audio_files(tmp_path):
    batch_dir = tmp_path / "empty"
    batch_dir.mkdir()
    rc = main(["--batch", str(batch_dir)])
    assert rc == 1


def test_main_batch_suffix_with_extension(tmp_path, wav_file):
    import shutil

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    shutil.copy(wav_file, batch_dir / "track.wav")
    out_dir = tmp_path / "out"
    rc = main(
        [
            "--batch",
            str(batch_dir),
            "--output-dir",
            str(out_dir),
            "--output-suffix",
            "_restored",
            "--denoise-method",
            "wavelet",
            "--no-plot",
            "--no-metrics",
            "--no-lufs",
        ]
    )
    assert rc == 0
    assert (out_dir / "track_restored.wav").is_file()


def test_main_single_file_output_required(tmp_path, wav_file):
    rc = main(["-i", wav_file])
    assert rc == 1


def test_main_single_file_unsupported_output_format(tmp_path, wav_file):
    out = str(tmp_path / "out.xyz")
    rc = main(["-i", wav_file, "-o", out])
    assert rc == 1


def test_main_batch_parallel_workers(tmp_path, wav_file):
    import shutil

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    for i in range(3):
        shutil.copy(wav_file, batch_dir / f"track{i}.wav")

    rc = main(
        [
            "--batch",
            str(batch_dir),
            "--workers",
            "3",
            "--denoise-method",
            "wavelet",
            "--no-plot",
            "--no-metrics",
            "--no-lufs",
        ]
    )
    assert rc == 0
    assert (batch_dir / "restored" / "track0.wav").is_file()
    assert (batch_dir / "restored" / "track2.wav").is_file()


def test_main_batch_parallel_failure(tmp_path, wav_file):
    import shutil

    batch_dir = tmp_path / "batch"
    batch_dir.mkdir()
    shutil.copy(wav_file, batch_dir / "good.wav")
    (batch_dir / "empty.WAV").write_bytes(b"")
    (batch_dir / "corrupt.mp3").write_bytes(b"garbage")

    rc = main(
        [
            "--batch",
            str(batch_dir),
            "--workers",
            "2",
            "--denoise-method",
            "wavelet",
            "--no-plot",
            "--no-metrics",
            "--no-lufs",
        ]
    )
    assert rc == 2  # some files fail, some succeed
