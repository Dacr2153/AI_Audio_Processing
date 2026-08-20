"""Command-line interface for the audio-restoration package.

Usage (after ``pip install -e .``)::

    audio-restore -i input.mp3 -o output.flac
    audio-restore --batch ./songs --output-dir ./restored --output-ext flac

The flag set mirrors the legacy ``restore.py`` command exactly.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from . import __version__
from .config import GENRE_PRESETS, PipelineConfig
from .io.format_handler import FORMAT_REGISTRY, FormatHandler
from .pipeline import RestorationPipeline

# Audio file extensions recognised for --batch scanning.
_AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".aiff",
    ".aif",
    ".mp3",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
    ".mp4",
}


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio-restore",
        description="AI-powered audio restoration tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    # --- Input / Output (single-file mode) ---
    io_group = p.add_argument_group(
        "Input / Output",
        "Use -i/-o for a single file, or --batch for an entire folder.",
    )
    io_group.add_argument(
        "-i",
        "--input",
        default=None,
        metavar="FILE",
        help="Input audio file. Required unless --batch is used.",
    )
    io_group.add_argument(
        "-o",
        "--output",
        default=None,
        metavar="FILE",
        help=(
            "Output file. Format inferred from the extension. "
            "Lossless: .wav .flac .aiff — Lossy: .mp3 .m4a .aac .ogg .opus "
            "(via ffmpeg). Required unless --batch is used."
        ),
    )

    # --- Batch mode ---
    batch_group = p.add_argument_group(
        "Batch processing",
        "Process every audio file in a folder with the same settings.",
    )
    batch_group.add_argument(
        "--batch",
        default=None,
        metavar="FOLDER",
        help="Input folder, scanned recursively.",
    )
    batch_group.add_argument(
        "--output-dir",
        default=None,
        metavar="FOLDER",
        help="Output folder. Defaults to 'restored/' inside the input folder.",
    )
    batch_group.add_argument(
        "--output-ext",
        default=None,
        metavar="EXT",
        help="Output extension (e.g. wav, flac, mp3). Defaults to '.wav'.",
    )
    batch_group.add_argument(
        "--output-suffix",
        default="",
        metavar="SUFFIX",
        help="Suffix appended to each output filename stem (e.g. '_restored').",
    )

    # --- Genre preset ---
    p.add_argument(
        "--preset",
        choices=sorted(GENRE_PRESETS),
        default=None,
        metavar="GENRE",
        help=(
            "Apply a genre-optimised EQ preset. Choices: "
            + ", ".join(sorted(GENRE_PRESETS))
            + ". Individual --bass/--mid/etc. flags override the preset."
        ),
    )

    # --- Denoising ---
    denoise_group = p.add_argument_group("Denoising options")
    denoise_group.add_argument(
        "--denoise-method",
        choices=["auto", "music", "noisereduce", "deepfilternet", "wavelet"],
        default="music",
        metavar="METHOD",
        help=(
            "Denoising algorithm. 'music' (default) = HPSS + Wiener filter. "
            "'noisereduce' = spectral subtraction. 'deepfilternet' = neural "
            "(speech only). 'wavelet' = BayesShrink fallback. "
            "'auto' = music → noisereduce → wavelet."
        ),
    )
    denoise_group.add_argument(
        "--prop-decrease",
        type=float,
        default=0.85,
        metavar="0.0-1.0",
        help="Noise reduction aggressiveness (0.0–1.0). Recommended 0.6–0.8 for music.",
    )
    denoise_group.add_argument(
        "--stationary",
        action="store_true",
        default=False,
        help="For noisereduce: assume stationary noise (tape/vinyl hiss).",
    )
    denoise_group.add_argument(
        "--n-std-thresh",
        type=float,
        default=1.5,
        metavar="THRESH",
        dest="denoise_n_std_thresh",
        help="Stationary-noise detection threshold in std-devs (lower = more aggressive).",
    )
    denoise_group.add_argument(
        "--denoise-passes",
        type=int,
        default=1,
        metavar="N",
        dest="denoise_passes",
        help="Number of sequential denoising passes (2–3 for stubborn static).",
    )

    # --- Source separation ---
    sep_group = p.add_argument_group("Source separation options (Demucs)")
    sep_group.add_argument(
        "--separate",
        action="store_true",
        default=False,
        dest="enable_source_separation",
        help="Separate into stems (vocals, drums, bass, other) with Demucs, then remix.",
    )
    sep_group.add_argument(
        "--demucs-model",
        default="htdemucs_ft",
        choices=["htdemucs_ft", "htdemucs", "mdx_extra"],
        metavar="MODEL",
        help="Demucs model to use. (default: htdemucs_ft)",
    )
    sep_group.add_argument(
        "--stem-denoise",
        action="store_true",
        default=False,
        dest="stem_denoise",
        help="Denoise each Demucs stem independently before remixing.",
    )

    # --- Super-resolution ---
    sr_group = p.add_argument_group("Super-resolution options")
    sr_group.add_argument(
        "--super-resolution",
        action="store_true",
        default=False,
        dest="enable_super_resolution",
        help="Upsample to 48 kHz using AudioSR (if available) or scipy.",
    )
    sr_group.add_argument(
        "--output-sr",
        type=int,
        default=None,
        metavar="HZ",
        help="Resample final output to this sample rate (e.g. 44100).",
    )

    # --- Equalization ---
    eq_group = p.add_argument_group(
        "Equalization options (5-band mastering EQ)",
        description="Pass 0 to any band to disable it.",
    )
    for flag, dest, hz, default_help in [
        ("--bass", "bass_gain_db", "80", "+2.5"),
        ("--mid", "mid_gain_db", "250", "-1.5"),
        ("--presence", "presence_gain_db", "3500", "+2.5"),
        ("--treble", "treble_gain_db", "8000", "+2.0"),
        ("--air", "air_gain_db", "12000", "+3.5"),
    ]:
        eq_group.add_argument(
            flag,
            type=float,
            default=None,
            metavar="DB",
            dest=dest,
            help=f"Gain @ {hz} Hz in dB. Overrides preset. (vinyl default: {default_help})",
        )
    eq_group.add_argument(
        "--no-rumble-filter",
        action="store_false",
        dest="rumble_filter",
        default=True,
        help="Disable the 30 Hz high-pass rumble filter.",
    )

    # --- Output ---
    out_group = p.add_argument_group("Output options")
    out_group.add_argument(
        "--bitrate",
        type=str,
        default=None,
        metavar="BITRATE",
        help="Bitrate for lossy formats (e.g. '320k', 'V0', '256k', quality 0–10).",
    )
    out_group.add_argument(
        "--no-vbr",
        action="store_false",
        dest="output_vbr",
        default=True,
        help="Use CBR instead of VBR for MP3 output.",
    )
    out_group.add_argument(
        "--bit-depth",
        type=int,
        default=24,
        choices=[16, 24, 32],
        metavar="BITS",
        dest="output_bit_depth",
        help="Bit depth for lossless output. (default: 24)",
    )
    out_group.add_argument(
        "--no-normalize",
        action="store_false",
        dest="normalize_output",
        default=True,
        help="Disable peak normalization of the output.",
    )
    out_group.add_argument(
        "--no-limit",
        action="store_false",
        dest="limit_output",
        default=True,
        help="Disable the soft limiter.",
    )
    out_group.add_argument(
        "--limit-threshold",
        type=float,
        default=-0.3,
        metavar="DBFS",
        help="Soft limiter threshold in dBFS. (default: -0.3)",
    )
    out_group.add_argument(
        "--lufs-target",
        type=float,
        default=-14.0,
        metavar="LUFS",
        dest="lufs_target",
        help="Target integrated loudness in LUFS (BS.1770-4 / EBU R128).",
    )
    out_group.add_argument(
        "--no-lufs",
        action="store_const",
        const=None,
        dest="lufs_target",
        help="Disable LUFS loudness normalization.",
    )

    # --- Dehum ---
    dehum_group = p.add_argument_group("Electrical hum removal")
    dehum_group.add_argument(
        "--dehum",
        type=float,
        default=None,
        metavar="HZ",
        dest="dehum_freq",
        help="Hum fundamental to remove: 50 (Europe) or 60 (Americas).",
    )
    dehum_group.add_argument(
        "--dehum-harmonics",
        type=int,
        default=5,
        metavar="N",
        help="Number of harmonics to notch. (default: 5)",
    )
    dehum_group.add_argument(
        "--dehum-q",
        type=float,
        default=35.0,
        metavar="Q",
        help="Notch filter Q. Higher = narrower. (default: 35)",
    )

    # --- Declicker ---
    declick_group = p.add_argument_group("Click / pop / crackle removal")
    declick_group.add_argument(
        "--declick",
        action="store_true",
        default=False,
        dest="enable_declicker",
        help="Enable click/pop removal (recommended for vinyl/cassette).",
    )
    declick_group.add_argument(
        "--declick-threshold",
        type=float,
        default=4.0,
        metavar="N",
        dest="declicker_threshold",
        help="Detection sensitivity (× local RMS).",
    )
    declick_group.add_argument(
        "--declick-lpc-order",
        type=int,
        default=32,
        metavar="N",
        dest="declicker_lpc_order",
        help="LPC interpolation order. (default: 32)",
    )
    declick_group.add_argument(
        "--declick-max-ms",
        type=float,
        default=30.0,
        metavar="MS",
        dest="declicker_max_click_ms",
        help="Maximum short-click length in ms repaired with LPC. (default: 30)",
    )
    declick_group.add_argument(
        "--declick-max-scratch",
        type=float,
        default=200.0,
        metavar="MS",
        dest="declicker_max_scratch_ms",
        help="Maximum scratch length in ms repaired with crossfade. (default: 200)",
    )

    # --- Multiband compression ---
    mb_group = p.add_argument_group("Multiband compression")
    mb_group.add_argument(
        "--multiband",
        action="store_true",
        default=False,
        dest="enable_multiband",
        help="Enable 3-band multiband compressor.",
    )
    for flag, dest, default in [
        ("--mb-low-threshold", "multiband_low_threshold_db", -20.0),
        ("--mb-low-ratio", "multiband_low_ratio", 2.5),
        ("--mb-low-makeup", "multiband_low_makeup_db", 1.5),
        ("--mb-mid-threshold", "multiband_mid_threshold_db", -18.0),
        ("--mb-mid-ratio", "multiband_mid_ratio", 3.0),
        ("--mb-mid-makeup", "multiband_mid_makeup_db", 1.0),
        ("--mb-high-threshold", "multiband_high_threshold_db", -16.0),
        ("--mb-high-ratio", "multiband_high_ratio", 2.0),
        ("--mb-high-makeup", "multiband_high_makeup_db", 0.5),
    ]:
        mb_group.add_argument(
            flag,
            type=float,
            default=default,
            metavar="DB" if "threshold" in flag or "makeup" in flag else "R",
            dest=dest,
            help=f"(default: {default})",
        )

    # --- M/S stereo processing ---
    ms_group = p.add_argument_group("M/S (Mid/Side) stereo processing")
    ms_group.add_argument(
        "--ms",
        action="store_true",
        default=False,
        dest="enable_ms",
        help="Enable M/S stereo processing. No-op for mono.",
    )
    ms_group.add_argument(
        "--ms-side-prop",
        type=float,
        default=0.6,
        metavar="0.0-1.0",
        dest="ms_side_prop_decrease",
        help="Side-channel noise reduction strength.",
    )
    ms_group.add_argument(
        "--ms-no-side-denoise",
        action="store_false",
        default=True,
        dest="ms_side_denoise",
        help="Disable Side channel noise reduction.",
    )
    ms_group.add_argument(
        "--ms-width",
        type=float,
        default=1.0,
        metavar="SCALE",
        dest="ms_side_gain",
        help="Stereo width scale applied to the Side channel.",
    )
    ms_group.add_argument(
        "--ms-mid-presence",
        type=float,
        default=0.0,
        metavar="DB",
        dest="ms_mid_presence_db",
        help="Mid-channel presence EQ boost/cut (0 = off).",
    )
    ms_group.add_argument(
        "--ms-mid-presence-freq",
        type=float,
        default=3500.0,
        metavar="HZ",
        dest="ms_mid_presence_freq",
        help="Mid presence EQ centre frequency in Hz.",
    )

    # --- Wow & flutter ---
    wf_group = p.add_argument_group("Wow & flutter correction")
    wf_group.add_argument(
        "--wow-flutter",
        action="store_true",
        default=False,
        dest="enable_wow_flutter",
        help="Enable pitch correction for vinyl/cassette.",
    )
    wf_group.add_argument(
        "--wf-max-cents",
        type=float,
        default=100.0,
        metavar="CENTS",
        dest="wow_flutter_max_cents",
        help="Maximum pitch correction in cents.",
    )
    wf_group.add_argument(
        "--wf-smoothing",
        type=float,
        default=200.0,
        metavar="MS",
        dest="wow_flutter_smoothing_ms",
        help="Nominal-pitch smoothing window in ms (longer = wow only).",
    )
    wf_group.add_argument(
        "--wf-max-freq",
        type=float,
        default=100.0,
        metavar="HZ",
        dest="wow_flutter_max_freq_hz",
        help="Upper frequency bound of fluctuations to correct in Hz.",
    )

    # --- Reporting ---
    report_group = p.add_argument_group("Reporting options")
    report_group.add_argument(
        "--no-plot",
        action="store_false",
        dest="save_comparison_plot",
        default=True,
        help="Do not save the before/after comparison plot.",
    )
    report_group.add_argument(
        "--no-metrics",
        action="store_false",
        dest="print_metrics",
        default=True,
        help="Do not print the quality metrics report.",
    )
    report_group.add_argument(
        "--metrics-csv",
        default=None,
        metavar="PATH",
        dest="metrics_summary_csv",
        help="Append each file's metrics to a summary CSV at this path.",
    )

    # --- Misc ---
    p.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Processing device for neural models. (default: auto)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )

    return p


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    """Build a typed PipelineConfig from parsed CLI args."""
    preset_defaults = GENRE_PRESETS.get(args.preset, {}) if args.preset else {}

    config = PipelineConfig(
        denoise_method=args.denoise_method,
        denoise_stationary=args.stationary,
        denoise_prop_decrease=args.prop_decrease,
        denoise_n_std_thresh=args.denoise_n_std_thresh,
        denoise_passes=args.denoise_passes,
        enable_source_separation=args.enable_source_separation,
        demucs_model=args.demucs_model,
        demucs_device=args.device,
        stem_denoise=args.stem_denoise,
        enable_super_resolution=args.enable_super_resolution,
        super_resolution_device=args.device,
        dehum_freq=args.dehum_freq,
        dehum_harmonics=args.dehum_harmonics,
        dehum_q=args.dehum_q,
        enable_declicker=args.enable_declicker,
        declicker_threshold=args.declicker_threshold,
        declicker_lpc_order=args.declicker_lpc_order,
        declicker_max_click_ms=args.declicker_max_click_ms,
        declicker_max_scratch_ms=args.declicker_max_scratch_ms,
        enable_multiband=args.enable_multiband,
        multiband_low_threshold_db=args.multiband_low_threshold_db,
        multiband_low_ratio=args.multiband_low_ratio,
        multiband_low_makeup_db=args.multiband_low_makeup_db,
        multiband_mid_threshold_db=args.multiband_mid_threshold_db,
        multiband_mid_ratio=args.multiband_mid_ratio,
        multiband_mid_makeup_db=args.multiband_mid_makeup_db,
        multiband_high_threshold_db=args.multiband_high_threshold_db,
        multiband_high_ratio=args.multiband_high_ratio,
        multiband_high_makeup_db=args.multiband_high_makeup_db,
        enable_ms=args.enable_ms,
        ms_side_denoise=args.ms_side_denoise,
        ms_side_prop_decrease=args.ms_side_prop_decrease,
        ms_side_gain=args.ms_side_gain,
        ms_mid_presence_db=args.ms_mid_presence_db,
        ms_mid_presence_freq=args.ms_mid_presence_freq,
        enable_wow_flutter=args.enable_wow_flutter,
        wow_flutter_max_cents=args.wow_flutter_max_cents,
        wow_flutter_smoothing_ms=args.wow_flutter_smoothing_ms,
        wow_flutter_max_freq_hz=args.wow_flutter_max_freq_hz,
        lufs_target=args.lufs_target,
        output_sample_rate=args.output_sr,
        output_bitrate=args.bitrate,
        output_bit_depth=args.output_bit_depth,
        output_vbr=args.output_vbr,
        normalize_output=args.normalize_output,
        limit_output=args.limit_output,
        limit_threshold_db=args.limit_threshold,
        save_comparison_plot=args.save_comparison_plot,
        print_metrics=args.print_metrics,
        metrics_summary_csv=args.metrics_summary_csv,
        genre=args.preset,
    )

    # EQ bands: CLI flag wins over the genre preset, which wins over vinyl.
    for key in (
        "bass_gain_db",
        "mid_gain_db",
        "presence_gain_db",
        "treble_gain_db",
        "air_gain_db",
    ):
        if getattr(args, key) is not None:
            setattr(config.eq, key, getattr(args, key))
        elif key not in preset_defaults:
            setattr(config.eq, key, GENRE_PRESETS["vinyl"][key])
    return config


def _collect_audio_files(folder: str) -> list[str]:
    """Return a sorted list of audio file paths found recursively in *folder*."""
    results: list[str] = []
    for root, _dirs, files in os.walk(folder):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in _AUDIO_EXTENSIONS:
                results.append(os.path.join(root, fname))
    return results


def _write_batch_summary(csv_path: str, summaries: list[dict]) -> None:
    """Append per-file metric summaries to the CSV at *csv_path*."""
    try:
        from .reporting.batch_report import BatchReport

        report = BatchReport()
        for item in summaries:
            name = item.pop("file", "unknown")
            report.add_file(name, item)
        report.save(
            os.path.dirname(csv_path) or ".", filename=os.path.basename(csv_path)
        )
    except Exception:  # noqa: BLE001 — CSV summary is best-effort
        logging.getLogger("audio-restore").warning(
            "Could not write metrics CSV (missing pandas?): %s", csv_path
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    log = logging.getLogger("audio-restore")

    # ------------------------------------------------------------------
    # Validate: either -i/--input or --batch, but not both.
    # ------------------------------------------------------------------
    if args.batch and args.input:
        print(
            "ERROR: Use either -i/--input (single file) or --batch (folder), not both.",
            file=sys.stderr,
        )
        return 1
    if not args.batch and not args.input:
        print(
            "ERROR: Provide -i FILE (single file) or --batch FOLDER.", file=sys.stderr
        )
        parser.print_usage(sys.stderr)
        return 1

    config = _build_config(args)
    pipeline = RestorationPipeline(config)
    fmt_handler = FormatHandler()

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    if args.batch:
        batch_folder = os.path.abspath(args.batch)
        if not os.path.isdir(batch_folder):
            print(f"ERROR: Batch folder not found: {batch_folder}", file=sys.stderr)
            return 1

        raw_ext = (args.output_ext or "wav").lstrip(".")
        out_ext = "." + raw_ext.lower()
        if out_ext not in FORMAT_REGISTRY:
            print(
                f"ERROR: Unsupported output extension {raw_ext!r}. "
                f"Supported: {', '.join(sorted(FORMAT_REGISTRY))}",
                file=sys.stderr,
            )
            return 1

        out_dir = (
            os.path.abspath(args.output_dir)
            if args.output_dir
            else os.path.join(batch_folder, "restored")
        )
        os.makedirs(out_dir, exist_ok=True)

        input_files = _collect_audio_files(batch_folder)
        if not input_files:
            print(f"No audio files found in: {batch_folder}", file=sys.stderr)
            return 1

        log.info("Batch mode: %d files found in %s", len(input_files), batch_folder)
        log.info("Output folder: %s  |  Extension: %s", out_dir, out_ext)

        summaries: list[dict] = []
        ok, failed = 0, []
        for idx, in_path in enumerate(input_files, 1):
            stem = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(out_dir, stem + args.output_suffix + out_ext)
            log.info(
                "[%d/%d] %s  →  %s",
                idx,
                len(input_files),
                os.path.basename(in_path),
                os.path.basename(out_path),
            )
            try:
                report = pipeline.restore(in_path, out_path)
                summaries.append({"file": out_path, **report})
                ok += 1
            except Exception as exc:
                log.error("  FAILED: %s", exc, exc_info=args.verbose)
                failed.append((in_path, str(exc)))

        if summaries and args.metrics_summary_csv:
            _write_batch_summary(args.metrics_summary_csv, summaries)

        print(f"\nBatch complete: {ok}/{len(input_files)} succeeded.")
        if failed:
            print("Failed files:")
            for path, reason in failed:
                print(f"  {path}: {reason}")
            return 2
        return 0

    # ------------------------------------------------------------------
    # Single-file mode
    # ------------------------------------------------------------------
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1
    if not args.output:
        print("ERROR: -o/--output is required in single-file mode.", file=sys.stderr)
        return 1

    if ("." + fmt_handler.detect_format(args.output)) not in FORMAT_REGISTRY:
        print(
            f"ERROR: Unsupported output format for {args.output!r}. "
            f"Supported: {', '.join(sorted(FORMAT_REGISTRY))}",
            file=sys.stderr,
        )
        return 1

    try:
        report = pipeline.restore(args.input, args.output)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception:
        log.exception("Pipeline failed")
        return 2

    if args.metrics_summary_csv:
        _write_batch_summary(
            args.metrics_summary_csv, [{"file": args.output, **report}]
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
