#!/usr/bin/env python3
"""
restore.py — CLI entry point for the Audio Restoration Pipeline.

Restores degraded audio recordings using a chain of AI and DSP algorithms:
  1. Load & normalize   (WAV, MP3, FLAC, OGG, M4A, AIFF, OPUS, WMA, …)
  2. Denoise            (DeepFilterNet / noisereduce / wavelet)
  3. Source separation  — optional (Demucs htdemucs_ft)
  4. Super-resolution   — optional (AudioSR / scipy resampler → 48 kHz)
  5. 3-band equalization (Butterworth zero-phase)
  6. Peak normalize + soft limit
  7. Encode output      (WAV, FLAC, AIFF, MP3, M4A, OGG, OPUS, …)
  8. Quality metrics report + comparison plot

OUTPUT FORMATS (format detected from extension):
  Lossless: .wav (PCM 24-bit)  .flac (FLAC 24-bit)  .aiff (AIFF 24-bit)
  Lossy:    .mp3 (VBR V0 / libmp3lame)  .m4a / .aac (AAC 256k)
            .ogg (Vorbis q8)  .opus (192k)

EXAMPLES:

  # Basic restoration — WAV output (lossless, PCM 24-bit)
  python restore.py --input old_song.wav --output restored.wav

  # Restore MP3 → MP3 at highest VBR quality (libmp3lame V0 ≈ 245 kbps)
  python restore.py -i old_song.mp3 -o restored.mp3

  # Restore MP3 → 320kbps CBR MP3
  python restore.py -i song.mp3 -o song_320.mp3 --bitrate 320k --no-vbr

  # Restore to FLAC (lossless)
  python restore.py -i recording.wav -o restored.flac

  # Restore to M4A/AAC (streaming-friendly)
  python restore.py -i song.mp3 -o song.m4a --bitrate 256k

  # Restore with bass boost and treble enhancement
  python restore.py -i old_vinyl.wav -o out.wav --bass 4 --treble 3

  # Full pipeline: denoise + separate sources + super-resolution + EQ
  python restore.py -i recording.wav -o out.wav --separate --super-resolution

  # Aggressive noise removal for very noisy old cassette tapes
  python restore.py -i tape.wav -o clean.wav --denoise-method noisereduce --prop-decrease 0.95 --stationary

  # Process with mid-frequency presence boost (good for muffled vocals)
  python restore.py -i muffled.wav -o clear.wav --mid 2 --treble 2

ACTIVATE VIRTUAL ENVIRONMENT FIRST:
  source venv_audio/bin/activate
  python restore.py --input song.mp3 --output restored.flac
"""

import argparse
import logging
import os
import sys

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from restoration_pipeline import PipelineConfig, RestorationPipeline, GENRE_PRESETS
from format_handler import FormatHandler, FORMAT_REGISTRY


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="restore",
        description="AI-powered audio restoration tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # --- Input / Output (single-file mode) ---
    io_group = p.add_argument_group(
        "Input / Output",
        "Use -i/-o for a single file, or --batch for an entire folder.",
    )
    io_group.add_argument(
        "-i", "--input",
        default=None,
        metavar="FILE",
        help="Input audio file (WAV, MP3, FLAC, OGG, M4A, etc.).  Required unless --batch is used.",
    )
    io_group.add_argument(
        "-o", "--output",
        default=None,
        metavar="FILE",
        help=(
            "Output file. Format is inferred from the extension. "
            "Lossless: .wav .flac .aiff  —  "
            "Lossy: .mp3 .m4a .aac .ogg .opus  (encoded via ffmpeg). "
            "Required unless --batch is used."
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
        help="Input folder. All audio files inside are processed recursively.",
    )
    batch_group.add_argument(
        "--output-dir",
        default=None,
        metavar="FOLDER",
        help=(
            "Output folder for --batch mode. "
            "Defaults to a 'restored/' sub-folder inside the input folder."
        ),
    )
    batch_group.add_argument(
        "--output-ext",
        default=None,
        metavar="EXT",
        help=(
            "Output file extension for --batch mode (e.g. wav, flac, mp3). "
            "Defaults to '.wav'. The dot is optional."
        ),
    )
    batch_group.add_argument(
        "--output-suffix",
        default="",
        metavar="SUFFIX",
        help="Suffix appended to each output filename stem (e.g. '_restored'). (default: '')",
    )

    # --- Genre preset ---
    p.add_argument(
        "--preset",
        choices=sorted(GENRE_PRESETS.keys()),
        default=None,
        metavar="GENRE",
        help=(
            "Apply a genre-optimised EQ preset. "
            "Choices: " + ", ".join(sorted(GENRE_PRESETS.keys())) + ". "
            "Individual --bass/--mid/etc. flags override the preset per-band."
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
            "Denoising algorithm. "
            "'music' (default) = HPSS + Wiener filter, preserves instruments and vocals. "
            "'noisereduce' = spectral subtraction (good for speech/stationary hiss). "
            "'deepfilternet' = neural model (speech only, NOT for music). "
            "'wavelet' = BayesShrink (fast fallback). "
            "'auto' = music → noisereduce → wavelet."
        ),
    )
    denoise_group.add_argument(
        "--prop-decrease",
        type=float,
        default=0.85,
        metavar="0.0-1.0",
        help=(
            "Noise reduction aggressiveness. "
            "0.0 = no reduction, 1.0 = full suppression. "
            "Recommended for music (vinyl/static): 0.6–0.8 (default: 0.7). "
            "For speech with noisereduce: 0.7–0.9."
        ),
    )
    denoise_group.add_argument(
        "--stationary",
        action="store_true",
        default=False,
        help=(
            "For noisereduce: assume constant/stationary noise (tape hiss, vinyl hiss). "
            "Disable for variable background noise. (default: off)"
        ),
    )
    denoise_group.add_argument(
        "--n-std-thresh",
        type=float,
        default=1.5,
        metavar="THRESH",
        dest="denoise_n_std_thresh",
        help=(
            "For noisereduce --stationary: std-dev threshold for noise detection. "
            "Lower = more aggressive. 1.5 = conservative (default) | "
            "1.0 = balanced | 0.5 = aggressive (eliminates stubborn static)."
        ),
    )
    denoise_group.add_argument(
        "--denoise-passes",
        type=int,
        default=1,
        metavar="N",
        dest="denoise_passes",
        help=(
            "Number of sequential denoising passes. Each pass cleans residual "
            "noise left by the previous. 1 = standard (default). "
            "2–3 = for stubborn background static."
        ),
    )

    # --- Source separation ---
    sep_group = p.add_argument_group("Source separation options (Demucs)")
    sep_group.add_argument(
        "--separate",
        action="store_true",
        default=False,
        dest="enable_source_separation",
        help=(
            "Separate audio into stems (vocals, drums, bass, other) using Demucs, "
            "then reconstruct. Improves per-stem processing quality. "
            "WARNING: slow — htdemucs_ft takes ~4x real-time on CPU."
        ),
    )
    sep_group.add_argument(
        "--demucs-model",
        default="htdemucs_ft",
        choices=["htdemucs_ft", "htdemucs", "mdx_extra"],
        metavar="MODEL",
        help="Demucs model to use. (default: htdemucs_ft — best quality)",
    )
    sep_group.add_argument(
        "--stem-denoise",
        action="store_true",
        default=False,
        dest="stem_denoise",
        help=(
            "Denoise each Demucs stem (vocals, drums, bass, other) independently "
            "before remixing. Gives the denoiser a simpler signal per instrument. "
            "Requires --separate. (default: off)"
        ),
    )

    # --- Super-resolution ---
    sr_group = p.add_argument_group("Super-resolution options")
    sr_group.add_argument(
        "--super-resolution",
        action="store_true",
        default=False,
        dest="enable_super_resolution",
        help=(
            "Upsample to 48 kHz using AudioSR (if available) or scipy. "
            "AudioSR recovers high-frequency content lost in MP3/tape. "
            "scipy resampler performs clean sinc interpolation only."
        ),
    )
    sr_group.add_argument(
        "--output-sr",
        type=int,
        default=None,
        metavar="HZ",
        help="Resample final output to this sample rate (e.g. 44100). Default: keep pipeline SR.",
    )

    # --- Equalization ---
    eq_group = p.add_argument_group(
        "Equalization options (5-band mastering EQ)",
        description=(
            "Vinyl-restoration defaults are applied automatically. "
            "Pass 0 to any band to disable it."
        ),
    )
    eq_group.add_argument(
        "--bass",
        type=float,
        default=None,
        metavar="DB",
        dest="bass_gain_db",
        help="Low-shelf gain @ 80 Hz in dB. Overrides preset. (vinyl default: +2.5)",
    )
    eq_group.add_argument(
        "--mid",
        type=float,
        default=None,
        metavar="DB",
        dest="mid_gain_db",
        help="Peaking gain @ 250 Hz in dB. Negative = mud cut. Overrides preset. (vinyl default: -1.5)",
    )
    eq_group.add_argument(
        "--presence",
        type=float,
        default=None,
        metavar="DB",
        dest="presence_gain_db",
        help="Peaking gain @ 3500 Hz in dB. Vocal/attack clarity. Overrides preset. (vinyl default: +2.5)",
    )
    eq_group.add_argument(
        "--treble",
        type=float,
        default=None,
        metavar="DB",
        dest="treble_gain_db",
        help="High-shelf gain @ 8000 Hz in dB. Overrides preset. (vinyl default: +2.0)",
    )
    eq_group.add_argument(
        "--air",
        type=float,
        default=None,
        metavar="DB",
        dest="air_gain_db",
        help="High-shelf gain @ 12000 Hz in dB. Sparkle/openness. Overrides preset. (vinyl default: +3.5)",
    )
    eq_group.add_argument(
        "--no-rumble-filter",
        action="store_false",
        dest="rumble_filter",
        default=True,
        help="Disable the 30 Hz high-pass rumble filter.",
    )
    eq_group.add_argument(
        "--crossover-low",
        type=float,
        default=250.0,
        metavar="HZ",
        help="(Legacy — no longer used.) Bass/mid crossover in Hz.",
    )
    eq_group.add_argument(
        "--crossover-high",
        type=float,
        default=4000.0,
        metavar="HZ",
        help="(Legacy — no longer used.) Mid/treble crossover in Hz.",
    )

    # --- Output ---
    out_group = p.add_argument_group("Output options")
    out_group.add_argument(
        "--bitrate",
        type=str,
        default=None,
        metavar="BITRATE",
        help=(
            "Bitrate for lossy output formats. "
            "MP3: '320k' (CBR) or 'V0' (VBR, default). "
            "M4A/AAC: '256k', '192k'. "
            "OGG: quality 0–10 (default: 8). "
            "OPUS: '192k'. "
            "Ignored for lossless (.wav, .flac, .aiff)."
        ),
    )
    out_group.add_argument(
        "--no-vbr",
        action="store_false",
        dest="output_vbr",
        default=True,
        help="Use CBR instead of VBR for MP3 output. (default: VBR)",
    )
    out_group.add_argument(
        "--bit-depth",
        type=int,
        default=24,
        choices=[16, 24, 32],
        metavar="BITS",
        dest="output_bit_depth",
        help="Bit depth for lossless output (.wav, .flac, .aiff). (default: 24)",
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
        help="Disable soft limiter.",
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
        help=(
            "Target integrated loudness in LUFS (BS.1770-4 / EBU R128). "
            "-14 = Spotify/YouTube/Apple Music (default). "
            "-23 = EBU R128 broadcast. "
            "Pass --no-lufs to disable."
        ),
    )
    out_group.add_argument(
        "--no-lufs",
        action="store_const",
        const=None,
        dest="lufs_target",
        help="Disable LUFS loudness normalization (use peak normalize only).",
    )

    # --- Dehum ---
    dehum_group = p.add_argument_group(
        "Electrical hum removal",
        description="Removes 50/60 Hz power-line hum and harmonics via IIR notch filters.",
    )
    dehum_group.add_argument(
        "--dehum",
        type=float,
        default=None,
        metavar="HZ",
        dest="dehum_freq",
        help=(
            "Hum fundamental frequency to remove. "
            "50 = Europe/Asia/Africa/Australia. "
            "60 = Americas/Japan. (default: off)"
        ),
    )
    dehum_group.add_argument(
        "--dehum-harmonics",
        type=int,
        default=5,
        metavar="N",
        help="Number of harmonics to notch (including fundamental). (default: 5)",
    )
    dehum_group.add_argument(
        "--dehum-q",
        type=float,
        default=35.0,
        metavar="Q",
        help="Notch filter Q. Higher = narrower notch. (default: 35)",
    )

    # --- Declicker ---
    declick_group = p.add_argument_group(
        "Click / pop / crackle removal",
        description=(
            "Detects and repairs vinyl clicks, pops, and crackles using "
            "LPC (Linear Predictive Coding) interpolation. "
            "Operates in the time domain — does not affect continuous noise."
        ),
    )
    declick_group.add_argument(
        "--declick",
        action="store_true",
        default=False,
        dest="enable_declicker",
        help="Enable click/pop removal. Recommended for vinyl and cassette.",
    )
    declick_group.add_argument(
        "--declick-threshold",
        type=float,
        default=4.0,
        metavar="N",
        dest="declicker_threshold",
        help=(
            "Click detection sensitivity (× local RMS). "
            "Lower = more aggressive. 4–6 for vinyl, 3–4 for heavy crackle. (default: 4)"
        ),
    )
    declick_group.add_argument(
        "--declick-lpc-order",
        type=int,
        default=32,
        metavar="N",
        dest="declicker_lpc_order",
        help="LPC interpolation order. Higher = better for complex timbres. (default: 32)",
    )
    declick_group.add_argument(
        "--declick-max-ms",
        type=float,
        default=30.0,
        metavar="MS",
        dest="declicker_max_click_ms",
        help="Maximum short-click length in ms repaired with LPC. Longer events use scratch repair. (default: 30)",
    )
    declick_group.add_argument(
        "--declick-max-scratch",
        type=float,
        default=200.0,
        metavar="MS",
        dest="declicker_max_scratch_ms",
        help=(
            "Maximum needle-scratch length in ms repaired with bidirectional crossfade. "
            "Scratches between --declick-max-ms and this limit are repaired; "
            "longer events are left untouched. (default: 200)"
        ),
    )

    # --- Multiband compression ---
    mb_group = p.add_argument_group(
        "Multiband compression",
        description=(
            "3-band Linkwitz-Riley compressor applied after EQ. "
            "Each band (low/mid/high) has independent threshold, ratio, and make-up gain."
        ),
    )
    mb_group.add_argument(
        "--multiband",
        action="store_true",
        default=False,
        dest="enable_multiband",
        help="Enable 3-band multiband compressor. (default: off)",
    )
    mb_group.add_argument(
        "--mb-low-threshold",
        type=float,
        default=-20.0,
        metavar="DB",
        dest="multiband_low_threshold_db",
        help="Low-band compressor threshold in dBFS. (default: -20)",
    )
    mb_group.add_argument(
        "--mb-low-ratio",
        type=float,
        default=2.5,
        metavar="R",
        dest="multiband_low_ratio",
        help="Low-band compression ratio. (default: 2.5)",
    )
    mb_group.add_argument(
        "--mb-low-makeup",
        type=float,
        default=1.5,
        metavar="DB",
        dest="multiband_low_makeup_db",
        help="Low-band make-up gain in dB. (default: 1.5)",
    )
    mb_group.add_argument(
        "--mb-mid-threshold",
        type=float,
        default=-18.0,
        metavar="DB",
        dest="multiband_mid_threshold_db",
        help="Mid-band compressor threshold in dBFS. (default: -18)",
    )
    mb_group.add_argument(
        "--mb-mid-ratio",
        type=float,
        default=3.0,
        metavar="R",
        dest="multiband_mid_ratio",
        help="Mid-band compression ratio. (default: 3.0)",
    )
    mb_group.add_argument(
        "--mb-mid-makeup",
        type=float,
        default=1.0,
        metavar="DB",
        dest="multiband_mid_makeup_db",
        help="Mid-band make-up gain in dB. (default: 1.0)",
    )
    mb_group.add_argument(
        "--mb-high-threshold",
        type=float,
        default=-16.0,
        metavar="DB",
        dest="multiband_high_threshold_db",
        help="High-band compressor threshold in dBFS. (default: -16)",
    )
    mb_group.add_argument(
        "--mb-high-ratio",
        type=float,
        default=2.0,
        metavar="R",
        dest="multiband_high_ratio",
        help="High-band compression ratio. (default: 2.0)",
    )
    mb_group.add_argument(
        "--mb-high-makeup",
        type=float,
        default=0.5,
        metavar="DB",
        dest="multiband_high_makeup_db",
        help="High-band make-up gain in dB. (default: 0.5)",
    )

    # --- M/S stereo processing ---
    ms_group = p.add_argument_group(
        "M/S (Mid/Side) stereo processing",
        description=(
            "Encodes stereo to Mid/Side, processes each independently, then decodes. "
            "Effective for reducing incoherent noise in the stereo image and controlling width."
        ),
    )
    ms_group.add_argument(
        "--ms",
        action="store_true",
        default=False,
        dest="enable_ms",
        help="Enable M/S stereo processing. No-op for mono. (default: off)",
    )
    ms_group.add_argument(
        "--ms-side-prop",
        type=float,
        default=0.6,
        metavar="0.0-1.0",
        dest="ms_side_prop_decrease",
        help="Noise reduction strength on the Side channel (0–1). (default: 0.6)",
    )
    ms_group.add_argument(
        "--ms-no-side-denoise",
        action="store_false",
        default=True,
        dest="ms_side_denoise",
        help="Disable Side channel noise reduction in M/S mode.",
    )
    ms_group.add_argument(
        "--ms-width",
        type=float,
        default=1.0,
        metavar="SCALE",
        dest="ms_side_gain",
        help="Stereo width scale applied to the Side channel. < 1 = narrower, > 1 = wider. (default: 1.0)",
    )
    ms_group.add_argument(
        "--ms-mid-presence",
        type=float,
        default=0.0,
        metavar="DB",
        dest="ms_mid_presence_db",
        help="Presence EQ boost/cut on Mid channel in dB (0 = off). (default: 0.0)",
    )
    ms_group.add_argument(
        "--ms-mid-presence-freq",
        type=float,
        default=3500.0,
        metavar="HZ",
        dest="ms_mid_presence_freq",
        help="Mid presence EQ centre frequency in Hz. (default: 3500)",
    )

    # --- Wow & flutter ---
    wf_group = p.add_argument_group("Wow & flutter correction (Phase 1.3)")
    wf_group.add_argument(
        "--wow-flutter",
        action="store_true",
        default=False,
        dest="enable_wow_flutter",
        help="Enable wow & flutter pitch correction for vinyl / cassette recordings.",
    )
    wf_group.add_argument(
        "--wf-max-cents",
        type=float,
        default=100.0,
        metavar="CENTS",
        dest="wow_flutter_max_cents",
        help="Maximum pitch correction in cents (100 = 1 semitone). (default: 100)",
    )
    wf_group.add_argument(
        "--wf-smoothing",
        type=float,
        default=200.0,
        metavar="MS",
        dest="wow_flutter_smoothing_ms",
        help="Smoothing window for nominal pitch in ms. Longer = slow wow only. (default: 200)",
    )
    wf_group.add_argument(
        "--wf-max-freq",
        type=float,
        default=100.0,
        metavar="HZ",
        dest="wow_flutter_max_freq_hz",
        help="Upper frequency bound of fluctuations to correct in Hz. (default: 100)",
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


# Audio file extensions recognised for --batch scanning
_AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".aac",
    ".ogg", ".opus", ".wma", ".mp4",
}


def _build_config(args) -> "PipelineConfig":
    """Build a PipelineConfig from parsed CLI args."""
    preset_defaults = GENRE_PRESETS.get(args.preset, GENRE_PRESETS["vinyl"]) if args.preset else {}
    vinyl = GENRE_PRESETS["vinyl"]

    def _band(arg_val, key):
        if arg_val is not None:
            return arg_val
        return preset_defaults.get(key, vinyl[key])

    return PipelineConfig(
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
        bass_gain_db=_band(args.bass_gain_db,     "bass_gain_db"),
        mid_gain_db=_band(args.mid_gain_db,       "mid_gain_db"),
        presence_gain_db=_band(args.presence_gain_db, "presence_gain_db"),
        treble_gain_db=_band(args.treble_gain_db, "treble_gain_db"),
        air_gain_db=_band(args.air_gain_db,       "air_gain_db"),
        eq_rumble_filter=args.rumble_filter,
        eq_crossover_low=args.crossover_low,
        eq_crossover_high=args.crossover_high,
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
    )


def _collect_audio_files(folder: str) -> list:
    """Return a sorted list of audio file paths found recursively in *folder*."""
    results = []
    for root, _dirs, files in os.walk(folder):
        for fname in sorted(files):
            if os.path.splitext(fname)[1].lower() in _AUDIO_EXTENSIONS:
                results.append(os.path.join(root, fname))
    return results


def main():
    parser = build_argument_parser()
    args = parser.parse_args()

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    log = logging.getLogger("restore")

    # ------------------------------------------------------------------
    # Validate: must have either -i/--input or --batch (but not both)
    # ------------------------------------------------------------------
    if args.batch and args.input:
        print("ERROR: Use either -i/--input (single file) or --batch (folder), not both.", file=sys.stderr)
        sys.exit(1)
    if not args.batch and not args.input:
        print("ERROR: Provide -i FILE (single file) or --batch FOLDER.", file=sys.stderr)
        parser.print_usage(sys.stderr)
        sys.exit(1)

    config = _build_config(args)
    pipeline = RestorationPipeline(config)

    # ------------------------------------------------------------------
    # Batch mode
    # ------------------------------------------------------------------
    if args.batch:
        batch_folder = os.path.abspath(args.batch)
        if not os.path.isdir(batch_folder):
            print(f"ERROR: Batch folder not found: {batch_folder}", file=sys.stderr)
            sys.exit(1)

        # Determine output extension
        raw_ext = (args.output_ext or "wav").lstrip(".")
        out_ext = "." + raw_ext.lower()

        # Validate the output extension
        fmt_handler = FormatHandler()
        try:
            fmt_handler.detect_format("dummy" + out_ext)
        except ValueError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)

        # Determine output directory
        out_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(batch_folder, "restored")
        os.makedirs(out_dir, exist_ok=True)

        suffix = args.output_suffix  # e.g. "" or "_restored"

        input_files = _collect_audio_files(batch_folder)
        if not input_files:
            print(f"No audio files found in: {batch_folder}", file=sys.stderr)
            sys.exit(1)

        log.info("Batch mode: %d files found in %s", len(input_files), batch_folder)
        log.info("Output folder: %s  |  Extension: %s", out_dir, out_ext)

        ok = 0
        failed = []
        for idx, in_path in enumerate(input_files, 1):
            stem = os.path.splitext(os.path.basename(in_path))[0]
            out_path = os.path.join(out_dir, stem + suffix + out_ext)
            log.info("[%d/%d] %s  →  %s", idx, len(input_files), os.path.basename(in_path), os.path.basename(out_path))
            try:
                pipeline.restore(in_path, out_path)
                ok += 1
            except Exception as exc:
                log.error("  FAILED: %s", exc, exc_info=args.verbose)
                failed.append((in_path, str(exc)))

        print(f"\nBatch complete: {ok}/{len(input_files)} succeeded.")
        if failed:
            print("Failed files:")
            for path, reason in failed:
                print(f"  {path}: {reason}")
            sys.exit(2)
        return

    # ------------------------------------------------------------------
    # Single-file mode
    # ------------------------------------------------------------------
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("ERROR: -o/--output is required in single-file mode.", file=sys.stderr)
        sys.exit(1)

    # Validate output format
    fmt_handler = FormatHandler()
    try:
        fmt_handler.detect_format(args.output)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        pipeline.restore(args.input, args.output)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        log.error("Pipeline failed: %s", exc, exc_info=True)
        sys.exit(2)


if __name__ == "__main__":
    main()
