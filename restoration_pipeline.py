"""
Audio Restoration Pipeline — Phase 6 (Orchestrator).

Full end-to-end restoration pipeline for degraded audio recordings.
Integrates all processing phases in the correct order:

  Phase 1: Load & pre-process  (Pre-processing.py → AudioPreprocessor)
  Phase 2: Denoising           (denoiser.py       → Denoiser)
  Phase 3: Source separation   (source_separator.py → SourceSeparator)  [optional]
  Phase 4: Super-resolution    (super_resolution.py → SuperResolution)  [optional]
  Phase 5: Equalization        (equalizer.py      → AudioEqualizer)
  Phase 6: Metrics & report    (metrics.py        → QualityMetrics)

Usage::

    # Programmatic use
    from restoration_pipeline import RestorationPipeline, PipelineConfig

    config = PipelineConfig(
        denoise_method="auto",
        bass_gain_db=2.0,
        treble_gain_db=3.0,
        enable_source_separation=False,
        enable_super_resolution=True,
    )
    pipeline = RestorationPipeline(config)
    pipeline.restore("old_song.wav", "restored_song.wav")

    # Or use the CLI:
    #   python restore.py --input old_song.wav --output restored.wav --bass 3 --treble 2
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import soundfile as sf

# Add the project directory to sys.path so sibling modules are importable
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "pre_processing",
    os.path.join(_THIS_DIR, "Pre-processing.py"),
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
AudioPreprocessor = _mod.AudioPreprocessor

from denoiser import Denoiser
from declicker import Declicker
from equalizer import AudioEqualizer
from format_handler import FormatHandler
from metrics import QualityMetrics
from multiband_compressor import MultibandCompressor, BandSettings
from ms_processor import MSProcessor, MSConfig
from source_separator import SourceSeparator
from wow_flutter import WowFlutterCorrector
from super_resolution import SuperResolution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RestorationPipeline")

# ---------------------------------------------------------------------------
# Genre EQ presets
# ---------------------------------------------------------------------------
# Each preset is a dict of EQ gains (dB) tuned for the target genre.
# The pipeline applies the preset unless individual band flags override it.
GENRE_PRESETS: dict = {
    "vinyl":      {"bass_gain_db":  2.5, "mid_gain_db": -1.5, "presence_gain_db":  2.5, "treble_gain_db":  2.0, "air_gain_db":  3.5},
    "jazz":       {"bass_gain_db":  1.0, "mid_gain_db":  0.0, "presence_gain_db":  1.0, "treble_gain_db":  1.5, "air_gain_db":  2.0},
    "classical":  {"bass_gain_db":  1.5, "mid_gain_db": -0.5, "presence_gain_db":  1.0, "treble_gain_db":  1.5, "air_gain_db":  3.0},
    "hiphop":     {"bass_gain_db":  5.0, "mid_gain_db": -2.0, "presence_gain_db":  1.5, "treble_gain_db":  1.5, "air_gain_db":  2.5},
    "metal":      {"bass_gain_db":  2.0, "mid_gain_db": -3.5, "presence_gain_db":  3.0, "treble_gain_db":  2.5, "air_gain_db":  2.0},
    "electronic": {"bass_gain_db":  4.0, "mid_gain_db": -1.0, "presence_gain_db":  2.0, "treble_gain_db":  2.5, "air_gain_db":  3.0},
    "podcast":    {"bass_gain_db":  0.0, "mid_gain_db": -1.0, "presence_gain_db":  3.5, "treble_gain_db":  1.5, "air_gain_db":  1.0},
    "flat":       {"bass_gain_db":  0.0, "mid_gain_db":  0.0, "presence_gain_db":  0.0, "treble_gain_db":  0.0, "air_gain_db":  0.0},
}


@dataclass
class PipelineConfig:
    """
    Configuration for the AudioRestorationPipeline.

    All parameters have sensible defaults for old recordings.
    """
    # --- Denoising ---
    denoise_method: str = "music"
    """Denoiser method: 'auto', 'music', 'noisereduce', 'deepfilternet', 'wavelet'."""

    denoise_stationary: bool = False
    """For noisereduce only: assume stationary noise (constant hiss)."""

    denoise_prop_decrease: float = 0.85
    """Noise reduction aggressiveness (0–1). 0.5 = balanced for music. 0.8+ for speech."""

    denoise_n_std_thresh: float = 1.5
    """For noisereduce stationary mode: number of std-devs above noise mean before a
    frequency bin is treated as signal. Lower = more noise removed.
    1.5 = conservative (default) | 1.0 = balanced | 0.5 = aggressive (eliminates static)."""

    denoise_passes: int = 1
    """Number of sequential denoising passes. Each pass cleans up what the previous
    left behind. 1 = standard. 2–3 = for stubborn static/hiss. (default: 1)"""

    # --- Source separation ---
    enable_source_separation: bool = False
    """Separate the audio into stems with Demucs. Recommended for songs."""

    demucs_model: str = "htdemucs_ft"
    """Demucs model: 'htdemucs_ft' (best), 'htdemucs', 'mdx_extra'."""

    demucs_device: str = "auto"
    """Device for Demucs: 'auto', 'cpu', 'cuda'."""

    reconstruct_from_stems: bool = True
    """After separation, reconstruct the full mix from processed stems."""

    # --- Super-resolution ---
    enable_super_resolution: bool = False
    """Upsample audio to 48 kHz using AudioSR or scipy resampler."""

    super_resolution_device: str = "auto"
    """Device for AudioSR: 'auto', 'cpu', 'cuda'."""

    # --- Equalization (5-band mastering chain) ---
    bass_gain_db: float = 2.5
    """Low-shelf gain @ 80 Hz in dB. +2.5 restores warmth lost in vinyl transfer."""

    mid_gain_db: float = -1.5
    """Peaking-bell gain @ 250 Hz. -1.5 cleans up muddy/boxy resonance common in vinyl."""

    presence_gain_db: float = 2.5
    """Peaking-bell gain @ 3500 Hz. +2.5 sharpens vocal articulation and instrument attack."""

    treble_gain_db: float = 2.0
    """High-shelf gain @ 8000 Hz. +2.0 restores high-frequency clarity and definition."""

    air_gain_db: float = 3.5
    """High-shelf gain @ 12000 Hz. +3.5 adds sparkle and open 'air' of modern masters."""

    eq_rumble_filter: bool = True
    """Apply sub-sonic HPF @ 30 Hz to remove turntable rumble."""

    # Legacy crossover fields — kept for backward compat, no longer used by equalizer
    eq_crossover_low: float = 250.0
    eq_crossover_high: float = 4000.0

    # --- Electrical hum removal ---
    dehum_freq: Optional[float] = None
    """Hum fundamental frequency: 50.0 (Europe/Asia) or 60.0 (Americas). None = disabled."""

    dehum_harmonics: int = 5
    """How many harmonics to notch (including the fundamental). Default 5."""

    dehum_q: float = 35.0
    """Notch filter Q. Higher = narrower notch, less music coloration. Default 35."""

    # --- Click / pop / crackle removal ---
    enable_declicker: bool = False
    """Enable click/pop/crackle removal. Recommended for vinyl and cassette recordings."""

    declicker_threshold: float = 4.0
    """Click detection threshold (× local RMS). Lower = more aggressive.
    5–8 for vinyl, 4–6 for heavy crackle."""

    declicker_margin_ms: float = 5.0
    """Extra padding around detected clicks in ms. Ensures tails are covered."""

    declicker_lpc_order: int = 32
    """LPC interpolation order. Higher = better timbre modeling for complex sounds."""

    declicker_max_click_ms: float = 30.0
    """Maximum short-click duration in ms. Clicks shorter than this use LPC
    forward prediction.  Set lower to be more conservative about what counts
    as a click (vs. a drum attack)."""

    declicker_max_scratch_ms: float = 200.0
    """Maximum scratch duration in ms repaired with bidirectional crossfade.
    Needle drags and long crackles (30–200 ms) are handled here.
    Events longer than this are left untouched (too risky to replace)."""

    # --- Loudness normalization (LUFS) ---
    lufs_target: Optional[float] = -14.0
    """Target integrated loudness in LUFS (EBU R128 / BS.1770-4).
    -14 LUFS = Spotify / Apple Music / YouTube streaming standard.
    -23 LUFS = EBU R128 broadcast standard.
    None = disabled (use peak normalization only)."""

    # --- Multiband compression ---
    enable_multiband: bool = False
    """Enable 3-band Linkwitz-Riley compressor (Phase 5.3)."""

    multiband_low_threshold_db: float = -20.0
    """Compressor threshold for the low band (dBFS)."""
    multiband_low_ratio: float = 2.5
    multiband_low_makeup_db: float = 1.5

    multiband_mid_threshold_db: float = -18.0
    """Compressor threshold for the mid band (dBFS)."""
    multiband_mid_ratio: float = 3.0
    multiband_mid_makeup_db: float = 1.0

    multiband_high_threshold_db: float = -16.0
    """Compressor threshold for the high band (dBFS)."""
    multiband_high_ratio: float = 2.0
    multiband_high_makeup_db: float = 0.5

    # --- M/S stereo processing ---
    enable_ms: bool = False
    """Enable Mid/Side stereo processing (Phase 5.7). Only affects stereo audio."""

    ms_side_denoise: bool = False
    """Denoise the Side channel in M/S mode. Off by default — the Side channel
    contains real stereo content (panning) that noisereduce would interpret as
    'noise' and remove, collapsing the stereo image. Only enable for recordings
    with a clearly corrupted side channel (e.g. heavy tape hiss on one channel)."""

    ms_side_prop_decrease: float = 0.6
    """Noise reduction strength on the Side channel (0–1). Default 0.6."""

    ms_side_gain: float = 1.0
    """Side channel amplitude scale: < 1 = narrower, > 1 = wider. Default 1.0."""

    ms_mid_presence_db: float = 0.0
    """Presence boost/cut on the Mid channel in dB. 0 = disabled."""

    ms_mid_presence_freq: float = 3500.0
    """Centre frequency for Mid presence EQ in Hz. (default: 3500)"""

    # --- Wow & flutter correction ---
    enable_wow_flutter: bool = False
    """Enable wow & flutter correction (Phase 1.3). Corrects slow pitch instability
    caused by worn turntable/cassette mechanics. Best for vinyl and cassette."""

    wow_flutter_frame_ms: float = 50.0
    """Analysis frame length in ms for pitch tracking. (default: 50 ms)"""

    wow_flutter_hop_ms: float = 10.0
    """Hop size between frames in ms. Smaller = higher temporal resolution. (default: 10 ms)"""

    wow_flutter_max_cents: float = 100.0
    """Maximum pitch correction in cents (100 cents = 1 semitone). Deviations larger
    than this are clamped to avoid over-correction. (default: 100)"""

    wow_flutter_smoothing_ms: float = 200.0
    """Smoothing window for the long-term nominal pitch trajectory in ms.
    Longer = only correct slow wow; shorter = also correct fast flutter. (default: 200)"""

    wow_flutter_max_freq_hz: float = 100.0
    """Upper frequency limit of fluctuations to correct in Hz. (default: 100)"""

    # --- Stem-by-stem denoising ---
    stem_denoise: bool = False
    """When True and enable_source_separation is also True, each Demucs stem
    (vocals, drums, bass, other) is denoised independently before being summed
    back into the full mix.  This gives the denoiser a cleaner, simpler signal
    to work with per instrument and typically yields better results than
    denoising the full mix alone.  Uses noisereduce with half the
    denoise_prop_decrease for a lighter touch on already-separated stems."""

    # --- Output ---
    output_sample_rate: Optional[int] = None
    """If set, resample output to this rate. None = match pipeline output SR."""

    normalize_output: bool = True
    """Peak-normalize the final output."""

    limit_output: bool = True
    """Apply soft limiter to prevent digital clipping."""

    limit_threshold_db: float = -0.3
    """Limiter threshold (dBFS). Default -0.3 leaves slight headroom."""

    # --- Reporting ---
    # --- Output format ---
    output_bitrate: Optional[str] = None
    """Bitrate for lossy output formats. Examples: '320k', '256k', 'V0'. None = format default."""

    output_bit_depth: int = 24
    """Bit depth for lossless output formats: 16, 24, or 32."""

    output_vbr: bool = True
    """Use VBR (variable bitrate) for MP3 output. False = CBR."""

    save_comparison_plot: bool = True
    """Save waveform/spectrogram comparison PNG next to the output file."""

    print_metrics: bool = True
    """Print quality metrics report to stdout."""


class RestorationPipeline:
    """
    Orchestrates the full audio restoration pipeline.

    Instantiate once, then call restore() for each file.
    Heavy models (DeepFilterNet, Demucs) are lazy-loaded on first use.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self._metrics = QualityMetrics()
        self._fmt = FormatHandler()

    def restore(self, input_path: str, output_path: str) -> dict:
        """
        Processes a single audio file through the full restoration pipeline.

        Args:
            input_path: Path to the input audio file.
            output_path: Destination path. Format is inferred from the extension
                         (.wav, .flac, .aiff, .mp3, .m4a, .ogg, .opus, etc.).

        Returns:
            Dict with quality metrics report.
        """
        cfg = self.config
        t_start = time.time()
        logger.info("=" * 60)
        logger.info("RESTORING: %s", input_path)
        logger.info("=" * 60)

        # ----------------------------------------------------------------
        # Phase 1: Load & pre-process
        # ----------------------------------------------------------------
        logger.info("[Phase 1] Loading and pre-processing audio…")
        preprocessor = AudioPreprocessor(
            trim_silence=False,  # Don't trim — old recordings may start with noise
            normalize=True,
        )
        audio, sr = preprocessor.load_and_prepare(input_path)
        original_audio = audio.copy()  # Keep for metrics comparison
        original_sr = sr
        logger.info("  Loaded: %d samples @ %d Hz (%.2f s)", len(audio), sr, len(audio) / sr)

        # ----------------------------------------------------------------
        # Phase 1.3: Wow & flutter correction
        # ----------------------------------------------------------------
        if cfg.enable_wow_flutter:
            logger.info(
                "[Phase 1.3] Wow & flutter correction (max_cents=%.0f, smoothing=%.0f ms)…",
                cfg.wow_flutter_max_cents, cfg.wow_flutter_smoothing_ms,
            )
            wfc = WowFlutterCorrector(
                frame_ms=cfg.wow_flutter_frame_ms,
                hop_ms=cfg.wow_flutter_hop_ms,
                max_deviation_cents=cfg.wow_flutter_max_cents,
                correction_smoothing_ms=cfg.wow_flutter_smoothing_ms,
                max_freq_hz=cfg.wow_flutter_max_freq_hz,
            )
            audio = wfc.process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 1.5: Electrical hum removal (dehum)
        # ----------------------------------------------------------------
        if cfg.dehum_freq is not None:
            audio = self._apply_dehum(audio, sr)
            logger.info(
                "  Dehum: %.0f Hz + %d harmonics (Q=%.0f)",
                cfg.dehum_freq, cfg.dehum_harmonics, cfg.dehum_q,
            )

        # ----------------------------------------------------------------
        # Phase 1.7: Click / pop / crackle removal
        # ----------------------------------------------------------------
        if cfg.enable_declicker:
            logger.info(
                "[Phase 1.7] Declicking (threshold=%.1f, lpc_order=%d)…",
                cfg.declicker_threshold, cfg.declicker_lpc_order,
            )
            declicker = Declicker(
                threshold=cfg.declicker_threshold,
                margin_ms=cfg.declicker_margin_ms,
                lpc_order=cfg.declicker_lpc_order,
                max_click_ms=cfg.declicker_max_click_ms,
                max_scratch_ms=cfg.declicker_max_scratch_ms,
            )
            audio = declicker.process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 2: Denoising
        # ----------------------------------------------------------------
        logger.info("[Phase 2] Denoising (method=%s, passes=%d)…", cfg.denoise_method, cfg.denoise_passes)
        denoiser = Denoiser(
            method=cfg.denoise_method,
            prop_decrease=cfg.denoise_prop_decrease,
            stationary=cfg.denoise_stationary,
            n_std_thresh=cfg.denoise_n_std_thresh,
            passes=cfg.denoise_passes,
        )
        audio = denoiser.denoise(audio, sr)
        used_method = cfg.denoise_method if cfg.denoise_method != "auto" else denoiser.available_method()
        logger.info("  Denoiser applied: %s (prop_decrease=%.2f, n_std=%.1f, passes=%d)",
                    used_method, cfg.denoise_prop_decrease, cfg.denoise_n_std_thresh, cfg.denoise_passes)

        # ----------------------------------------------------------------
        # Phase 3: Source separation (optional)
        # ----------------------------------------------------------------
        if cfg.enable_source_separation:
            logger.info("[Phase 3] Source separation (Demucs %s)…", cfg.demucs_model)
            audio = self._run_source_separation(audio, sr, input_path)

        # ----------------------------------------------------------------
        # Phase 4: Super-resolution (optional)
        # ----------------------------------------------------------------
        if cfg.enable_super_resolution:
            logger.info("[Phase 4] Super-resolution (target 48 kHz)…")
            sr_module = SuperResolution(device=cfg.super_resolution_device)
            audio, sr = sr_module.upsample(audio, sr)
            logger.info("  New sample rate: %d Hz", sr)

        # ----------------------------------------------------------------
        # Phase 5: Equalization & mastering
        # ----------------------------------------------------------------
        logger.info(
            "[Phase 5] Equalizing (bass=%+.1f  mid=%+.1f  presence=%+.1f  treble=%+.1f  air=%+.1f dB)…",
            cfg.bass_gain_db, cfg.mid_gain_db, cfg.presence_gain_db,
            cfg.treble_gain_db, cfg.air_gain_db,
        )
        eq = AudioEqualizer(
            bass_gain_db=cfg.bass_gain_db,
            mid_gain_db=cfg.mid_gain_db,
            presence_gain_db=cfg.presence_gain_db,
            treble_gain_db=cfg.treble_gain_db,
            air_gain_db=cfg.air_gain_db,
            rumble_filter=cfg.eq_rumble_filter,
        )
        audio = eq.process(
            audio, sr,
            normalize=cfg.normalize_output,
            limit=cfg.limit_output,
            limit_threshold_db=cfg.limit_threshold_db,
        )

        # ----------------------------------------------------------------
        # Phase 5.3: Multiband compression
        # ----------------------------------------------------------------
        if cfg.enable_multiband:
            logger.info("[Phase 5.3] Multiband compression…")
            band_settings = [
                BandSettings(
                    threshold_db=cfg.multiband_low_threshold_db,
                    ratio=cfg.multiband_low_ratio,
                    makeup_gain_db=cfg.multiband_low_makeup_db,
                ),
                BandSettings(
                    threshold_db=cfg.multiband_mid_threshold_db,
                    ratio=cfg.multiband_mid_ratio,
                    makeup_gain_db=cfg.multiband_mid_makeup_db,
                ),
                BandSettings(
                    threshold_db=cfg.multiband_high_threshold_db,
                    ratio=cfg.multiband_high_ratio,
                    makeup_gain_db=cfg.multiband_high_makeup_db,
                ),
            ]
            mbc = MultibandCompressor(
                crossover_low=cfg.eq_crossover_low,
                crossover_high=cfg.eq_crossover_high,
                band_settings=band_settings,
            )
            audio = mbc.process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 5.7: M/S stereo processing (stereo only)
        # ----------------------------------------------------------------
        if cfg.enable_ms:
            ms_cfg = MSConfig(
                side_denoise=cfg.ms_side_denoise,
                side_prop_decrease=cfg.ms_side_prop_decrease,
                side_gain=cfg.ms_side_gain,
                mid_presence_db=cfg.ms_mid_presence_db,
                mid_presence_freq=cfg.ms_mid_presence_freq,
            )
            audio = MSProcessor(ms_cfg).process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 5.5: LUFS loudness normalization (streaming standard)
        # ----------------------------------------------------------------
        if cfg.lufs_target is not None:
            audio = self._apply_lufs(audio, sr)

        # ----------------------------------------------------------------
        # Optional: Final output resampling
        # ----------------------------------------------------------------
        if cfg.output_sample_rate is not None and sr != cfg.output_sample_rate:
            logger.info("[Output] Resampling to %d Hz…", cfg.output_sample_rate)
            from scipy import signal as sp_signal
            from math import gcd
            g = gcd(sr, cfg.output_sample_rate)
            up = cfg.output_sample_rate // g
            down = sr // g
            audio = sp_signal.resample_poly(audio.astype(np.float64), up, down).astype(np.float32)
            sr = cfg.output_sample_rate

        # ----------------------------------------------------------------
        # Save output
        # ----------------------------------------------------------------
        out_fmt = self._fmt.detect_format(output_path)
        fmt_info = self._fmt.format_info(output_path)
        logger.info("[Save] Writing %s…", fmt_info)
        self._fmt.write(
            audio.astype(np.float32),
            sr,
            output_path,
            bitrate=cfg.output_bitrate,
            bit_depth=cfg.output_bit_depth,
            vbr=cfg.output_vbr,
        )
        elapsed = time.time() - t_start
        logger.info("Saved: %s  [%.1f s processing time]", output_path, elapsed)

        # ----------------------------------------------------------------
        # Phase 6: Metrics & report
        # ----------------------------------------------------------------
        logger.info("[Phase 6] Computing quality metrics…")

        # Align lengths for comparison (original might be different SR)
        from scipy.signal import resample_poly
        from math import gcd as _gcd
        if original_sr != sr:
            g = _gcd(original_sr, sr)
            orig_resampled = resample_poly(
                original_audio.astype(np.float64),
                sr // g,
                original_sr // g,
            ).astype(np.float32)
        else:
            orig_resampled = original_audio

        report = self._metrics.compare(orig_resampled, audio, sr)

        if cfg.print_metrics:
            print(self._metrics.format_report(report))

        if cfg.save_comparison_plot:
            plot_path = os.path.splitext(output_path)[0] + "_comparison.png"
            self._metrics.plot_comparison(
                orig_resampled, audio, sr,
                output_path=plot_path,
                title=f"Restoration: {os.path.basename(input_path)}",
            )

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_source_separation(
        self,
        audio: np.ndarray,
        sr: int,
        original_input_path: str,
    ) -> np.ndarray:
        """
        Runs Demucs on the original input file, then returns the full
        reconstructed mix (all stems summed) so the pipeline can continue.

        We pass the original file directly to Demucs (rather than the
        in-memory array) because Demucs requires a WAV file on disk.
        If the original was already denoised, we write a temp WAV.
        """
        import tempfile

        cfg = self.config
        separator = SourceSeparator(
            model=cfg.demucs_model,
            device=cfg.demucs_device,
        )

        if not separator.is_available:
            logger.warning("Demucs not available — skipping source separation.")
            return audio

        # Write current (denoised) audio to a temp file for Demucs
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            sf.write(tmp_path, audio.astype(np.float32), sr, subtype='PCM_16')
            stems = separator.separate(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

        if not stems:
            logger.warning("No stems returned from Demucs — using original audio.")
            return audio

        # Optionally denoise each stem independently before mixing
        if cfg.stem_denoise:
            logger.info("  Stem-by-stem denoising (%d stems)…", len(stems))
            stem_denoiser = Denoiser(
                method="noisereduce",
                # Lighter pass — stems are already partially clean from Phase 2
                prop_decrease=min(cfg.denoise_prop_decrease * 0.5, 0.6),
                stationary=cfg.denoise_stationary,
            )
            denoised_stems = {}
            for name, stem_audio in stems.items():
                logger.info("    Denoising stem: %s", name)
                denoised_stems[name] = stem_denoiser.denoise(stem_audio, 44100)
            stems = denoised_stems

        # Sum all stems to reconstruct the full mix
        stem_arrays = list(stems.values())
        min_len = min(len(s) for s in stem_arrays)
        mixed = np.sum([s[:min_len] for s in stem_arrays], axis=0)
        mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

        # Demucs outputs at 44100 Hz — resample back to pipeline SR if needed
        if 44100 != sr:
            from scipy.signal import resample_poly
            from math import gcd as _gcd
            g = _gcd(44100, sr)
            mixed = resample_poly(
                mixed.astype(np.float64), sr // g, 44100 // g
            ).astype(np.float32)

        return mixed

    def _apply_dehum(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Removes electrical hum (50 or 60 Hz) and its harmonics using IIR
        notch filters (scipy.signal.iirnotch), applied zero-phase via filtfilt.

        A Q of 35 makes each notch about 1.4 Hz wide at 50 Hz — narrow enough
        to avoid musical coloration while fully eliminating the hum tone.
        """
        from scipy.signal import iirnotch, filtfilt

        cfg = self.config
        y = audio.astype(np.float64)
        nyq = sr / 2.0

        for h in range(1, cfg.dehum_harmonics + 1):
            freq = cfg.dehum_freq * h
            if freq >= nyq * 0.99:
                break
            b, a = iirnotch(freq, Q=cfg.dehum_q, fs=sr)
            y = filtfilt(b, a, y)

        return y.astype(audio.dtype)

    def _apply_lufs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Normalizes the audio to the target integrated loudness (LUFS) using
        the BS.1770-4 / EBU R128 algorithm via the pyloudnorm library.

        Falls back to peak normalization if pyloudnorm is not installed or if
        the signal is too short for a reliable loudness measurement (< 0.4 s).
        """
        cfg = self.config
        try:
            import pyloudnorm as pyln
        except ImportError:
            logger.warning("pyloudnorm not installed — skipping LUFS normalization.")
            return audio

        if len(audio) < int(sr * 0.4):
            logger.warning("Audio too short for LUFS measurement — skipping.")
            return audio

        meter = pyln.Meter(sr)  # BS.1770-4 loudness meter
        loudness = meter.integrated_loudness(audio.astype(np.float64))

        if not np.isfinite(loudness) or loudness < -70.0:
            logger.warning("LUFS measurement unreliable (%.1f LUFS) — skipping.", loudness)
            return audio

        normalized = pyln.normalize.loudness(
            audio.astype(np.float64), loudness, cfg.lufs_target
        )
        # Hard-clip to ±1.0 to prevent any post-normalization overs
        normalized = np.clip(normalized, -1.0, 1.0)
        logger.info(
            "  LUFS: %.1f → %.1f LUFS (target %.1f)",
            loudness, meter.integrated_loudness(normalized), cfg.lufs_target,
        )
        return normalized.astype(audio.dtype)
