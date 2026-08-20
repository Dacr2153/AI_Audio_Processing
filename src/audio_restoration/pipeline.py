"""Restoration pipeline — phase-6 orchestrator.

Coordinates all processing stages in the documented order:

  Phase 1      Load & pre-process        (io.preprocessing.AudioPreprocessor)
  Phase 1.3    Wow & flutter correction  (dsp.wow_flutter)   [optional]
  Phase 1.5    Electrical hum removal    (dsp.dehum)         [optional]
  Phase 1.7    Click / pop / crackle     (dsp.declicker)     [optional]
  Phase 2      Denoising                 (dsp.denoiser)
  Phase 3      Source separation         (neural)            [optional]
  Phase 4      Super-resolution          (neural)            [optional]
  Phase 5      Equalization & mastering  (dsp.equalizer)
  Phase 5.3    Multiband compression     (dsp.multiband_compressor) [optional]
  Phase 5.5    LUFS loudness normalise   (pyloudnorm)        [optional]
  Phase 5.7    M/S stereo processing     (dsp.ms_processor)  [optional]
  Phase 6      Metrics & report          (reporting)

Stereo is preserved end-to-end; every DSP stage is channel-aware.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from .config import PipelineConfig
from .dsp import (
    AudioEqualizer,
    BandSettings,
    Declicker,
    Dehummer,
    Denoiser,
    MSProcessor,
    MultibandCompressor,
    WowFlutterCorrector,
    audio_utils,
)
from .exceptions import NeuralModelUnavailableError
from .io.format_handler import FormatHandler
from .io.preprocessing import AudioPreprocessor
from .neural.source_separation import SourceSeparator
from .neural.super_resolution import SuperResolution
from .reporting.metrics import QualityMetrics

logger = logging.getLogger(__name__)


class RestorationPipeline:
    """Orchestrates the full audio restoration pipeline.

    Instantiate once (with a :class:`PipelineConfig`), then call
    :meth:`restore` for each file. Heavy neural models (DeepFilterNet,
    Demucs) are lazy-loaded on first use.
    """

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._metrics = QualityMetrics()
        self._fmt = FormatHandler()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def restore(self, input_path: str, output_path: str) -> dict:
        """Process a single audio file through the full restoration pipeline.

        Args:
            input_path: Path to the input audio file.
            output_path: Destination path; the format is inferred from the
                extension (.wav, .flac, .aiff, .mp3, .m4a, .ogg, .opus…).

        Returns:
            Dict with the quality-metrics report (see
            :meth:`reporting.metrics.QualityMetrics.compare`).
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
            trim_silence=False,  # Old recordings may start with noise
            normalize=True,
        )
        audio, sr = preprocessor.load_and_prepare(input_path)
        original_audio = audio.copy()  # For metrics comparison
        original_sr = sr
        logger.info("  Loaded: %s @ %d Hz (%.2f s)", audio.shape, sr, len(audio) / sr)

        # ----------------------------------------------------------------
        # Phase 1.3: Wow & flutter correction
        # ----------------------------------------------------------------
        if cfg.wow_flutter.enabled:
            logger.info(
                "[Phase 1.3] Wow & flutter correction (max_cents=%.0f, smoothing=%.0f ms)…",
                cfg.wow_flutter.max_cents,
                cfg.wow_flutter.smoothing_ms,
            )
            waffler = WowFlutterCorrector(
                frame_ms=cfg.wow_flutter.frame_ms,
                hop_ms=cfg.wow_flutter.hop_ms,
                max_deviation_cents=cfg.wow_flutter.max_cents,
                correction_smoothing_ms=cfg.wow_flutter.smoothing_ms,
                max_freq_hz=cfg.wow_flutter.max_freq_hz,
            )
            audio = waffler.process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 1.5: Electrical hum removal (dehum)
        # ----------------------------------------------------------------
        if cfg.dehum.freq is not None:
            logger.info(
                "[Phase 1.5] Dehum: %.0f Hz + %d harmonics (Q=%.0f)…",
                cfg.dehum.freq,
                cfg.dehum.harmonics,
                cfg.dehum.q,
            )
            hummer = Dehummer(
                freq=cfg.dehum.freq,
                harmonics=cfg.dehum.harmonics,
                q=cfg.dehum.q,
            )
            audio = hummer.process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 1.7: Click / pop / crackle removal
        # ----------------------------------------------------------------
        if cfg.declick.enabled:
            logger.info(
                "[Phase 1.7] Declicking (threshold=%.1f, lpc_order=%d)…",
                cfg.declick.threshold,
                cfg.declick.lpc_order,
            )
            declicker = Declicker(
                threshold=cfg.declick.threshold,
                margin_ms=cfg.declick.margin_ms,
                lpc_order=cfg.declick.lpc_order,
                max_click_ms=cfg.declick.max_click_ms,
                max_scratch_ms=cfg.declick.max_scratch_ms,
            )
            audio = declicker.process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 2: Denoising
        # ----------------------------------------------------------------
        logger.info(
            "[Phase 2] Denoising (method=%s, passes=%d)…",
            cfg.denoise.method,
            cfg.denoise.passes,
        )
        denoiser = Denoiser(
            method=cfg.denoise.method,
            prop_decrease=cfg.denoise.prop_decrease,
            stationary=cfg.denoise.stationary,
            n_std_thresh=cfg.denoise.n_std_thresh,
            passes=cfg.denoise.passes,
        )
        audio = denoiser.denoise(audio, sr)
        used_method = (
            cfg.denoise.method
            if cfg.denoise.method != "auto"
            else denoiser.available_method()
        )
        logger.info(
            "  Denoiser applied: %s (prop_decrease=%.2f, n_std=%.1f, passes=%d)",
            used_method,
            cfg.denoise.prop_decrease,
            cfg.denoise.n_std_thresh,
            cfg.denoise.passes,
        )

        # ----------------------------------------------------------------
        # Phase 3: Source separation (optional, stereo-aware)
        # ----------------------------------------------------------------
        if cfg.separate.enabled:
            logger.info("[Phase 3] Source separation (Demucs %s)…", cfg.separate.model)
            audio = self._run_source_separation(audio, sr)

        # ----------------------------------------------------------------
        # Phase 4: Super-resolution (optional)
        # ----------------------------------------------------------------
        if cfg.sr.enabled:
            logger.info("[Phase 4] Super-resolution (target %d Hz)…", cfg.sr.target_sr)
            sr_module = SuperResolution(
                target_sr=cfg.sr.target_sr, device=cfg.sr.device
            )
            audio, sr = sr_module.upsample(audio, sr)
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
            logger.info("  New sample rate: %d Hz", sr)

        # ----------------------------------------------------------------
        # Phase 5: Equalization & mastering
        # ----------------------------------------------------------------
        eq = cfg.eq
        logger.info(
            "[Phase 5] Equalizing (bass=%+.1f  mid=%+.1f  presence=%+.1f  treble=%+.1f  air=%+.1f dB)…",
            eq.bass_gain_db,
            eq.mid_gain_db,
            eq.presence_gain_db,
            eq.treble_gain_db,
            eq.air_gain_db,
        )
        equalizer = AudioEqualizer(
            bass_gain_db=eq.bass_gain_db,
            mid_gain_db=eq.mid_gain_db,
            presence_gain_db=eq.presence_gain_db,
            treble_gain_db=eq.treble_gain_db,
            air_gain_db=eq.air_gain_db,
            rumble_filter=eq.rumble_filter,
        )
        out_cfg = cfg.output
        audio = equalizer.process(
            audio,
            sr,
            normalize=out_cfg.normalize,
            limit=out_cfg.limit,
            limit_threshold_db=out_cfg.limit_threshold_db,
        )

        # ----------------------------------------------------------------
        # Phase 5.3: Multiband compression
        # ----------------------------------------------------------------
        mb = cfg.multiband
        if mb.enabled:
            logger.info("[Phase 5.3] Multiband compression…")
            audio = MultibandCompressor(
                crossover_low=mb.xover_low,
                crossover_high=mb.xover_high,
                band_settings=[
                    BandSettings(
                        threshold_db=mb.low_threshold_db,
                        ratio=mb.low_ratio,
                        makeup_gain_db=mb.low_makeup_db,
                    ),
                    BandSettings(
                        threshold_db=mb.mid_threshold_db,
                        ratio=mb.mid_ratio,
                        makeup_gain_db=mb.mid_makeup_db,
                    ),
                    BandSettings(
                        threshold_db=mb.high_threshold_db,
                        ratio=mb.high_ratio,
                        makeup_gain_db=mb.high_makeup_db,
                    ),
                ],
            ).process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 5.7: M/S stereo processing (stereo only)
        # ----------------------------------------------------------------
        if cfg.ms.enabled:
            logger.info("[Phase 5.7] M/S stereo processing…")
            audio = MSProcessor(cfg.ms).process(audio, sr)

        # ----------------------------------------------------------------
        # Phase 5.5: LUFS loudness normalization (streaming standard)
        # ----------------------------------------------------------------
        if cfg.loudness.target_lufs is not None:
            audio = self._apply_lufs(audio, sr)

        # ----------------------------------------------------------------
        # Optional: Final output resampling
        # ----------------------------------------------------------------
        if out_cfg.sample_rate is not None and sr != out_cfg.sample_rate:
            logger.info("[Output] Resampling to %d Hz…", out_cfg.sample_rate)
            audio = audio_utils.resample(audio, sr, out_cfg.sample_rate).astype(
                np.float32
            )
            sr = out_cfg.sample_rate

        # ----------------------------------------------------------------
        # Save output
        # ----------------------------------------------------------------
        logger.info("[Save] Writing %s…", self._fmt.format_info(output_path))
        self._fmt.write(
            np.asarray(audio, dtype=np.float32),
            sr,
            output_path,
            bitrate=out_cfg.bitrate,
            bit_depth=out_cfg.bit_depth,
            vbr=out_cfg.vbr,
        )
        elapsed = time.time() - t_start
        logger.info("Saved: %s  [%.1f s processing time]", output_path, elapsed)

        # ----------------------------------------------------------------
        # Phase 6: Metrics & report
        # ----------------------------------------------------------------
        logger.info("[Phase 6] Computing quality metrics…")
        if original_sr != sr:
            orig_resampled = audio_utils.resample(
                original_audio, original_sr, sr
            ).astype(np.float32)
        else:
            orig_resampled = original_audio

        report = self._metrics.compare(orig_resampled, audio, sr)

        if cfg.report.print_metrics:
            print(self._metrics.format_report(report))

        if cfg.report.save_comparison_plot:
            plot_path = os.path.splitext(output_path)[0] + "_comparison.png"
            self._metrics.plot_comparison(
                orig_resampled,
                audio,
                sr,
                output_path=plot_path,
                title=f"Restoration: {os.path.basename(input_path)}",
            )

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_source_separation(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Demucs stem separation, optional per-stem denoise, then remix.

        Demucs operates on a WAV on disk and returns stems at 44.1 kHz;
        the reconstructed mix is resampled back to the pipeline SR.
        """
        cfg = self.config
        separator = SourceSeparator(
            model=cfg.separate.model,
            device=cfg.separate.device,
        )

        if not separator.is_available:
            logger.warning("Demucs not available — skipping source separation.")
            return audio

        try:
            stems = separator.separate_from_array(
                np.asarray(audio, dtype=np.float32), sr
            )
        except NeuralModelUnavailableError:
            logger.warning("Demucs unavailable — skipping source separation.")
            return audio

        if not stems:
            logger.warning("No stems returned from Demucs — using original audio.")
            return audio

        # Optionally denoise each stem independently before mixing.
        if cfg.separate.stem_denoise:
            logger.info("  Stem-by-stem denoising (%d stems)…", len(stems))
            stem_denoiser = Denoiser(
                method="noisereduce",
                prop_decrease=min(cfg.denoise.prop_decrease * 0.5, 0.6),
                stationary=cfg.denoise.stationary,
            )
            stems = {
                name: stem_denoiser.denoise(np.asarray(stem, dtype=np.float32), 44_100)
                for name, stem in stems.items()
            }

        # Sum all stems to reconstruct the full mix.
        stem_arrays = list(stems.values())
        min_len = min(len(s) for s in stem_arrays)
        mixed = np.sum([np.asarray(s)[:min_len] for s in stem_arrays], axis=0)
        mixed = np.clip(mixed, -1.0, 1.0).astype(np.float32)

        # Demucs outputs at 44 100 Hz — resample back to the pipeline SR.
        if 44_100 != sr:
            mixed = audio_utils.resample(mixed, 44_100, sr).astype(np.float32)

        return mixed

    def _apply_lufs(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Normalize to the target integrated loudness (BS.1770-4 / EBU R128).

        Falls back to peak normalization if pyloudnorm is not installed or
        the signal is too short for a reliable loudness measurement (< 0.4 s).
        """
        cfg = self.config
        try:
            import pyloudnorm as pyln
        except ImportError:
            logger.warning("pyloudnorm not installed — skipping LUFS normalization.")
            return audio

        target = cfg.loudness.target_lufs
        if len(audio) < int(sr * 0.4):
            logger.warning("Audio too short for LUFS measurement — skipping.")
            return audio

        meter = pyln.Meter(sr)  # BS.1770-4 loudness meter
        audio_64 = np.asarray(audio, dtype=np.float64)
        loudness = meter.integrated_loudness(audio_64)

        if not np.isfinite(loudness) or loudness < -70.0:
            logger.warning(
                "LUFS measurement unreliable (%.1f LUFS) — skipping.", loudness
            )
            return audio

        normalized = pyln.normalize.loudness(audio_64, loudness, target)
        # Hard-clip to ±1.0 to prevent post-normalization overs.
        normalized = np.clip(normalized, -1.0, 1.0)
        logger.info(
            "  LUFS: %.1f → %.1f LUFS (target %.1f)",
            loudness,
            meter.integrated_loudness(normalized),
            target,
        )
        return normalized.astype(audio.dtype)
