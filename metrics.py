"""
Metrics — Phase 6 of the Audio Restoration Pipeline.

Provides quality assessment tools:
  - SNR (Signal-to-Noise Ratio) estimation
  - PSNR (Peak Signal-to-Noise Ratio)
  - Spectral centroid comparison (brightness before/after)
  - Waveform and spectrogram visualization (before/after)
  - Text-based quality report
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Matplotlib is required for plots; skip gracefully if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend (safe for scripts)
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.info("matplotlib not found — visualizations disabled.")

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


class QualityMetrics:
    """
    Computes and visualizes audio quality metrics.

    Usage::

        metrics = QualityMetrics()
        report = metrics.compare(original, restored, sample_rate=44100)
        print(metrics.format_report(report))
        metrics.plot_comparison(original, restored, 44100, "comparison.png")
    """

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    @staticmethod
    def snr(reference: np.ndarray, noisy: np.ndarray) -> float:
        """
        Estimates SNR (Signal-to-Noise Ratio) in dB.

        SNR = 10 * log10(signal_power / noise_power)

        Where noise = reference − noisy.
        A higher SNR means less noise relative to the signal.
        Typical values: CD quality >80 dB, cassette tape ~50 dB,
        vinyl record ~60 dB, AM radio ~40 dB.

        Args:
            reference: The clean or processed signal.
            noisy: The noisy or original signal.

        Returns:
            SNR in dB.
        """
        ref = reference.astype(np.float64)
        nsy = noisy.astype(np.float64)
        min_len = min(len(ref), len(nsy))
        ref, nsy = ref[:min_len], nsy[:min_len]

        signal_power = np.mean(ref ** 2)
        noise_power = np.mean((ref - nsy) ** 2)
        if noise_power < 1e-14:
            return 120.0  # Essentially identical
        if signal_power < 1e-14:
            return 0.0
        return 10.0 * np.log10(signal_power / noise_power)

    @staticmethod
    def psnr(reference: np.ndarray, processed: np.ndarray) -> float:
        """
        Peak Signal-to-Noise Ratio in dB.
        PSNR = 20 * log10(max_value / RMSE)

        Args:
            reference: Reference waveform.
            processed: Processed waveform.

        Returns:
            PSNR in dB.
        """
        ref = reference.astype(np.float64)
        proc = processed.astype(np.float64)
        min_len = min(len(ref), len(proc))
        ref, proc = ref[:min_len], proc[:min_len]

        mse = np.mean((ref - proc) ** 2)
        if mse < 1e-14:
            return 120.0
        max_val = max(np.max(np.abs(ref)), 1e-9)
        return 20.0 * np.log10(max_val / np.sqrt(mse))

    @staticmethod
    def rms_db(audio: np.ndarray) -> float:
        """Returns the RMS level in dBFS."""
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if rms < 1e-12:
            return -120.0
        return 20.0 * np.log10(rms)

    @staticmethod
    def peak_db(audio: np.ndarray) -> float:
        """Returns the peak level in dBFS."""
        peak = np.max(np.abs(audio.astype(np.float64)))
        if peak < 1e-12:
            return -120.0
        return 20.0 * np.log10(peak)

    @staticmethod
    def dynamic_range_db(audio: np.ndarray) -> float:
        """Returns peak - RMS (a rough crest factor measure)."""
        return QualityMetrics.peak_db(audio) - QualityMetrics.rms_db(audio)

    @staticmethod
    def spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
        """
        Returns the mean spectral centroid in Hz — a measure of audio "brightness".
        Higher centroid = brighter sound (more high frequency content).
        """
        if not _LIBROSA_AVAILABLE:
            return 0.0
        centroid = librosa.feature.spectral_centroid(y=audio.astype(np.float32), sr=sample_rate)
        return float(np.mean(centroid))

    @staticmethod
    def high_freq_energy_ratio(audio: np.ndarray, sample_rate: int, threshold_hz: float = 8000.0) -> float:
        """
        Fraction of total power in frequencies above threshold_hz.
        Uses numpy.fft.rfft power spectrum. Returns a value between 0 and 1.
        """
        n = len(audio)
        x = np.asarray(audio, dtype=np.float64)
        # Use scipy signal's periodogram for a clean, explicit power estimate
        from scipy.signal import periodogram
        f, pxx = periodogram(x, fs=float(sample_rate), window='boxcar', scaling='density')
        total_power = float(np.sum(pxx))
        if total_power < 1e-30:
            return 0.0
        # Boolean index on a simple 1-D float64 array — no numba interference
        hf_mask = f >= float(threshold_hz)
        hf_power = float(np.sum(pxx[hf_mask]))
        ratio = hf_power / total_power
        # Clamp to [0, 1] as a safeguard
        return float(np.clip(ratio, 0.0, 1.0))

    def compare(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """
        Computes a full set of comparison metrics between original and restored audio.

        Args:
            original: Original (input) audio waveform.
            restored: Restored (output) audio waveform.
            sample_rate: Sample rate in Hz.

        Returns:
            Dict with all metrics.
        """
        report = {
            "original_rms_db": self.rms_db(original),
            "restored_rms_db": self.rms_db(restored),
            "original_peak_db": self.peak_db(original),
            "restored_peak_db": self.peak_db(restored),
            "original_dynamic_range_db": self.dynamic_range_db(original),
            "restored_dynamic_range_db": self.dynamic_range_db(restored),
            "original_spectral_centroid_hz": self.spectral_centroid(original, sample_rate),
            "restored_spectral_centroid_hz": self.spectral_centroid(restored, sample_rate),
            "original_hf_energy_ratio": self.high_freq_energy_ratio(original, sample_rate),
            "restored_hf_energy_ratio": self.high_freq_energy_ratio(restored, sample_rate),
            "snr_db": self.snr(restored, original),
            "psnr_db": self.psnr(restored, original),
        }
        return report

    @staticmethod
    def format_report(report: dict) -> str:
        """Formats a metrics report dict as a readable text summary."""
        lines = [
            "=" * 55,
            "  AUDIO RESTORATION QUALITY REPORT",
            "=" * 55,
            f"  {'METRIC':<30} {'ORIGINAL':>10} {'RESTORED':>10}",
            "-" * 55,
            f"  {'RMS Level (dBFS)':<30} {report['original_rms_db']:>10.2f} {report['restored_rms_db']:>10.2f}",
            f"  {'Peak Level (dBFS)':<30} {report['original_peak_db']:>10.2f} {report['restored_peak_db']:>10.2f}",
            f"  {'Dynamic Range (dB)':<30} {report['original_dynamic_range_db']:>10.2f} {report['restored_dynamic_range_db']:>10.2f}",
            f"  {'Spectral Centroid (Hz)':<30} {report['original_spectral_centroid_hz']:>10.1f} {report['restored_spectral_centroid_hz']:>10.1f}",
            f"  {'HF Energy Ratio (>8kHz)':<30} {report['original_hf_energy_ratio']:>10.4f} {report['restored_hf_energy_ratio']:>10.4f}",
            "-" * 55,
            f"  {'SNR vs Original (dB)':<30} {report['snr_db']:>10.2f}",
            f"  {'PSNR vs Original (dB)':<30} {report['psnr_db']:>10.2f}",
            "=" * 55,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def plot_comparison(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sample_rate: int,
        output_path: str,
        title: str = "Audio Restoration Comparison",
    ):
        """
        Saves a 4-panel comparison figure:
          - Top-left:  Original waveform
          - Top-right: Restored waveform
          - Bottom-left:  Original mel-spectrogram
          - Bottom-right: Restored mel-spectrogram

        Args:
            original: Original audio array.
            restored: Restored audio array.
            sample_rate: Sample rate in Hz.
            output_path: Where to save the PNG figure.
            title: Figure super-title.
        """
        if not _MPL_AVAILABLE:
            logger.warning("matplotlib not available — skipping plot.")
            return
        if not _LIBROSA_AVAILABLE:
            logger.warning("librosa not available — skipping spectrogram plots.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        # --- Waveforms ---
        t_orig = np.linspace(0, len(original) / sample_rate, len(original))
        t_rest = np.linspace(0, len(restored) / sample_rate, len(restored))

        axes[0][0].plot(t_orig, original, color='steelblue', linewidth=0.5)
        axes[0][0].set_title("Original Waveform")
        axes[0][0].set_xlabel("Time (s)")
        axes[0][0].set_ylabel("Amplitude")
        axes[0][0].set_ylim(-1.1, 1.1)

        axes[0][1].plot(t_rest, restored, color='darkorange', linewidth=0.5)
        axes[0][1].set_title("Restored Waveform")
        axes[0][1].set_xlabel("Time (s)")
        axes[0][1].set_ylabel("Amplitude")
        axes[0][1].set_ylim(-1.1, 1.1)

        # --- Mel-spectrograms ---
        hop_length = 512
        n_mels = 128

        mel_orig = librosa.feature.melspectrogram(
            y=original.astype(np.float32), sr=sample_rate,
            n_mels=n_mels, hop_length=hop_length, n_fft=2048,
        )
        mel_rest = librosa.feature.melspectrogram(
            y=restored.astype(np.float32), sr=sample_rate,
            n_mels=n_mels, hop_length=hop_length, n_fft=2048,
        )
        mel_orig_db = librosa.power_to_db(mel_orig, ref=np.max)
        mel_rest_db = librosa.power_to_db(mel_rest, ref=np.max)

        librosa.display.specshow(
            mel_orig_db, sr=sample_rate, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[1][0], cmap='magma',
        )
        axes[1][0].set_title("Original Mel-Spectrogram")

        librosa.display.specshow(
            mel_rest_db, sr=sample_rate, hop_length=hop_length,
            x_axis='time', y_axis='mel', ax=axes[1][1], cmap='magma',
        )
        axes[1][1].set_title("Restored Mel-Spectrogram")

        plt.tight_layout()
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info("Comparison plot saved: %s", output_path)
