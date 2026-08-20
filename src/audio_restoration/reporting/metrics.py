"""Quality metrics — Phase 6 of the restoration pipeline.

Provides objective and perceptual-ish measures for before/after comparison:

* SNR / PSNR
* RMS, peak and dynamic range (in dBFS)
* Spectral centroid (brightness)
* High-frequency energy ratio (> threshold)
* Numpy-FFT based HF power instead of periodogram (faster, vectorised)

Matplotlib/librosa plotting is optional. Metrics operate identically on mono
``(N,)`` and stereo ``(N, 2)`` input (stereo is averaged across channels).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False
    logger.info("matplotlib not found — visualizations disabled.")

try:
    import librosa
    import librosa.display

    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Average multi-channel input to a mono array (no-op for mono)."""
    x = np.asarray(audio, dtype=np.float64)
    return x.mean(axis=1) if x.ndim == 2 else x


class QualityMetrics:
    """Computes and visualizes audio quality metrics.

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
        """SNR in dB, treating ``reference - noisy`` as noise."""
        ref = _to_mono(reference)
        nsy = _to_mono(noisy)
        n = min(len(ref), len(nsy))
        ref, nsy = ref[:n], nsy[:n]

        signal_power = float(np.mean(ref**2))
        noise_power = float(np.mean((ref - nsy) ** 2))
        if noise_power < 1e-14:
            return 120.0
        if signal_power < 1e-14:
            return 0.0
        return float(10.0 * np.log10(signal_power / noise_power))

    @staticmethod
    def psnr(reference: np.ndarray, processed: np.ndarray) -> float:
        """Peak Signal-to-Noise ratio in dB."""
        ref = _to_mono(reference)
        proc = _to_mono(processed)
        n = min(len(ref), len(proc))
        ref, proc = ref[:n], proc[:n]

        mse = float(np.mean((ref - proc) ** 2))
        if mse < 1e-14:
            return 120.0
        max_val = max(float(np.max(np.abs(ref))), 1e-9)
        return float(20.0 * np.log10(max_val / np.sqrt(mse)))

    @staticmethod
    def rms_db(audio: np.ndarray) -> float:
        """RMS level in dBFS."""
        x = _to_mono(audio)
        rms = float(np.sqrt(np.mean(x**2)))
        if rms < 1e-12:
            return -120.0
        return float(20.0 * np.log10(rms))

    @staticmethod
    def peak_db(audio: np.ndarray) -> float:
        """Peak level in dBFS."""
        x = np.asarray(audio, dtype=np.float64)
        peak = float(np.max(np.abs(x)))
        if peak < 1e-12:
            return -120.0
        return float(20.0 * np.log10(peak))

    @staticmethod
    def dynamic_range_db(audio: np.ndarray) -> float:
        """Crest factor measure (peak - RMS)."""
        return QualityMetrics.peak_db(audio) - QualityMetrics.rms_db(audio)

    @staticmethod
    def spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
        """Mean spectral centroid in Hz — a measure of audio brightness."""
        if not _LIBROSA_AVAILABLE:
            return 0.0
        x = _to_mono(audio)
        centroid = librosa.feature.spectral_centroid(
            y=x.astype(np.float32), sr=sample_rate
        )
        return float(np.mean(centroid))

    @staticmethod
    def high_freq_energy_ratio(
        audio: np.ndarray, sample_rate: int, threshold_hz: float = 8000.0
    ) -> float:
        """Fraction of total power above *threshold_hz* (DFT-based)."""
        x = _to_mono(audio)
        n = len(x)
        if n < 2:
            return 0.0
        spectrum = np.fft.rfft(x)
        freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)
        power = np.abs(spectrum) ** 2.0
        total = float(np.sum(power))
        if total < 1e-30:
            return 0.0
        hf = float(np.sum(power[freqs >= threshold_hz]))
        return float(np.clip(hf / total, 0.0, 1.0))

    def compare(
        self,
        original: np.ndarray,
        restored: np.ndarray,
        sample_rate: int,
    ) -> dict[str, float]:
        """Compute a full set of before/after metrics for a file."""
        return {
            "original_rms_db": self.rms_db(original),
            "restored_rms_db": self.rms_db(restored),
            "original_peak_db": self.peak_db(original),
            "restored_peak_db": self.peak_db(restored),
            "original_dynamic_range_db": self.dynamic_range_db(original),
            "restored_dynamic_range_db": self.dynamic_range_db(restored),
            "original_spectral_centroid_hz": self.spectral_centroid(
                original, sample_rate
            ),
            "restored_spectral_centroid_hz": self.spectral_centroid(
                restored, sample_rate
            ),
            "original_hf_energy_ratio": self.high_freq_energy_ratio(
                original, sample_rate
            ),
            "restored_hf_energy_ratio": self.high_freq_energy_ratio(
                restored, sample_rate
            ),
            "snr_db": self.snr(restored, original),
            "psnr_db": self.psnr(restored, original),
        }

    @staticmethod
    def format_report(report: dict) -> str:
        """Format a metrics report dict as a readable text summary."""
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
        """Save a 4-panel before/after figure (waveforms + mel-spectrograms)."""
        if not (_MPL_AVAILABLE and _LIBROSA_AVAILABLE):
            logger.warning("matplotlib/librosa not available — skipping plot.")
            return

        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(title, fontsize=14, fontweight="bold")

        t_orig = np.linspace(0, len(original) / sample_rate, len(original))
        t_rest = np.linspace(0, len(restored) / sample_rate, len(restored))

        axes[0][0].plot(t_orig, _to_mono(original), color="steelblue", linewidth=0.5)
        axes[0][0].set_title("Original Waveform")
        axes[0][0].set_xlabel("Time (s)")
        axes[0][0].set_ylabel("Amplitude")
        axes[0][0].set_ylim(-1.1, 1.1)

        axes[0][1].plot(t_rest, _to_mono(restored), color="darkorange", linewidth=0.5)
        axes[0][1].set_title("Restored Waveform")
        axes[0][1].set_xlabel("Time (s)")
        axes[0][1].set_ylabel("Amplitude")
        axes[0][1].set_ylim(-1.1, 1.1)

        hop_length = 512
        n_mels = 128

        mel_orig = librosa.feature.melspectrogram(
            y=_to_mono(original).astype(np.float32),
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=2048,
        )
        mel_rest = librosa.feature.melspectrogram(
            y=_to_mono(restored).astype(np.float32),
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=2048,
        )

        librosa.display.specshow(
            librosa.power_to_db(mel_orig, ref=np.max),
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=axes[1][0],
            cmap="magma",
        )
        axes[1][0].set_title("Original Mel-Spectrogram")

        librosa.display.specshow(
            librosa.power_to_db(mel_rest, ref=np.max),
            sr=sample_rate,
            hop_length=hop_length,
            x_axis="time",
            y_axis="mel",
            ax=axes[1][1],
            cmap="magma",
        )
        axes[1][1].set_title("Restored Mel-Spectrogram")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Comparison plot saved: %s", output_path)
