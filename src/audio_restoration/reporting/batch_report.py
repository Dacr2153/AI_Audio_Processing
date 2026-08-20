"""Batch report — aggregates per-file quality metrics into summaries.

Used when the pipeline restores a set of files (``--batch`` mode or explicit
list) so the results are easy to review, diff, and share:

* ``metrics_summary.csv`` — one row per file with all metrics.
* Text summary — per-file rows plus aggregate statistics (mean/std).

Depends on pandas; the whole reporting layer degrades gracefully without it
(only CSV writing requires it).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .metrics import QualityMetrics

logger = logging.getLogger(__name__)

try:
    import pandas as pd

    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

#: Metric columns ordered sensibly for review.
_COLUMN_ORDER = [
    "snr_db",
    "psnr_db",
    "original_rms_db",
    "restored_rms_db",
    "original_peak_db",
    "restored_peak_db",
    "original_dynamic_range_db",
    "restored_dynamic_range_db",
    "original_spectral_centroid_hz",
    "restored_spectral_centroid_hz",
    "original_hf_energy_ratio",
    "restored_hf_energy_ratio",
]


class BatchReport:
    """Collect and export metrics for a batch of restored files.

    Usage::

        report = BatchReport()
        report.add_file("song1.wav", metrics_dict)
        report.save("report_dir/", title="Demo")
        print(report.summary_text())
    """

    def __init__(self) -> None:
        self._files: list[tuple[str, dict]] = []

    @property
    def is_pandas_available(self) -> bool:
        """True when pandas is installed (required for CSV export)."""
        return _PANDAS_AVAILABLE

    @staticmethod
    def measure(
        original: np.ndarray,
        restored: np.ndarray,
        sample_rate: int,
    ) -> dict:
        """Convenience: compute a metrics dict for a single file pair."""
        return QualityMetrics().compare(original, restored, sample_rate)

    def add_file(self, name: str, metrics: dict) -> None:
        """Register one file's metrics under *name* (basename or path)."""
        self._files.append((os.path.basename(str(name)), dict(metrics)))

    def to_dataframe(self) -> pd.DataFrame:
        """Return all rows as a pandas DataFrame."""
        if not _PANDAS_AVAILABLE:
            raise RuntimeError(
                "pandas is required for BatchReport.to_dataframe / CSV export. "
                "Install it with: pip install pandas"
            )
        frame = pd.DataFrame([row for _name, row in self._files])
        frame.insert(0, "file", [name for name, _row in self._files])
        return frame

    def summary_text(self) -> str:
        """Render a readable text summary of the batch results."""
        if not self._files:
            return "No files recorded."

        full = self.to_dataframe()
        rows = _COLUMN_ORDER
        show = full[["file"] + rows].copy()

        header = f"{'file':<40}" + "".join(f"{c:>22}" for c in rows)
        lines = [header, "-" * len(header)]
        for _, row in show.iterrows():
            cells = [f"{row['file']:<40}"]
            cells += [
                f"{row[c]:>22.2f}"
                if isinstance(row[c], (int, float))
                else f"{row[c]:>22}"
                for c in rows
            ]
            lines.append("".join(cells))

        lines.append("-" * len(header))
        agg = full[rows].agg(["mean", "std"]) if _PANDAS_AVAILABLE else None
        if agg is not None:
            for stat in ("mean", "std"):
                cells = [f"{stat.capitalize():<40}"]
                cells += [f"{agg.loc[stat, c]:>22.2f}" for c in rows]
                lines.append("".join(cells))
        return "\n".join(lines)

    def save(
        self,
        output_dir: str,
        filename: str = "metrics_summary.csv",
        title: str = "Audio Restoration Batch Report",
    ) -> str:
        """Write the batch metrics to ``output_dir/filename`` and return the path."""
        if not self._files:
            raise ValueError("No files recorded — nothing to save.")

        if not _PANDAS_AVAILABLE:
            raise RuntimeError(
                "pandas is required to export the CSV report. "
                "Install it with: pip install pandas"
            )

        os.makedirs(output_dir, exist_ok=True)
        frame = self.to_dataframe()
        if title:
            frame.attrs["title"] = title
        out_path = os.path.join(output_dir, filename)
        frame.to_csv(out_path, index=False)
        logger.info("Batch report saved: %s (%d rows)", out_path, len(frame))
        return out_path
