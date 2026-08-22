"""Metrics display for a single restored file."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from .. import i18n

#: Metric rows shown in the results panel: (i18n label key, report key, format).
_METRICS = (
    ("results.snr", "snr_db", "{:.1f}"),
    ("results.psnr", "psnr_db", "{:.1f}"),
    ("results.original_rms", "original_rms_db", "{:.1f}"),
    ("results.restored_rms", "restored_rms_db", "{:.1f}"),
)


class ResultsPanel(Vertical):
    """Renders the metrics report returned by :meth:`pipeline.restore`.

    The panel keeps the raw report so it can re-render after a language change.
    """

    DEFAULT_CSS = """
    ResultsPanel {
        border: round $success 50%;
        padding: 1 2;
        margin-top: 1;
        background: $surface;
        height: auto;
        width: 60;
        max-width: 100%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._report: dict[str, float] | None = None
        self._error: str | None = None

    def compose(self) -> ComposeResult:
        yield Static(i18n.t("results.hint"), id="results-body")

    def show_report(self, report: dict[str, float]) -> None:
        """Render a successful metrics dict."""
        self._report = report
        self._error = None
        lines = [f"[b]{i18n.t('results.title')}[/b]:", ""]
        for label_key, metric_key, fmt in _METRICS:
            value = report.get(metric_key, 0.0)
            lines.append(f"  {i18n.t(label_key)}: {fmt.format(value)}")
        self._render_text("\n".join(lines))

    def show_error(self, message: str) -> None:
        """Render a failed run."""
        self._report = None
        self._error = message
        self._render_text(f"[b][red]{i18n.t('error.pipeline_failed')}[/red][/b]\n{message}")

    def _render_text(self, text: str) -> None:
        self.query_one("#results-body", Static).update(text)

    def refresh_labels(self) -> None:
        if self._error is not None:
            self.show_error(self._error)
        elif self._report is not None:
            self.show_report(self._report)