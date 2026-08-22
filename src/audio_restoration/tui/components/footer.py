"""Command bar footer for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Static


class CommandBar(Static):
    """Bottom command bar showing contextual shortcuts."""

    DEFAULT_CSS = """
    CommandBar {
        dock: bottom;
        height: 1;
        padding: 0 2;
        background: $surface;
        border-top: solid $border;
        color: $text-muted;
        content-align: left middle;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._bar_mode: str = "default"

    def compose(self) -> ComposeResult:
        yield Static(self._bar_text(), id="bar-content")

    def _bar_text(self) -> str:
        if self._bar_mode == "processing":
            return "  Ctrl+S Stop"
        return "  Enter Select    Tab Navigate    Esc Back    ? Help"

    def set_processing(self, processing: bool) -> None:
        self._bar_mode = "processing" if processing else "default"
        try:
            self.query_one("#bar-content", Static).update(self._bar_text())
        except Exception:  # noqa: BLE001, S110
            pass

    def refresh_labels(self) -> None:
        try:
            self.query_one("#bar-content", Static).update(self._bar_text())
        except Exception:  # noqa: BLE001, S110
            pass
