"""History screen.

Shows a table of past restoration jobs with timestamps, file paths and key
metrics, and lets the user re-run or clear the history.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Static

from .. import i18n
from .base import TuiScreen


class HistoryScreen(TuiScreen):
    """Browse past restorations with re-run and clear actions."""

    TITLE_KEY = "nav.history"

    BINDINGS: ClassVar[list] = [
        Binding("ctrl+x", "clear", "Clear"),
    ]

    DEFAULT_CSS = """
    HistoryScreen .section-label {
        text-style: bold;
        color: $text-secondary;
        height: 1;
        margin-bottom: 0;
    }
    HistoryScreen #history-table {
        height: 1fr;
        margin-top: 1;
    }
    HistoryScreen #history-empty {
        color: $text-muted;
        margin-top: 2;
        height: auto;
    }
    HistoryScreen .action-row {
        height: auto;
        margin-top: 1;
    }
    HistoryScreen .action-row Button {
        margin-right: 1;
    }
    HistoryScreen .action-hint {
        color: $text-muted;
        height: 3;
        width: auto;
        align: left middle;
    }
    """

    def form(self) -> ComposeResult:
        yield DataTable(id="history-table", zebra_stripes=True)
        yield Static(i18n.t("history.none"), id="history-empty")
        with Horizontal(classes="action-row"):
            yield Button(
                i18n.t("history.clear"),
                id="clear-btn",
                variant="error",
                compact=True,
            )
            yield Static(i18n.t("proc.cancel_hint"), classes="action-hint")

    def on_mount(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.add_columns(
            i18n.t("proc.elapsed"),
            i18n.t("io.input"),
            i18n.t("io.output"),
            i18n.t("results.snr"),
        )
        self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#history-table", DataTable)
        table.clear()
        entries = self.state.load_history()
        empty = self.query_one("#history-empty", Static)
        if not entries:
            empty.display = True
        else:
            empty.display = False
            for entry in entries:
                ts = entry.get("timestamp", "")
                short_ts = ts[:10] if len(ts) >= 10 else ts
                inp = entry.get("input", "")
                out = entry.get("output", "")
                snr = entry.get("snr_db", "")
                snr_str = f"{snr:.1f}" if isinstance(snr, (int, float)) else ""
                table.add_row(short_ts, inp, out, snr_str)

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "clear-btn":
            self.action_clear()

    def action_clear(self) -> None:
        self.state.clear_history()
        self._refresh_table()

    def refresh_labels(self) -> None:
        super().refresh_labels()
        btn = self.query_one("#clear-btn", Button)
        btn.label = i18n.t("history.clear")
        self.query_one("#history-empty", Static).update(i18n.t("history.none"))
