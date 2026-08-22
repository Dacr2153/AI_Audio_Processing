"""Single-file restoration screen.

Lets the user pick an input file and output path, run the pipeline and
view the quality-metrics report.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Static

from ...pipeline import RestorationPipeline
from .. import i18n
from ..components.file_picker import FilePickerScreen
from ..components.form import FieldRow
from ..components.results import ResultsPanel
from .base import TuiScreen

_AUDIO_EXTENSIONS = {
    ".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".aac",
    ".ogg", ".opus", ".wma", ".mp4",
}


class SingleScreen(TuiScreen):
    """Single-file restoration with file pickers, run and results."""

    TITLE_KEY = "nav.single"

    BINDINGS: ClassVar[list] = [
        Binding("ctrl+r", "run", "Run"),
    ]

    DEFAULT_CSS = """
    SingleScreen .section-label {
        text-style: bold;
        color: $text-secondary;
        height: 1;
        margin-bottom: 0;
    }
    SingleScreen .file-panel {
        background: $surface;
        border: solid $border;
        padding: 0 1;
        margin-bottom: 1;
        height: auto;
    }
    SingleScreen .action-row {
        height: auto;
        margin-top: 1;
        margin-bottom: 1;
    }
    SingleScreen .action-row Button {
        margin-right: 1;
    }
    SingleScreen .action-hint {
        color: $text-muted;
        height: 3;
        width: auto;
        align: left middle;
    }
    """

    def __init__(self, state, **kwargs) -> None:
        super().__init__(state, **kwargs)
        self._busy = False
        self._input_path = ""
        self._output_path = ""

    def form(self) -> ComposeResult:
        # Input file panel
        yield Static(i18n.t("io.input"), classes="section-label")
        with Vertical(classes="file-panel"):
            yield FieldRow(
                None, "input-path",
                browse_key="io.browse", browse_id="browse-input",
                id="row-input",
            )

        # Output file panel
        yield Static(i18n.t("io.output"), classes="section-label")
        with Vertical(classes="file-panel"):
            yield FieldRow(
                None, "output-path",
                browse_key="io.browse", browse_id="browse-output",
                id="row-output",
            )

        # Action row
        with Horizontal(classes="action-row"):
            yield Button(
                i18n.t("config.run_single"),
                id="run-btn",
                variant="primary",
                compact=True,
            )
            yield Static(i18n.t("proc.cancel_hint"), classes="action-hint")

        # Results
        yield ResultsPanel(id="results")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-btn":
            self.action_run()
        elif event.button.id == "browse-input":
            self._open_picker(folders_only=False)
        elif event.button.id == "browse-output":
            self._open_picker(folders_only=True)

    def _open_picker(self, folders_only: bool) -> None:
        start = Path(self.state.input_path or str(Path.home()))
        picker = FilePickerScreen(
            start_path=start,
            extensions=None if folders_only else _AUDIO_EXTENSIONS,
            folders_only=folders_only,
            title=i18n.t(
                "generic.select_dir" if folders_only else "generic.select_input"
            ),
        )

        def on_dismiss(path: Path | None) -> None:
            if path is None:
                return
            if folders_only:
                self.state.output_path = str(path)
                self.query_one("#output-path", Input).value = str(path)
            else:
                self.state.input_path = str(path)
                self.query_one("#input-path", Input).value = str(path)

        self.app.push_screen(picker, on_dismiss)

    def action_run(self) -> None:
        if self._busy:
            return
        self._input_path = self.query_one("#input-path", Input).value.strip()
        self._output_path = self.query_one("#output-path", Input).value.strip()
        if not self._input_path:
            self.notify(i18n.t("io.input"), severity="error")
            return
        if not self._output_path:
            self.notify(i18n.t("io.output"), severity="error")
            return
        inp = Path(self._input_path)
        if not inp.is_file():
            self.notify(i18n.t("error.not_found"), severity="error")
            return
        if inp.suffix.lower() not in _AUDIO_EXTENSIONS:
            self.notify(i18n.t("error.unsupported_format"), severity="error")
            return
        out = Path(self._output_path)
        if out.is_dir():
            ext = self.state.output_ext or inp.suffix.lstrip(".")
            suffix = self.state.output_suffix
            out = out / f"{inp.stem}{suffix}.{ext}"
            self._output_path = str(out)
            self.query_one("#output-path", Input).value = self._output_path
        self._busy = True
        btn = self.query_one("#run-btn", Button)
        btn.disabled = True
        btn.label = i18n.t("proc.title")
        self.run_worker(self._restore, exclusive=True, group="restore")  # type: ignore[arg-type]

    async def _restore(self) -> dict:
        return RestorationPipeline(self.state.config).restore(
            self._input_path, self._output_path
        )

    def on_worker_state_changed(self, event) -> None:
        worker = event.worker
        if worker.name != "_restore" or not worker.is_finished:
            return
        results = self.query_one("#results", ResultsPanel)
        if worker.error is not None:
            results.show_error(str(worker.error))
            self.notify(i18n.t("error.pipeline_failed"), severity="error")
        else:
            report = worker.result
            self.state.add_history_entry(
                {
                    "input": self._input_path,
                    "output": self._output_path,
                    **{
                        k: v
                        for k, v in report.items()
                        if isinstance(v, (int, float))
                    },
                }
            )
            results.show_report(report)
            self.notify(i18n.t("proc.done"), severity="information")
        self._busy = False
        btn = self.query_one("#run-btn", Button)
        btn.disabled = False
        btn.label = i18n.t("config.run_single")

    def refresh_labels(self) -> None:
        super().refresh_labels()
        self.query_one("#row-input", FieldRow).refresh_labels()
        self.query_one("#row-output", FieldRow).refresh_labels()
        btn = self.query_one("#run-btn", Button)
        btn.label = i18n.t("proc.title" if self._busy else "config.run_single")
