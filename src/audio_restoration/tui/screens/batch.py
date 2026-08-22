"""Batch-file restoration screen.

Lets the user pick an input/output folder, configure options, and process
all audio files in the folder with a progress table.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Button, DataTable, Input, Select, Static
from textual.widgets._data_table import Coordinate

from ...constants import AUDIO_EXTENSIONS
from ...pipeline import RestorationPipeline
from .. import i18n
from ..components.file_picker import FilePickerScreen
from ..components.form import FieldRow
from .base import TuiScreen


class BatchScreen(TuiScreen):
    """Batch restoration with folder pickers, options and progress table."""

    TITLE_KEY = "nav.batch"

    BINDINGS: ClassVar[list] = [
        Binding("ctrl+c", "cancel", "Cancel"),
    ]

    DEFAULT_CSS = """
    BatchScreen #batch-table {
        height: 1fr;
        margin-top: 1;
    }
    BatchScreen #batch-summary {
        height: auto;
        margin-top: 1;
        padding: 1 2;
        background: $surface;
        border: round $success 50%;
    }
    BatchScreen #batch-controls {
        height: auto;
        margin-top: 1;
    }
    BatchScreen #batch-folder-row,
    BatchScreen #batch-output-row {
        width: 70;
        max-width: 100%;
    }
    BatchScreen Select {
        width: 18;
    }
    BatchScreen #workers-row {
        width: 30;
    }
    """

    def __init__(self, state, **kwargs) -> None:
        super().__init__(state, **kwargs)
        self._busy = False
        self._files: list[Path] = []
        self._succeeded = 0
        self._failed = 0

    def form(self) -> ComposeResult:
        with FieldRow(
            "batch.folder", "batch-folder",
            browse_key="io.browse", browse_id="browse-batch-folder",
            id="batch-folder-row",
        ):
            pass
        with FieldRow(
            "batch.output_dir", "batch-output-dir",
            browse_key="io.browse", browse_id="browse-batch-output",
            id="batch-output-row",
        ):
            pass
        with Horizontal(id="batch-options"):
            yield FieldRow(
                "batch.ext", "batch-ext",
                select=True, id="batch-ext-row",
            )
            yield FieldRow(
                "batch.suffix", "batch-suffix", id="batch-suffix-row",
            )
            yield FieldRow(
                "batch.workers", "batch-workers",
                select=True, id="batch-workers-row",
            )
        with Horizontal(id="batch-controls"):
            yield Button(i18n.t("batch.start"), id="batch-start-btn", variant="primary")
            yield Static(i18n.t("proc.cancel_hint"), id="batch-hint")
        yield DataTable(id="batch-table")
        yield Static("", id="batch-summary")

    def on_mount(self) -> None:
        table = self.query_one("#batch-table", DataTable)
        table.add_columns(
            i18n.t("io.input"),
            i18n.t("batch.ext"),
        )
        ext_select = self.query_one("#batch-ext", Select)
        ext_select.set_options(
            [(ext, ext) for ext in ("wav", "mp3", "flac", "ogg")]
        )
        ext_select.value = self.state.output_ext
        workers_select = self.query_one("#batch-workers", Select)
        workers_select.set_options([(str(n), str(n)) for n in range(1, 5)])
        workers_select.value = str(self.state.workers)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        match event.button.id:
            case "batch-start-btn":
                self.action_start_batch()
            case "browse-batch-folder":
                self._open_picker("batch_folder", folders_only=True)
            case "browse-batch-output":
                self._open_picker("output_dir", folders_only=True)

    def _open_picker(self, target: str, *, folders_only: bool) -> None:
        current = getattr(self.state, target) or str(Path.home())
        picker = FilePickerScreen(
            start_path=Path(current),
            extensions=None,
            folders_only=True,
            title=i18n.t(
                "generic.select_dir" if folders_only else "generic.output_dir"
            ),
        )

        def on_dismiss(path: Path | None) -> None:
            if path is None:
                return
            setattr(self.state, target, str(path))
            field_id = (
                "batch-folder" if target == "batch_folder" else "batch-output-dir"
            )
            self.query_one(f"#{field_id}", Input).value = str(path)

        self.app.push_screen(picker, on_dismiss)

    def action_start_batch(self) -> None:
        if self._busy:
            return

        folder = self.query_one("#batch-folder", Input).value.strip()
        output_dir = self.query_one("#batch-output-dir", Input).value.strip()
        if not folder:
            self.notify(i18n.t("batch.folder"), severity="error")
            return
        if not output_dir:
            self.notify(i18n.t("batch.output_dir"), severity="error")
            return

        if not Path(folder).is_dir():
            self.notify(i18n.t("error.not_found"), severity="error")
            return

        self._files = sorted(
            p for p in Path(folder).iterdir()
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not self._files:
            self.notify(i18n.t("error.batch_empty"), severity="warning")
            return

        table = self.query_one("#batch-table", DataTable)
        table.clear()
        self._succeeded = 0
        self._failed = 0
        for f in self._files:
            table.add_row(f.name, i18n.t("batch.pending"))

        self._busy = True
        btn = self.query_one("#batch-start-btn", Button)
        btn.disabled = True
        btn.label = i18n.t("proc.title")
        self.query_one("#batch-summary", Static).update("")

        ext_val = self.query_one("#batch-ext", Select).value
        ext = str(ext_val) if ext_val else self.state.output_ext
        suffix = self.query_one("#batch-suffix", Input).value.strip()
        self.state.output_ext = ext
        self.state.output_suffix = suffix
        self.state.batch_folder = folder
        self.state.output_dir = output_dir

        self.run_worker(
            self._run_batch, exclusive=True, group="batch"  # type: ignore[arg-type]
        )

    async def _run_batch(self) -> None:
        out_dir = self.state.output_dir or ""
        ext = self.state.output_ext
        suffix = self.state.output_suffix
        pipeline = RestorationPipeline(self.state.config)
        table = self.query_one("#batch-table", DataTable)

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        for i, f in enumerate(self._files):
            worker = asyncio.current_task()
            if hasattr(worker, "cancelled") and worker.cancelled():  # type: ignore[union-attr]
                break

            table.update_cell_at(Coordinate(i, 1), i18n.t("batch.processing"))
            out_name = f"{f.stem}{suffix}{ext}"
            out_path = str(Path(out_dir) / out_name)
            try:
                await asyncio.to_thread(pipeline.restore, str(f), out_path)
                table.update_cell_at(Coordinate(i, 1), i18n.t("batch.ok"))
                self._succeeded += 1
            except OSError:
                table.update_cell_at(Coordinate(i, 1), i18n.t("batch.failed"))
                self._failed += 1

    def on_worker_state_changed(self, event) -> None:
        worker = event.worker
        if worker.name != "_run_batch" or not worker.is_finished:
            return
        self._busy = False
        btn = self.query_one("#batch-start-btn", Button)
        btn.disabled = False
        btn.label = i18n.t("batch.start")

        summary = (
            f"{i18n.t('batch.summary')}: "
            f"{self._succeeded} {i18n.t('batch.succeeded')}, "
            f"{self._failed} {i18n.t('batch.failures')}"
        )
        if worker.error is not None:
            summary += f"\n{worker.error}"
        self.query_one("#batch-summary", Static).update(summary)
        self.notify(i18n.t("batch.complete"), severity="information")

    def action_cancel(self) -> None:
        if self._busy:
            self._busy = False
            btn = self.query_one("#batch-start-btn", Button)
            btn.disabled = False
            btn.label = i18n.t("batch.start")

    def refresh_labels(self) -> None:
        super().refresh_labels()
        self.query_one("#batch-folder-row", FieldRow).refresh_labels()
        self.query_one("#batch-output-row", FieldRow).refresh_labels()
        self.query_one("#batch-ext-row", FieldRow).refresh_labels()
        self.query_one("#batch-suffix-row", FieldRow).refresh_labels()
        self.query_one("#batch-workers-row", FieldRow).refresh_labels()
        btn = self.query_one("#batch-start-btn", Button)
        btn.label = i18n.t("proc.title" if self._busy else "batch.start")
        self.query_one("#batch-hint", Static).update(i18n.t("proc.cancel_hint"))
