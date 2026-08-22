"""Modal screen for picking a file from the filesystem."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Footer, Input, Label

from .. import i18n

#: Maximum number of results kept in the picker.
_MAX_HISTORY = 20


class FilePicked(Message):
    """Sent when the user confirms a selection."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path


class FilePickerScreen(ModalScreen[Path]):
    """Full-screen modal to navigate the filesystem and pick a file.

    Filters to audio extensions by default (positional ``extensions``); pass
    ``extensions=None`` to allow any file, or ``folders_only=True`` to only
    select directories.
    """

    BINDINGS: ClassVar[list] = [
        Binding("escape", "dismiss", "Cancel"),
        Binding("enter", "confirm", "Confirm"),
    ]

    def __init__(
        self,
        *,
        start_path: Path,
        extensions: set[str] | None,
        folders_only: bool = False,
        title: str | None = None,
    ) -> None:
        super().__init__()
        self.start_path = start_path if start_path.is_dir() else start_path.parent
        self.extensions = extensions
        self.folders_only = folders_only
        self.title_text = title or i18n.t("generic.browse_input")
        self._current: Path | None = None

    def compose(self) -> ComposeResult:
        yield Label(self.title_text, id="picker-title")
        with Vertical(id="picker-body"):
            yield DirectoryTree(self.start_path, id="picker-tree")
            yield Input(id="picker-path", placeholder=str(self.start_path))
        with Vertical(id="picker-actions"):
            yield Button(i18n.t("generic.cancel"), id="picker-cancel")
            yield Button(i18n.t("generic.confirm"), id="picker-confirm", variant="primary")
        yield Footer()

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        if not self.folders_only:
            self._current = event.path
            self._sync_input()

    def on_directory_tree_directory_selected(self, event: DirectoryTree.DirectorySelected) -> None:
        self._current = event.path
        self._sync_input()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        candidate = Path(event.value).expanduser()
        if candidate.exists():
            self._current = candidate
            self.dismiss(candidate)
        else:
            self.notify(f"{i18n.t('error.not_found')}: {candidate}", severity="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "picker-cancel":
            self.dismiss()
        elif event.button.id == "picker-confirm":
            self._confirm_current()

    def action_confirm(self) -> None:
        self._confirm_current()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _confirm_current(self) -> None:
        path = self._current
        if path is None:
            self.notify(i18n.t("generic.select_first"), severity="warning")
            return
        if self.folders_only:
            if not path.is_dir():
                self.notify(i18n.t("generic.select_first"), severity="warning")
                return
        elif self.extensions is not None and path.suffix.lower() not in self.extensions:
            self.notify(i18n.t("error.unsupported_format"), severity="error")
            return
        self.dismiss(path)

    def _sync_input(self) -> None:
        if self._current is not None:
            self.query_one("#picker-path", Input).value = str(self._current)

    def refresh_labels(self) -> None:
        self.query_one("#picker-title", Label).update(self.title_text)
        self.query_one("#picker-cancel", Button).label = i18n.t("generic.cancel")
        self.query_one("#picker-confirm", Button).label = i18n.t("generic.confirm")