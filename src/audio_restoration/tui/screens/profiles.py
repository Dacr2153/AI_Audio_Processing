"""Profiles management screen.

Lets the user view, load, delete and save pipeline-configuration profiles.
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Input, Static
from textual.widgets._data_table import Coordinate

from .. import i18n
from ..components.form import FieldRow
from .base import TuiScreen


class ProfilesScreen(TuiScreen):
    """Profile list with load / delete / save actions."""

    TITLE_KEY = "nav.profiles"

    BINDINGS: ClassVar[list] = [
        Binding("ctrl+s", "save", "Save"),
    ]

    DEFAULT_CSS = """
    ProfilesScreen .section-label {
        text-style: bold;
        color: $text-secondary;
        height: 1;
        margin-bottom: 0;
    }
    ProfilesScreen #profiles-table {
        height: 1fr;
        margin-top: 1;
    }
    ProfilesScreen #profiles-empty {
        color: $text-muted;
        margin-top: 2;
        height: auto;
    }
    ProfilesScreen .save-panel {
        background: $surface;
        border: solid $border;
        padding: 0 1;
        margin-top: 1;
        height: auto;
    }
    ProfilesScreen .save-panel Horizontal {
        height: auto;
    }
    ProfilesScreen .save-panel Button {
        margin: 0 0 0 1;
    }
    """

    def __init__(self, state, **kwargs) -> None:
        super().__init__(state, **kwargs)

    def form(self) -> ComposeResult:
        yield DataTable(id="profiles-table", zebra_stripes=True)
        yield Static(i18n.t("profiles.none"), id="profiles-empty")
        with Vertical(classes="save-panel"):
            yield Static(i18n.t("config.save_profile"), classes="section-label")
            with Horizontal():
                yield FieldRow(
                    None, "profile-name",
                    id="profile-name-row",
                )
                yield Button(
                    i18n.t("config.save_profile"),
                    id="save-btn",
                    variant="primary",
                    compact=True,
                )

    def on_mount(self) -> None:
        table = self.query_one("#profiles-table", DataTable)
        table.add_columns(
            i18n.t("config.profile_name"),
            "",
            "",
        )
        self._refresh_table()

    def _refresh_table(self) -> None:
        table = self.query_one("#profiles-table", DataTable)
        table.clear()
        profiles = self.state.list_profiles()
        empty = self.query_one("#profiles-empty", Static)
        if not profiles:
            empty.display = True
        else:
            empty.display = False
            for name in profiles:
                table.add_row(
                    name,
                    i18n.t("profiles.load"),
                    i18n.t("profiles.delete"),
                )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            self.action_save()

    def on_data_table_cell_selected(self, event: DataTable.CellSelected) -> None:
        col = event.coordinate.column
        row = event.coordinate.row
        if col == 1:
            self._load_profile(row)
        elif col == 2:
            self._delete_profile(row)

    def _load_profile(self, row: int) -> None:
        table = self.query_one("#profiles-table", DataTable)
        name = table.get_cell_at(Coordinate(row, 0))
        self.state.apply_profile(name)
        self.notify(f"{i18n.t('profiles.saved')} {name}", severity="information")

    def _delete_profile(self, row: int) -> None:
        table = self.query_one("#profiles-table", DataTable)
        name = table.get_cell_at(Coordinate(row, 0))
        self.state.delete_profile(name)
        self._refresh_table()

    def action_save(self) -> None:
        name = self.query_one("#profile-name", Input).value.strip()
        if not name:
            self.notify(i18n.t("profiles.empty_name"), severity="warning")
            return
        self.state.save_profile(name)
        self.query_one("#profile-name", Input).value = ""
        self._refresh_table()
        self.notify(f"{i18n.t('profiles.saved')} {name}", severity="information")

    def refresh_labels(self) -> None:
        super().refresh_labels()
        self.query_one("#profile-name-row", FieldRow).refresh_labels()
        btn = self.query_one("#save-btn", Button)
        btn.label = i18n.t("config.save_profile")
        self.query_one("#profiles-empty", Static).update(i18n.t("profiles.none"))
