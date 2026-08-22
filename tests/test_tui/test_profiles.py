"""Headless tests for the profiles TUI screen."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch
from textual.widgets import Button, DataTable, Input, Static

from audio_restoration.tui.app import AudioRestorationTUI
from audio_restoration.tui.screens.profiles import ProfilesScreen


@pytest.fixture(autouse=True)
def _isolated_profiles(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Redirect profile storage to a temp directory for every test."""
    import audio_restoration.tui.state as state_mod

    monkeypatch.setattr(state_mod, "_profiles_file", lambda: tmp_path / "profiles.json")
    monkeypatch.setattr(state_mod, "_history_file", lambda: tmp_path / "history.json")


@pytest.mark.asyncio
async def test_profiles_screen_mounts():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("profiles")
        await pilot.pause()
        assert app.query_one(ProfilesScreen) is not None


@pytest.mark.asyncio
async def test_profiles_fields_exist():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("profiles")
        await pilot.pause()
        assert app.query_one("#profiles-table", DataTable) is not None
        assert app.query_one("#profiles-empty", Static) is not None
        assert app.query_one("#profile-name", Input) is not None
        assert app.query_one("#save-btn", Button) is not None


@pytest.mark.asyncio
async def test_profiles_empty_initially():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("profiles")
        await pilot.pause()
        table = app.query_one("#profiles-table", DataTable)
        assert table.row_count == 0
        empty = app.query_one("#profiles-empty", Static)
        assert empty.display is True


@pytest.mark.asyncio
async def test_save_profile():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("profiles")
        await pilot.pause()
        app.query_one("#profile-name", Input).value = "test-profile"
        app.query_one("#save-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        table = app.query_one("#profiles-table", DataTable)
        assert table.row_count == 1
        name = table.get_cell_at((0, 0))
        assert name == "test-profile"


@pytest.mark.asyncio
async def test_save_empty_name_shows_warning():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("profiles")
        await pilot.pause()
        app.query_one("#save-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        table = app.query_one("#profiles-table", DataTable)
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_load_profile_cell_label():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("profiles")
        await pilot.pause()
        app.query_one("#profile-name", Input).value = "my-profile"
        app.query_one("#save-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        table = app.query_one("#profiles-table", DataTable)
        assert table.row_count == 1
        cell = table.get_cell_at((0, 1))
        assert cell == "Load"


@pytest.mark.asyncio
async def test_delete_profile_cell_label():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("profiles")
        await pilot.pause()
        app.query_one("#profile-name", Input).value = "del-me"
        app.query_one("#save-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        table = app.query_one("#profiles-table", DataTable)
        assert table.row_count == 1
        cell = table.get_cell_at((0, 2))
        assert cell == "Delete"


@pytest.mark.asyncio
async def test_registered_in_factories():
    from audio_restoration.tui.screens.registry import SCREEN_FACTORIES
    from audio_restoration.tui.state import TuiState

    assert "profiles" in SCREEN_FACTORIES
    screen = SCREEN_FACTORIES["profiles"](TuiState())
    assert isinstance(screen, ProfilesScreen)
