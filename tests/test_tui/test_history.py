"""Headless tests for the history TUI screen."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pytest import MonkeyPatch
from textual.widgets import Button, DataTable, Static

from audio_restoration.tui.app import AudioRestorationTUI
from audio_restoration.tui.screens.history import HistoryScreen


@pytest.fixture(autouse=True)
def _isolated_history(tmp_path: Path, monkeypatch: MonkeyPatch):
    """Redirect history storage to a temp directory for every test."""
    import audio_restoration.tui.state as state_mod

    monkeypatch.setattr(state_mod, "_history_file", lambda: tmp_path / "history.json")
    monkeypatch.setattr(state_mod, "_profiles_file", lambda: tmp_path / "profiles.json")


@pytest.mark.asyncio
async def test_history_screen_mounts():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("history")
        await pilot.pause()
        assert app.query_one(HistoryScreen) is not None


@pytest.mark.asyncio
async def test_history_fields_exist():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("history")
        await pilot.pause()
        assert app.query_one("#history-table", DataTable) is not None
        assert app.query_one("#history-empty", Static) is not None
        assert app.query_one("#clear-btn", Button) is not None


@pytest.mark.asyncio
async def test_history_empty_initially():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("history")
        await pilot.pause()
        table = app.query_one("#history-table", DataTable)
        assert table.row_count == 0
        empty = app.query_one("#history-empty", Static)
        assert empty.display is True


@pytest.mark.asyncio
async def test_history_shows_entries(tmp_path: Path):
    history_file = tmp_path / "history.json"
    entry = {
        "input": "/tmp/in.wav",
        "output": "/tmp/out.wav",
        "timestamp": "2026-08-22T10:30:00+02:00",
        "snr_db": 15.3,
        "psnr_db": 20.1,
        "original_rms_db": -18.0,
        "restored_rms_db": -14.5,
    }
    history_file.write_text(json.dumps([entry]))

    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("history")
        await pilot.pause()
        table = app.query_one("#history-table", DataTable)
        assert table.row_count == 1
        assert table.get_cell_at((0, 0)) == "2026-08-22"
        assert table.get_cell_at((0, 1)) == "/tmp/in.wav"
        assert table.get_cell_at((0, 2)) == "/tmp/out.wav"
        assert table.get_cell_at((0, 3)) == "15.3"


@pytest.mark.asyncio
async def test_clear_history(tmp_path: Path):
    history_file = tmp_path / "history.json"
    history_file.write_text(json.dumps([{"input": "a.wav", "output": "b.wav", "timestamp": "2026-01-01T00:00:00"}]))

    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("history")
        await pilot.pause()
        table = app.query_one("#history-table", DataTable)
        assert table.row_count == 1
        app.query_one("#clear-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert table.row_count == 0
        assert app.query_one("#history-empty", Static).display is True


@pytest.mark.asyncio
async def test_registered_in_factories():
    from audio_restoration.tui.screens.registry import SCREEN_FACTORIES
    from audio_restoration.tui.state import TuiState

    assert "history" in SCREEN_FACTORIES
    screen = SCREEN_FACTORIES["history"](TuiState())
    assert isinstance(screen, HistoryScreen)
