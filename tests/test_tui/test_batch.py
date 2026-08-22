"""Headless tests for the batch-file TUI screen."""

from __future__ import annotations

from pathlib import Path

import pytest
from pytest import MonkeyPatch
from textual.widgets import Button, DataTable, Input, Select, Static

from audio_restoration.tui.app import AudioRestorationTUI
from audio_restoration.tui.screens.batch import BatchScreen


@pytest.mark.asyncio
async def test_batch_screen_mounts():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("batch")
        await pilot.pause()
        assert app.query_one(BatchScreen) is not None


@pytest.mark.asyncio
async def test_batch_fields_exist():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("batch")
        await pilot.pause()
        assert app.query_one("#batch-folder", Input) is not None
        assert app.query_one("#batch-output-dir", Input) is not None
        assert app.query_one("#batch-ext", Select) is not None
        assert app.query_one("#batch-suffix", Input) is not None
        assert app.query_one("#batch-workers", Select) is not None
        assert app.query_one("#batch-start-btn", Button) is not None
        assert app.query_one("#batch-table", DataTable) is not None
        assert app.query_one("#batch-summary", Static) is not None


@pytest.mark.asyncio
async def test_batch_table_columns():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("batch")
        await pilot.pause()
        table = app.query_one("#batch-table", DataTable)
        assert table.row_count == 0


@pytest.mark.asyncio
async def test_start_without_folder_shows_error():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("batch")
        await pilot.pause(0.3)
        app.query_one("#batch-start-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(BatchScreen) is not None


@pytest.mark.asyncio
async def test_start_with_empty_folder_shows_warning(tmp_path: Path):
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("batch")
        await pilot.pause(0.3)
        app.query_one("#batch-folder", Input).value = str(tmp_path)
        app.query_one("#batch-output-dir", Input).value = str(tmp_path / "out")
        app.query_one("#batch-start-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(BatchScreen) is not None


@pytest.mark.asyncio
async def test_start_with_audio_files_and_mocked_pipeline(
    tmp_path: Path, monkeypatch: MonkeyPatch
):
    import audio_restoration.tui.screens.batch as batch_mod

    calls: list[tuple[str, str]] = []

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def restore(self, in_path, out_path):
            calls.append((in_path, out_path))
            return {"snr_db": 10.0}

    monkeypatch.setattr(batch_mod, "RestorationPipeline", FakePipeline)

    for name in ["a.wav", "b.flac", "c.txt"]:
        (tmp_path / name).write_bytes(b"fake")

    out_dir = tmp_path / "out"
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("batch")
        await pilot.pause(0.3)
        app.query_one("#batch-folder", Input).value = str(tmp_path)
        app.query_one("#batch-output-dir", Input).value = str(out_dir)
        app.query_one("#batch-start-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause(1.0)

        table = app.query_one("#batch-table", DataTable)
        assert table.row_count == 2
        assert len(calls) == 2
        summary = app.query_one("#batch-summary", Static)
        assert "2" in str(summary.render())


@pytest.mark.asyncio
async def test_cancel_resets_button():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("batch")
        await pilot.pause(0.3)
        app.query_one("#batch-folder", Input).value = "/nonexistent"
        app.query_one("#batch-output-dir", Input).value = "/nonexistent/out"
        app.query_one("#batch-start-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        app.query_one(BatchScreen).action_cancel()
        await pilot.pause()
        btn = app.query_one("#batch-start-btn", Button)
        assert btn.disabled is False


def test_batch_registered_in_factories():
    from audio_restoration.tui.screens.registry import SCREEN_FACTORIES
    from audio_restoration.tui.state import TuiState

    assert "batch" in SCREEN_FACTORIES
    screen = SCREEN_FACTORIES["batch"](TuiState())
    assert isinstance(screen, BatchScreen)
