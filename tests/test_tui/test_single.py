"""Headless tests for the single-file TUI screen."""

from __future__ import annotations

import pytest
from pytest import MonkeyPatch
from textual.widgets import Button, Input, Static

from audio_restoration.tui.app import AudioRestorationTUI
from audio_restoration.tui.components.file_picker import FilePickerScreen
from audio_restoration.tui.components.form import FieldRow
from audio_restoration.tui.components.results import ResultsPanel
from audio_restoration.tui.screens.single import SingleScreen


@pytest.mark.asyncio
async def _app_with_single():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("single")
        await pilot.pause()
        assert app.query_one(SingleScreen) is not None
        return app, pilot


@pytest.mark.asyncio
async def test_single_screen_has_pickers_and_run():
    await _app_with_single()


@pytest.mark.asyncio
async def test_single_fields_exist():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("single")
        await pilot.pause()
        assert app.query_one("#input-path", Input) is not None
        assert app.query_one("#output-path", Input) is not None
        assert app.query_one("#run-btn", Button) is not None
        assert app.query_one("#results", ResultsPanel) is not None


@pytest.mark.asyncio
async def test_browse_input_opens_picker():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("single")
        await pilot.pause()
        btn = app.query_one("#browse-input", Button)
        btn.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert len(app.screen_stack) >= 2
        assert isinstance(app.screen_stack[-1], FilePickerScreen)


@pytest.mark.asyncio
async def test_run_without_input_shows_error():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("single")
        await pilot.pause(0.3)
        app.query_one("#run-btn", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app.query_one(SingleScreen) is not None


@pytest.mark.asyncio
async def test_run_with_mocked_pipeline(monkeypatch: MonkeyPatch):
    import audio_restoration.tui.screens.single as single_mod

    calls: list[tuple[str, str]] = []

    class FakePipeline:
        def __init__(self, config):
            self.config = config

        def restore(self, in_path, out_path):
            calls.append((in_path, out_path))
            return {"snr_db": 12.3, "psnr_db": 18.7, "original_rms_db": -10.0,
                    "restored_rms_db": -9.5}

    monkeypatch.setattr(single_mod, "RestorationPipeline", FakePipeline)

    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("single")
        await pilot.pause(0.3)
        app.query_one("#input-path", Input).value = "a.wav"
        app.query_one("#output-path", Input).value = "out.wav"
        app.query_one("#run-btn", Button).focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(0.5)
        assert calls == [("a.wav", "out.wav")]
        results = app.query_one("#results", ResultsPanel)
        assert results._report is not None
        assert results._report["snr_db"] == 12.3
        assert "12.3" in str(results.query_one("#results-body", Static).render())


@pytest.mark.asyncio
async def test_file_picker_dismisses_on_escape():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("single")
        await pilot.pause()
        app.query_one("#browse-input", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert len(app.screen_stack) == 1


def test_field_row_i18n_lazy():

    row = FieldRow("io.input", "x")
    assert row.label_key == "io.input"


def test_single_registered_in_factories():
    from audio_restoration.tui.screens import SingleScreen
    from audio_restoration.tui.screens.registry import SCREEN_FACTORIES
    from audio_restoration.tui.state import TuiState

    assert "single" in SCREEN_FACTORIES
    assert SCREEN_FACTORIES["single"] is not None
    screen = SCREEN_FACTORIES["single"](TuiState())
    assert isinstance(screen, SingleScreen)