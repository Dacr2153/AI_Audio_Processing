"""Headless tests for the about TUI screen."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from audio_restoration import __version__
from audio_restoration.tui.app import AudioRestorationTUI
from audio_restoration.tui.screens.about import AboutScreen


@pytest.mark.asyncio
async def test_about_screen_mounts():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("about")
        await pilot.pause()
        assert app.query_one(AboutScreen) is not None


@pytest.mark.asyncio
async def test_about_version_shown():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("about")
        await pilot.pause()
        version = app.query_one("#about-version", Static)
        assert __version__ in str(version.render())


@pytest.mark.asyncio
async def test_about_deps_shown():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("about")
        await pilot.pause()
        assert app.query_one("#about-deepfilter", Static) is not None
        assert app.query_one("#about-demucs", Static) is not None
        assert app.query_one("#about-audiosr", Static) is not None


@pytest.mark.asyncio
async def test_about_license_shown():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("about")
        await pilot.pause()
        license_widget = app.query_one("#about-license", Static)
        assert "MIT" in str(license_widget.render())


@pytest.mark.asyncio
async def test_about_repo_shown():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        app.navigate_to("about")
        await pilot.pause()
        repo = app.query_one("#about-repo", Static)
        assert "github.com" in str(repo.render())


@pytest.mark.asyncio
async def test_registered_in_factories():
    from audio_restoration.tui.screens.registry import SCREEN_FACTORIES
    from audio_restoration.tui.state import TuiState

    assert "about" in SCREEN_FACTORIES
    screen = SCREEN_FACTORIES["about"](TuiState())
    assert isinstance(screen, AboutScreen)
