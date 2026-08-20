"""Headless tests for the Textual app shell (no terminal required)."""

from __future__ import annotations

import pytest
from textual.widgets import Static

from audio_restoration.tui import i18n
from audio_restoration.tui.app import AudioRestorationTUI
from audio_restoration.tui.components.sidebar import Sidebar


@pytest.mark.asyncio
async def test_app_mounts_with_sidebar_and_placeholder():
    app = AudioRestorationTUI()
    async with app.run_test():
        assert app.query_one(Sidebar) is not None
        placeholder = app.query_one("#screen-placeholder", Static)
        assert "Welcome to" in str(placeholder.render())


@pytest.mark.asyncio
async def test_app_switches_screen_via_sidebar():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#nav-about").focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app._active == "about"


@pytest.mark.asyncio
async def test_language_toggle_updates_sidebar():
    i18n.set_language("en")
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        await pilot.press("l")
        assert i18n.get_language() == "es"
        assert app.state.language == "es"


@pytest.mark.asyncio
async def test_theme_toggle_switches_registered_themes():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        await pilot.press("d")
        assert app.theme == "audio-light"
        await pilot.press("d")
        assert app.theme == "audio-dark"


@pytest.mark.asyncio
async def test_quit_binding():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        await pilot.press("ctrl+q")
        await pilot.pause()