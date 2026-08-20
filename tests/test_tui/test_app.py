"""Headless tests for the Textual app shell (no terminal required)."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from audio_restoration.tui import i18n
from audio_restoration.tui.app import AudioRestorationTUI
from audio_restoration.tui.components.sidebar import Sidebar
from audio_restoration.tui.screens import SCREEN_FACTORIES, HomeScreen, TuiScreen


@pytest.mark.asyncio
async def test_app_mounts_with_sidebar_and_home_screen():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one(Sidebar) is not None
        assert app.query_one(HomeScreen) is not None
        assert app.query_one("#welcome-intro", Static) is not None


@pytest.mark.asyncio
async def test_home_action_buttons_navigate():
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#action-batch", Button).focus()
        await pilot.press("enter")
        await pilot.pause()
        assert app._active == "batch"
        from audio_restoration.tui.screens import PlaceholderScreen

        assert app.query_one(PlaceholderScreen) is not None
        assert app.query_one(".screen-title", Static).render() == "Batch"


@pytest.mark.asyncio
async def test_sidebar_navigation_via_enter():
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
async def test_language_toggle_repeints_home_buttons():
    i18n.set_language("en")
    app = AudioRestorationTUI()
    async with app.run_test() as pilot:
        await pilot.pause()
        await pilot.press("l")
        await pilot.pause()
        button = app.query_one("#action-batch", Button)
        assert button.label == "Restaurar una carpeta completa"


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


def test_screen_factories_cover_all_sidebar_sections():
    from audio_restoration.tui.navigation import SCREENS

    for screen_id, _label_key in SCREENS:
        assert screen_id in SCREEN_FACTORIES
        factory = SCREEN_FACTORIES[screen_id]
        from audio_restoration.tui.state import TuiState

        screen = factory(TuiState())
        assert isinstance(screen, TuiScreen)