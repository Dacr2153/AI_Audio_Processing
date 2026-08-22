"""Main Textual application for audio-restore TUI."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.theme import Theme
from textual.widgets import Footer

from . import i18n
from .components.footer import CommandBar
from .components.sidebar import ScreenRequested, Sidebar
from .screens import SCREEN_FACTORIES
from .screens.base import TuiScreen
from .state import TuiState

_DARK_THEME = Theme(
    name="audio-dark",
    primary="#E5A24F",
    secondary="#61AFEF",
    warning="#E5A24F",
    error="#E06C75",
    success="#55C7A0",
    accent="#E5A24F",
    foreground="#D8DEE9",
    background="#0B0E12",
    surface="#11161D",
    panel="#11161D",
    boost="#171D25",
    dark=True,
    variables={
        "border": "#252D37",
        "border-active": "#394553",
        "text-primary": "#D8DEE9",
        "text-secondary": "#A5AEBB",
        "text-muted": "#707A88",
        "surface-active": "#171D25",
        "surface-hover": "#1C232D",
        "accent-bright": "#F0B45F",
    },
)

_LIGHT_THEME = Theme(
    name="audio-light",
    primary="#B0651A",
    secondary="#2B6EA8",
    warning="#A8861B",
    error="#B03338",
    success="#2C8F5C",
    accent="#B0651A",
    foreground="#2C2F33",
    background="#F4F1EA",
    surface="#FFFDF8",
    panel="#FFFDF8",
    boost="#E3DDD0",
    dark=False,
    variables={
        "border": "#D5CFC4",
        "border-active": "#B0A898",
        "text-primary": "#2C2F33",
        "text-secondary": "#5A5D62",
        "text-muted": "#8A8D92",
        "surface-active": "#EDE9E0",
        "surface-hover": "#E5E1D8",
        "accent-bright": "#C87520",
    },
)


class AudioRestorationTUI(App):
    """Bilingual, navigable terminal UI for the restoration pipeline."""

    TITLE = "audio-restore"
    SUB_TITLE = i18n.t("header.subtitle")

    CSS = """
    Screen { layout: grid; }

    Sidebar {
        width: 24;
        dock: left;
        background: $surface;
        border-right: solid $border;
    }

    #content {
        height: 1fr;
        padding: 1 2 1 2;
    }
    """

    BINDINGS: ClassVar[
        list[Binding | tuple[str, str] | tuple[str, str, str]]
    ] = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("l", "toggle_language", "Language", show=False),
        Binding("d", "toggle_theme", "Theme", show=False),
        Binding("?", "toggle_help", "Help", show=False),
    ]

    def __init__(self, state: TuiState | None = None) -> None:
        super().__init__()
        self.state = state or TuiState()
        self._active: str = "home"
        self._screen: TuiScreen | None = None

    def compose(self) -> ComposeResult:
        self.screen.sub_title = self.SUB_TITLE
        yield Sidebar()
        with VerticalScroll(id="content"):
            yield Vertical(id="screen-body")
        yield CommandBar(id="command-bar")
        yield Footer()

    async def on_mount(self) -> None:
        self.register_theme(_DARK_THEME)
        self.register_theme(_LIGHT_THEME)
        self.theme = "audio-dark" if self.state.theme == "dark" else "audio-light"
        await self._mount_screen()

    def on_screen_requested(self, message: ScreenRequested) -> None:
        self.navigate_to(message.screen)

    # ------------------------------------------------------------------
    # Screen management
    # ------------------------------------------------------------------

    async def _mount_screen(self) -> None:
        body = self.query_one("#screen-body", Vertical)
        body.remove_children(selector=Vertical)
        self._screen = SCREEN_FACTORIES[self._active](self.state)
        await body.mount(self._screen)
        self.query_one(Sidebar).set_active(self._active)

    def navigate_to(self, name: str) -> None:
        if name == self._active:
            return
        self._active = name
        self.call_after_refresh(self._mount_screen)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_language(self) -> None:
        lang = i18n.toggle_language()
        self.state.language = lang
        self.sub_title = i18n.t("header.subtitle")
        self.notify(i18n.t("lang.es" if lang == "es" else "lang.en"))
        self.refresh_labels()

    def action_toggle_theme(self) -> None:
        dark = self.theme == "audio-dark"
        self.theme = "audio-light" if dark else "audio-dark"
        self.state.theme = "dark" if dark else "light"

    def action_toggle_help(self) -> None:
        # Fase 5 ships a full help overlay.
        self.notify(i18n.t("help.title"))

    def refresh_labels(self) -> None:
        self.query_one(Sidebar).refresh()
        if self._screen is not None:
            self._screen.refresh_labels()