"""Main Textual application for audio-restore TUI."""

from __future__ import annotations

from typing import ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.theme import Theme
from textual.widgets import Footer, Static

from . import i18n
from .components.sidebar import ScreenRequested, Sidebar
from .state import TuiState

_DARK_THEME = Theme(
    name="audio-dark",
    primary="#e0a458",
    secondary="#58a6e0",
    warning="#e0c058",
    error="#e05558",
    success="#56c98a",
    accent="#58a6e0",
    foreground="#d7dee8",
    background="#0c0e12",
    surface="#161b23",
    panel="#11141a",
    boost="#1d2430",
    dark=True,
)

_LIGHT_THEME = Theme(
    name="audio-light",
    primary="#b0651a",
    secondary="#2b6ea8",
    warning="#a8861b",
    error="#b03338",
    success="#2c8f5c",
    accent="#2b6ea8",
    foreground="#2c2f33",
    background="#f4f1ea",
    surface="#fffdf8",
    panel="#ebe6dc",
    boost="#e3ddd0",
    dark=False,
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
        background: $panel;
        border-right: solid $background-lighten-1;
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

    def on_mount(self) -> None:
        self.register_theme(_DARK_THEME)
        self.register_theme(_LIGHT_THEME)
        self.theme = "audio-dark" if self.state.theme == "dark" else "audio-light"

    def compose(self) -> ComposeResult:
        self.screen.sub_title = self.SUB_TITLE
        yield Sidebar()
        with VerticalScroll(id="content"):
            yield Static(self._welcome_text(), id="screen-placeholder")
        yield Footer()

    def on_screen_requested(self, message: ScreenRequested) -> None:
        self.navigate_to(message.screen)

    # ------------------------------------------------------------------
    # Screen management
    # ------------------------------------------------------------------

    def navigate_to(self, name: str) -> None:
        if name == self._active:
            return
        self._active = name
        placeholder = self.query_one("#screen-placeholder", Static)
        placeholder.update(self._screen_for(name))
        self.query_one(Sidebar).refresh()

    def _screen_for(self, name: str) -> str:
        return f"[b]{i18n.t(f'nav.{name}')}[/b]\n\n{i18n.t('welcome.tip')}"

    def _welcome_text(self) -> str:
        return (
            f"[b]{i18n.t('welcome.title')}[/b]\n\n"
            f"{i18n.t('welcome.intro')}\n\n"
            f"{i18n.t('welcome.tip')}"
        )

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
        placeholder = self.query_one("#screen-placeholder", Static)
        placeholder.update(self._welcome_text())