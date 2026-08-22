"""Sidebar navigation rail for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static

from audio_restoration import __version__

from .. import i18n
from ..navigation import SCREENS


class ScreenRequested(Message):
    """Posted when the user asks to switch to a screen."""

    def __init__(self, screen: str) -> None:
        super().__init__()
        self.screen = screen


class Sidebar(Vertical):
    """Left navigation rail with one button per screen."""

    DEFAULT_CSS = """
    Sidebar {
        width: 24;
        padding: 1 0 1 1;
        background: $surface;
        border-right: solid $border;
    }
    Sidebar #sidebar-title {
        text-style: bold;
        color: $accent;
        height: 1;
        padding: 0 0 1 0;
    }
    Sidebar Button {
        width: 100%;
        height: 1;
        margin: 0;
        padding: 0 1;
        background: transparent;
        color: $text-secondary;
        border: none;
        text-align: left;
    }
    Sidebar Button:hover {
        color: $text-primary;
        background: $surface-active;
    }
    Sidebar Button.-active {
        color: $accent;
        text-style: bold;
    }
    Sidebar #nav-separator-1,
    Sidebar #nav-separator-2 {
        height: 1;
        color: $border;
    }
    Sidebar #sidebar-version {
        color: $text-muted;
        height: 1;
        padding: 1 0 0 0;
    }
    """

    def __init__(self) -> None:
        super().__init__(id="sidebar")
        self._active_screen: str = "home"

    def compose(self) -> ComposeResult:
        yield Static(i18n.t("aria.sidebar"), id="sidebar-title")
        for i, (screen_id, label_key) in enumerate(SCREENS):
            if i == 3:
                yield Static("─", id="nav-separator-1")
            elif i == 5:
                yield Static("─", id="nav-separator-2")
            yield Button(
                i18n.t(label_key), id=f"nav-{screen_id}", compact=True
            )
        yield Static(f"v{__version__}", id="sidebar-version")

    def set_active(self, screen_id: str) -> None:
        self._active_screen = screen_id
        for sid, _label_key in SCREENS:
            try:
                btn = self.query_one(f"#nav-{sid}", Button)
                if sid == screen_id:
                    btn.add_class("-active")
                else:
                    btn.remove_class("-active")
            except Exception:  # noqa: BLE001, S110
                pass

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("nav-"):
            self.post_message(ScreenRequested(event.button.id[4:]))

    def refresh_labels(self) -> None:
        for screen_id, label_key in SCREENS:
            self.query_one(f"#nav-{screen_id}", Button).label = i18n.t(label_key)
