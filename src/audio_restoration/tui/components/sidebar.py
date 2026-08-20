"""Sidebar navigation rail for the TUI."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import Button, Static

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
    Sidebar { width: 24; padding: 1; }
    Sidebar #sidebar-title {
        text-style: bold;
        color: $accent;
        padding-bottom: 1;
        text-align: center;
    }
    Sidebar Button { width: 100%; margin-bottom: 1; }
    """

    def __init__(self) -> None:
        super().__init__(id="sidebar")

    def compose(self) -> ComposeResult:
        yield Static(i18n.t("aria.sidebar"), id="sidebar-title")
        for screen_id, label_key in SCREENS:
            yield Button(i18n.t(label_key), id=f"nav-{screen_id}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("nav-"):
            self.post_message(ScreenRequested(event.button.id[4:]))

    def refresh_labels(self) -> None:
        for screen_id, label_key in SCREENS:
            self.query_one(f"#nav-{screen_id}", Button).label = i18n.t(label_key)