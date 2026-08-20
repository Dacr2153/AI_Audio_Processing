"""Home screen: welcome banner and quick-action shortcuts."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Button, Static

from .. import i18n
from ..components.sidebar import ScreenRequested
from .base import TuiScreen

#: Quick actions: (i18n label key, destination screen id).
_ACTIONS = (
    ("welcome.single", "single"),
    ("welcome.batch", "batch"),
    ("welcome.presets", "profiles"),
    ("welcome.history", "history"),
    ("welcome.about", "about"),
)


class HomeScreen(TuiScreen):
    """Landing screen with a welcome message and navigation buttons."""

    TITLE_KEY = "nav.home"

    def form(self) -> ComposeResult:
        yield Static(i18n.t("welcome.intro"), id="welcome-intro")
        for label_key, dest in _ACTIONS:
            yield Button(i18n.t(label_key), id=f"action-{dest}", variant="primary")
        yield Static(i18n.t("welcome.tip"), id="welcome-tip")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("action-"):
            self.post_message(ScreenRequested(event.button.id[7:]))

    def refresh_labels(self) -> None:
        super().refresh_labels()
        self.query_one("#welcome-intro", Static).update(i18n.t("welcome.intro"))
        self.query_one("#welcome-tip", Static).update(i18n.t("welcome.tip"))
        for label_key, dest in _ACTIONS:
            self.query_one(f"#action-{dest}", Button).label = i18n.t(label_key)