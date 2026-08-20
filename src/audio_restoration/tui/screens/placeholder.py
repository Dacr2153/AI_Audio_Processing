"""Placeholder screen shown for sections not yet implemented."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from .. import i18n
from .base import TuiScreen


class PlaceholderScreen(TuiScreen):
    """Temporary body for single/batch/profiles/history/about screens."""

    def __init__(self, state, name: str, **kwargs) -> None:
        super().__init__(state, **kwargs)
        self._name = name

    def compose(self) -> ComposeResult:
        yield Static(i18n.t(f"nav.{self._name}"), classes="screen-title")
        with Vertical():
            yield Static(i18n.t("welcome.tip"), id="welcome-tip")

    def refresh_labels(self) -> None:
        title = self.query_one(".screen-title", Static)
        title.update(i18n.t(f"nav.{self._name}"))
        tip = self.query_one("#welcome-tip", Static)
        tip.update(i18n.t("welcome.tip"))