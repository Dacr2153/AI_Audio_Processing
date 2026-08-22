"""Home screen: dashboard with quick actions, recent activity and active profile."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Static

from .. import i18n
from ..components.sidebar import ScreenRequested
from .base import TuiScreen


class HomeScreen(TuiScreen):
    """Dashboard with quick actions, recent activity and active profile."""

    TITLE_KEY = "nav.home"

    DEFAULT_CSS = """
    HomeScreen .section-title {
        text-style: bold;
        color: $text-secondary;
        height: 1;
        margin-bottom: 1;
    }
    HomeScreen .quick-actions {
        height: auto;
        margin-bottom: 2;
    }
    HomeScreen .quick-actions Button {
        margin-right: 2;
        min-width: 20;
    }
    HomeScreen .recent-list {
        height: auto;
        margin-bottom: 2;
    }
    HomeScreen .recent-item {
        height: 1;
        color: $text-primary;
    }
    HomeScreen .recent-item .status-ok {
        color: $success;
    }
    HomeScreen .recent-item .status-fail {
        color: $error;
    }
    HomeScreen .recent-empty {
        color: $text-muted;
        height: 1;
        margin-bottom: 1;
    }
    HomeScreen .profile-section {
        height: auto;
    }
    HomeScreen .profile-row {
        height: 1;
        color: $text-secondary;
    }
    """

    def form(self) -> ComposeResult:
        yield Static(i18n.t("welcome.intro"), id="welcome-intro")

        # Quick Actions
        yield Static(i18n.t("dashboard.quick_actions"), classes="section-title")
        with Horizontal(classes="quick-actions"):
            yield Button(
                i18n.t("welcome.single"),
                id="action-single",
                variant="primary",
                compact=True,
            )
            yield Button(
                i18n.t("welcome.batch"),
                id="action-batch",
                variant="primary",
                compact=True,
            )

        # Recent Activity
        yield Static(i18n.t("dashboard.recent"), classes="section-title")
        with Vertical(classes="recent-list", id="recent-list"):
            pass

        # Active Profile
        yield Static(i18n.t("dashboard.active_profile"), classes="section-title")
        with Vertical(classes="profile-section", id="profile-section"):
            pass

    def on_mount(self) -> None:
        self._refresh_recent()
        self._refresh_profile()

    def _refresh_recent(self) -> None:
        container = self.query_one("#recent-list", Vertical)
        try:
            container.remove_children()
        except Exception:  # noqa: BLE001, S110
            pass
        entries = self.state.load_history()[:5]
        if not entries:
            container.mount(
                Static(
                    i18n.t("dashboard.recent_empty"),
                    classes="recent-empty",
                )
            )
            return
        for entry in entries:
            name = entry.get("input", "").split("/")[-1]
            ts = entry.get("timestamp", "")
            short_ts = ts[:10] if len(ts) >= 10 else ts
            container.mount(
                Static(f"● {name}  {short_ts}", classes="recent-item")
            )

    def _refresh_profile(self) -> None:
        container = self.query_one("#profile-section", Vertical)
        try:
            container.remove_children()
        except Exception:  # noqa: BLE001, S110
            pass
        cfg = self.state.config
        rows = [
            (f"denoise  {cfg.denoise.method}"),
            (f"declick  {'on' if cfg.declick.enabled else 'off'}"),
            (f"dehum    {cfg.dehum.freq or 'off'}"),
            (f"eq       {cfg.genre or 'default'}"),
        ]
        for text in rows:
            container.mount(Static(text, classes="profile-row"))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("action-"):
            self.post_message(ScreenRequested(event.button.id[7:]))

    def refresh_labels(self) -> None:
        super().refresh_labels()
        self.query_one("#welcome-intro", Static).update(i18n.t("welcome.intro"))
        self.query_one("#action-single", Button).label = i18n.t("welcome.single")
        self.query_one("#action-batch", Button).label = i18n.t("welcome.batch")
