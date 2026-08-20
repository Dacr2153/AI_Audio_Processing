"""Navigation registry shared by the app and its components."""

from __future__ import annotations

#: Screens reachable from the sidebar, in order: (screen id, i18n label key).
SCREENS: tuple[tuple[str, str], ...] = (
    ("home", "nav.home"),
    ("single", "nav.single"),
    ("batch", "nav.batch"),
    ("profiles", "nav.profiles"),
    ("history", "nav.history"),
    ("about", "nav.about"),
)