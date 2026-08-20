"""Screen registry shared by the app.

Maps a screen id to a factory that builds a :class:`TuiScreen`.  New sections
(``single``, ``batch``, ``profiles``, ``history``, ``about``) graduate from the
placeholder to a real screen in their dedicated phase.
"""

from __future__ import annotations

from collections.abc import Callable

from ..state import TuiState
from .base import TuiScreen
from .home import HomeScreen
from .placeholder import PlaceholderScreen

ScreenFactory = Callable[[TuiState], TuiScreen]


def _placeholder(name: str) -> ScreenFactory:
    def build(state: TuiState) -> TuiScreen:
        return PlaceholderScreen(state, name=name)

    return build


#: Screen id → factory.  Ordered the same as the sidebar.
SCREEN_FACTORIES: dict[str, ScreenFactory] = {
    "home": lambda state: HomeScreen(state),
    "single": _placeholder("single"),
    "batch": _placeholder("batch"),
    "profiles": _placeholder("profiles"),
    "history": _placeholder("history"),
    "about": _placeholder("about"),
}

__all__ = ["SCREEN_FACTORIES", "TuiScreen"]