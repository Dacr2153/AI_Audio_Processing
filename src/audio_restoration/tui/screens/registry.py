"""Screen registry shared by the app.

Maps a screen id to a factory that builds a :class:`TuiScreen`.  New sections
(``batch``, ``profiles``, ``history``, ``about``) graduate from the placeholder
to a real screen in their dedicated phase.
"""

from __future__ import annotations

from collections.abc import Callable

from ..state import TuiState
from .about import AboutScreen
from .base import TuiScreen
from .batch import BatchScreen
from .history import HistoryScreen
from .home import HomeScreen
from .placeholder import PlaceholderScreen
from .profiles import ProfilesScreen
from .single import SingleScreen

ScreenFactory = Callable[[TuiState], TuiScreen]


def _placeholder(name: str) -> ScreenFactory:
    def build(state: TuiState) -> TuiScreen:
        return PlaceholderScreen(state, name=name)

    return build


#: Screen id → factory.  Ordered the same as the sidebar.
SCREEN_FACTORIES: dict[str, ScreenFactory] = {
    "home": lambda state: HomeScreen(state),
    "single": lambda state: SingleScreen(state),
    "batch": lambda state: BatchScreen(state),
    "profiles": lambda state: ProfilesScreen(state),
    "history": lambda state: HistoryScreen(state),
    "about": lambda state: AboutScreen(state),
}

__all__ = ["SCREEN_FACTORIES", "TuiScreen"]