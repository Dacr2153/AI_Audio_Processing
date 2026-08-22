"""TUI screens: each sidebar section renders a :class:`TuiScreen`."""

from __future__ import annotations

from .base import TuiScreen
from .batch import BatchScreen
from .home import HomeScreen
from .placeholder import PlaceholderScreen
from .profiles import ProfilesScreen
from .registry import SCREEN_FACTORIES
from .single import SingleScreen

__all__ = [
    "SCREEN_FACTORIES",
    "BatchScreen",
    "HomeScreen",
    "PlaceholderScreen",
    "ProfilesScreen",
    "SingleScreen",
    "TuiScreen",
]