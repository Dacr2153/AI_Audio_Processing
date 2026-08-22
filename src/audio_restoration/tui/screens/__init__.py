"""TUI screens: each sidebar section renders a :class:`TuiScreen`."""

from __future__ import annotations

from .base import TuiScreen
from .home import HomeScreen
from .placeholder import PlaceholderScreen
from .registry import SCREEN_FACTORIES
from .single import SingleScreen

__all__ = [
    "SCREEN_FACTORIES",
    "HomeScreen",
    "PlaceholderScreen",
    "SingleScreen",
    "TuiScreen",
]