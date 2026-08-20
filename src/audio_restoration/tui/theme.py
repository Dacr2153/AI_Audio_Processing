"""Theme definitions for the TUI.

Two built-in looks — ``dark`` (default, amber/cyan on near-black) and
``light`` (ink on warm paper).  Both expose the same semantic CSS variables so
the rest of the app does not care which theme is active.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThemeName = Literal["dark", "light"]


@dataclass(frozen=True)
class Palette:
    """Semantic colours used by the TUI stylesheets."""

    bg: str
    bg_alt: str
    surface: str
    border: str
    fg: str
    fg_muted: str
    accent: str
    accent2: str
    ok: str
    warn: str
    error: str


_PALETTES: dict[ThemeName, Palette] = {
    "dark": Palette(
        bg="#0c0e12",
        bg_alt="#11141a",
        surface="#161b23",
        border="#2a3240",
        fg="#d7dee8",
        fg_muted="#8b96a5",
        accent="#e0a458",   # amber
        accent2="#58a6e0",  # cyan
        ok="#56c98a",
        warn="#e0c058",
        error="#e05558",
    ),
    "light": Palette(
        bg="#f4f1ea",
        bg_alt="#ebe6dc",
        surface="#fffdf8",
        border="#cfc8b8",
        fg="#2c2f33",
        fg_muted="#6b7078",
        accent="#b0651a",   # deep amber
        accent2="#2b6ea8",  # deep cyan
        ok="#2c8f5c",
        warn="#a8861b",
        error="#b03338",
    ),
}


def palette(theme: ThemeName) -> Palette:
    return _PALETTES[theme]


def theme_vars(theme: ThemeName) -> str:
    """Return a ``:root``-compatible set of ``--var`` lines for CSS."""
    p = _PALETTES[theme]
    return "\n".join(
        f"App {{ --{key}: {value}; }}"
        for key, value in {
            "bg": p.bg,
            "bg-alt": p.bg_alt,
            "surface": p.surface,
            "border": p.border,
            "fg": p.fg,
            "fg-muted": p.fg_muted,
            "accent": p.accent,
            "accent-2": p.accent2,
            "ok": p.ok,
            "warn": p.warn,
            "error": p.error,
        }.items()
    )