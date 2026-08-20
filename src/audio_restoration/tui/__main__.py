"""Entry point: ``python -m audio_restoration.tui`` / ``audio-restore-tui``."""

from __future__ import annotations

import sys


def run() -> None:
    """Launch the TUI (gracefully bail if Textual isn't installed)."""
    try:
        from .app import AudioRestorationTUI
    except ImportError as exc:  # textual not installed
        print(
            "audio-restore-tui requires the 'tui' extra.\n"
            "Install with: pip install \"audio-restoration[tui]\"",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    AudioRestorationTUI().run()


if __name__ == "__main__":
    run()