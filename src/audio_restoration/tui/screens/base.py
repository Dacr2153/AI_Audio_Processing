"""Base class for TUI screens."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.widgets import Static

from .. import i18n
from ..state import TuiState


class TuiScreen(Vertical):
    """A navigable content panel mounted into the app's content area.

    Subclasses implement :meth:`form` to supply the widgets rendered inside a
    shared, titled, framed layout.
    """

    DEFAULT_CSS = """
    TuiScreen {
        align: left top;
    }
    TuiScreen > .screen-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
        height: auto;
    }
    TuiScreen > Vertical > #welcome-intro {
        color: $text-muted;
        margin-bottom: 1;
    }
    TuiScreen > Vertical > #welcome-tip {
        color: $text-muted;
        margin-top: 1;
    }

    /* Button hierarchy */
    TuiScreen Button {
        margin-bottom: 1;
        min-width: 14;
        height: 3;
    }
    TuiScreen Button.primary {
        background: $accent;
        color: $background;
        text-style: bold;
        border: none;
        min-width: 20;
    }
    TuiScreen Button.default {
        background: $surface-active;
        color: $text-primary;
        border: solid $border;
    }
    TuiScreen Button.flat {
        background: transparent;
        color: $text-secondary;
        border: none;
    }

    /* Inputs */
    TuiScreen Input {
        background: $surface;
        border: solid $border;
        color: $text-primary;
        padding: 0 1;
        height: 3;
    }
    TuiScreen Input:focus {
        border: solid $accent;
    }

    /* Panels */
    TuiScreen .panel {
        background: $surface;
        border: solid $border;
        padding: 1 2;
        margin-bottom: 1;
    }

    /* Field row widths */
    TuiScreen #row-input, TuiScreen #row-output {
        width: 70;
        max-width: 100%;
    }
    """

    #: i18n key for the screen title.
    TITLE_KEY = "nav.home"

    def __init__(self, state: TuiState, **kwargs) -> None:
        super().__init__(**kwargs)
        self.state = state

    def compose(self) -> ComposeResult:
        yield Static(i18n.t(self.TITLE_KEY), classes="screen-title")
        with Vertical():
            yield from self.form()

    def form(self) -> ComposeResult:
        """Extra widgets rendered below the title."""
        if False:
            yield

    def refresh_labels(self) -> None:
        """Re-translate visible labels after a language change."""
        title = self.query_one(".screen-title", Static)
        title.update(i18n.t(self.TITLE_KEY))