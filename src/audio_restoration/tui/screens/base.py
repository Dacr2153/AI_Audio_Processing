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
    TuiScreen Button {
        margin-bottom: 1;
        width: 30;
    }
    TuiScreen #run-btn {
        width: 20;
        margin-right: 1;
    }
    TuiScreen #run-hint {
        height: 3;
        width: auto;
        color: $text-muted;
        align: left middle;
    }
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