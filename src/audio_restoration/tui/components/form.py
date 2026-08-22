"""Reusable form widgets for TUI screens."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Input, Label, Select

from .. import i18n


class FieldRow(Vertical):
    """A labelled input/select with an optional browse button."""

    DEFAULT_CSS = """
    FieldRow {
        margin-bottom: 2;
    }
    FieldRow Label {
        color: $text-secondary;
        margin-bottom: 0;
        height: 1;
    }
    FieldRow Horizontal {
        height: auto;
    }
    FieldRow Input, FieldRow Select {
        width: 1fr;
    }
    FieldRow Button {
        width: 14;
        margin: 0 0 0 1;
    }
    """

    def __init__(
        self,
        label_key: str,
        input_id: str,
        *,
        browse_key: str | None = None,
        browse_id: str | None = None,
        select: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.label_key = label_key
        self.input_id = input_id
        self.browse_key = browse_key
        self.browse_id = browse_id
        self.select = select

    def compose(self) -> ComposeResult:
        yield Label(i18n.t(self.label_key))
        with Horizontal():
            if self.select:
                yield Select([], allow_blank=True, id=self.input_id)
            else:
                yield Input(id=self.input_id)
            if self.browse_key is not None:
                yield Button(
                    i18n.t(self.browse_key), id=self.browse_id or f"browse-{self.input_id}"
                )

    def refresh_labels(self) -> None:
        self.query_one(Label).update(i18n.t(self.label_key))
        if self.browse_key is not None:
            self.query_one(f"#{self.browse_id}", Button).label = i18n.t(self.browse_key)