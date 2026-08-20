"""Shared application state for the TUI.

A lightweight container that lets screens exchange configuration, profiles and
job history without global singletons or deep coupling.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..config import PipelineConfig
from .config_serde import config_to_flat, flat_to_config


def _data_dir() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")
    return Path(base) / "audio-restore"


def _profiles_file() -> Path:
    return _data_dir() / "profiles.json"


def _history_file() -> Path:
    return _data_dir() / "history.json"


@dataclass
class TuiState:
    """Cross-screen state: config, profiles and an event bus for refresh."""

    config: PipelineConfig = field(default_factory=PipelineConfig)
    input_path: str | None = None
    output_path: str | None = None
    batch_folder: str | None = None
    output_dir: str | None = None
    output_ext: str = "wav"
    output_suffix: str = ""
    workers: int = 1

    #: UI flags — not part of the processing config.
    language: str = "en"
    theme: str = "dark"

    #: Subscribers keyed by event name; each is a zero-arg callable.
    _listeners: dict[str, list[Callable[[], None]]] = field(default_factory=dict)

    def add_listener(self, event: str, callback: Callable[[], None]) -> None:
        self._listeners.setdefault(event, []).append(callback)

    def emit(self, event: str) -> None:
        for callback in self._listeners.get(event, []):
            callback()

    # ------------------------------------------------------------------
    # Profiles
    # ------------------------------------------------------------------

    def save_profile(self, name: str) -> None:
        directory = _data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        profiles = self.load_profiles()
        profiles[name] = self.profile_data()
        _profiles_file().write_text(
            json.dumps(profiles, indent=2, sort_keys=True, ensure_ascii=False)
        )

    def load_profiles(self) -> dict[str, dict]:
        file = _profiles_file()
        if not file.is_file():
            return {}
        try:
            data = json.loads(file.read_text())
            return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def delete_profile(self, name: str) -> None:
        profiles = self.load_profiles()
        profiles.pop(name, None)
        _data_dir().mkdir(parents=True, exist_ok=True)
        _profiles_file().write_text(
            json.dumps(profiles, indent=2, sort_keys=True, ensure_ascii=False)
        )

    def list_profiles(self) -> list[str]:
        return sorted(self.load_profiles())

    def apply_profile(self, name: str) -> None:
        profiles = self.load_profiles()
        if name in profiles:
            self.config = flat_to_config(profiles[name])

    def profile_data(self) -> dict:
        """Serialise the current config to a flat dict for storage."""
        return config_to_flat(self.config)

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def add_history_entry(self, entry: dict) -> None:
        directory = _data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        entry = {**entry, "timestamp": str(datetime.now().astimezone().isoformat(timespec="seconds"))}
        history = self.load_history()
        history.insert(0, entry)
        _history_file().write_text(json.dumps(history, indent=2, ensure_ascii=False))

    def load_history(self) -> list[dict]:
        file = _history_file()
        if not file.is_file():
            return []
        try:
            data = json.loads(file.read_text())
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def clear_history(self) -> None:
        _history_file().unlink(missing_ok=True)