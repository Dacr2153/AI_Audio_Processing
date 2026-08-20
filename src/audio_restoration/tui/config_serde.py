"""Serialize a :class:`PipelineConfig` to a flat dict and back.

Flat dicts are stored in TUI profiles/history and can also be handed to
``PipelineConfig(**flat)`` — the same contract the documented constructor
accepts via ``_LEGACY_FIELD_MAP``.
"""

from __future__ import annotations

from ..config import _LEGACY_FIELD_MAP, PipelineConfig

#: Reverse map: (sub-config attr, field) → flat legacy keyword.
_GROUP_TO_ATTR = {
    "denoise": "denoise",
    "declick": "declick",
    "dehum": "dehum",
    "eq": "eq",
    "multiband": "multiband",
    "ms": "ms",
    "wow_flutter": "wow_flutter",
    "separate": "separate",
    "sr": "sr",
    "loudness": "loudness",
    "output": "output",
    "report": "report",
}

#: Extra fields not present in the legacy map but worth serialising.
_EXTRA_FIELDS: dict[str, tuple[str, str]] = {
    "sr_target_sr": ("sr", "target_sr"),
}


def config_to_flat(config: PipelineConfig) -> dict:
    """Dump every mapped field as a flat ``{legacy_key: value}`` dict."""
    flat: dict = {}
    for legacy_key, (group, field) in _LEGACY_FIELD_MAP.items():
        attr = _GROUP_TO_ATTR[group]
        flat[legacy_key] = getattr(getattr(config, attr), field)
    for legacy_key, (group, field) in _EXTRA_FIELDS.items():
        flat[legacy_key] = getattr(getattr(config, _GROUP_TO_ATTR[group]), field)
    flat["genre"] = config.genre
    return flat


def flat_to_config(data: dict) -> PipelineConfig:
    """Rebuild a :class:`PipelineConfig` from a flat dict (returns a copy)."""
    # Only accept known keys so we never crash on stale profile fields.
    known = set(_LEGACY_FIELD_MAP) | set(_EXTRA_FIELDS) | {"genre"}
    sanitised = {k: v for k, v in data.items() if k in known}
    return PipelineConfig(**sanitised)