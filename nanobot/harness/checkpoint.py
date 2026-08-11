"""Bounded, secret-safe checkpoint projection for durable agent recovery."""

from __future__ import annotations

import json
from typing import Any, cast

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "apikey",
    "authorization",
    "cookie",
    "credential",
    "encrypted_content",
    "password",
    "provider_state",
    "secret",
    "token",
)


def _sensitive(key: object) -> bool:
    normalized = str(key).casefold().replace("-", "_")
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _project(
    value: object,
    *,
    depth: int,
    max_depth: int,
    max_items: int,
    max_string: int,
) -> Any:
    if depth >= max_depth:
        return "<depth-truncated>"
    if isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        projected: dict[str, Any] = {}
        for key, child in list(mapping.items())[:max_items]:
            if _sensitive(key):
                continue
            projected[str(key)] = _project(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )
        if len(mapping) > max_items:
            projected["_items_truncated"] = len(mapping) - max_items
        return projected
    if isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        selected = list(sequence)[-max_items:]
        return [
            _project(
                child,
                depth=depth + 1,
                max_depth=max_depth,
                max_items=max_items,
                max_string=max_string,
            )
            for child in selected
        ]
    if isinstance(value, str):
        return value[:max_string]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:max_string]


def _size(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=str).encode("utf-8"))


def sanitize_checkpoint_state(
    state: dict[str, Any],
    *,
    max_bytes: int = 262_144,
) -> tuple[dict[str, Any], bool]:
    """Remove credential-like fields and bound serialized checkpoint size."""
    if max_bytes < 1_024:
        raise ValueError("checkpoint max_bytes must be at least 1024")
    projected_value = _project(
        state,
        depth=0,
        max_depth=12,
        max_items=1_000,
        max_string=64_000,
    )
    assert isinstance(projected_value, dict)
    projected = cast(dict[str, Any], projected_value)
    if _size(projected) <= max_bytes:
        return projected, projected != state

    compact_value = _project(
        projected,
        depth=0,
        max_depth=6,
        max_items=40,
        max_string=2_048,
    )
    assert isinstance(compact_value, dict)
    compact = cast(dict[str, Any], compact_value)
    compact["_checkpoint_truncated"] = True
    if _size(compact) <= max_bytes:
        return compact, True

    essential: dict[str, Any] = {"_checkpoint_truncated": True}
    for key in ("phase", "iteration", "stop_reason", "error", "usage"):
        if key in compact:
            essential[key] = compact[key]
    essential["available_state_keys"] = [
        key for key in compact if key != "_checkpoint_truncated"
    ][:50]
    while _size(essential) > max_bytes and essential["available_state_keys"]:
        essential["available_state_keys"].pop()
    return essential, True
