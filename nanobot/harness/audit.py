"""Immutable, queryable audit envelopes for Harness and turn events.

The audit trail intentionally retains the business payload needed to replay an
execution trace, while redacting credential-shaped fields before persistence.
It is separate from operational logs: journals are for operators, this module
is for durable per-message/per-event evidence.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping
from typing import Any

_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|authorization|token|secret|password|cookie)", re.I)
_TOKEN = re.compile(
    r"(?:github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_-]{12,}|"
    r"(?i:bearer)\s+[A-Za-z0-9._~+/=-]{16,})"
)


def redact_audit_value(value: Any, *, key: str | None = None) -> Any:
    """Return a JSON-compatible audit value without credential material."""
    if key and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return _TOKEN.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        return {str(name): redact_audit_value(item, key=str(name)) for name, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [redact_audit_value(item) for item in value]
    if value is None or isinstance(value, bool | int | float):
        return value
    return str(value)


def payload_digest(payload: Mapping[str, Any]) -> str:
    """Stable digest used to prove the exact redacted payload that was logged."""
    import json

    rendered = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
