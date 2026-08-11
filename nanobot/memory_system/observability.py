"""Optional structlog and OpenTelemetry integration with lightweight fallbacks."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from typing import Any, Protocol, cast


class StructuredLogger(Protocol):
    def debug(self, event: str, **values: Any) -> None: ...
    def info(self, event: str, **values: Any) -> None: ...
    def warning(self, event: str, **values: Any) -> None: ...
    def exception(self, event: str, **values: Any) -> None: ...


class _StructlogModule(Protocol):
    def get_logger(self, name: str) -> StructuredLogger: ...

try:
    _structlog = cast(_StructlogModule, importlib.import_module("structlog"))
    logger: StructuredLogger = _structlog.get_logger("nanobot.memory")
except ImportError:  # pragma: no cover - exercised only in minimal installations
    from loguru import logger as _loguru_logger

    class _CompatLogger:
        def _log(self, level: str, event: str, **values: Any) -> None:
            suffix = " ".join(f"{key}={value!r}" for key, value in values.items())
            getattr(_loguru_logger, level)(f"{event} {suffix}".rstrip())

        def debug(self, event: str, **values: Any) -> None:
            self._log("debug", event, **values)

        def info(self, event: str, **values: Any) -> None:
            self._log("info", event, **values)

        def warning(self, event: str, **values: Any) -> None:
            self._log("warning", event, **values)

        def exception(self, event: str, **values: Any) -> None:
            self._log("exception", event, **values)

    logger = _CompatLogger()

try:
    from opentelemetry import trace

    tracer = trace.get_tracer("nanobot.memory_system")
except ImportError:  # pragma: no cover - exercised only in minimal installations
    class _NoopTracer:
        def start_as_current_span(self, _name: str, **_kwargs: Any) -> Any:
            return nullcontext()

    tracer = _NoopTracer()
