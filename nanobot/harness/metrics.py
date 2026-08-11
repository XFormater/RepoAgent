"""Low-cardinality OpenTelemetry metrics for harness control-plane operations."""

from __future__ import annotations

import importlib
from collections import Counter
from typing import Protocol, cast


class _Counter(Protocol):
    def add(self, amount: int, attributes: dict[str, str]) -> None: ...


class _Histogram(Protocol):
    def record(self, amount: int | float, attributes: dict[str, str]) -> None: ...


class _Meter(Protocol):
    def create_counter(self, name: str) -> _Counter: ...
    def create_histogram(self, name: str, *, unit: str) -> _Histogram: ...


class _Metrics(Protocol):
    def get_meter(self, name: str) -> _Meter: ...


class HarnessMetrics:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self._events: _Counter | None = None
        self._context_chars: _Histogram | None = None
        self._durations: _Histogram | None = None
        self.values: dict[str, float] = {}
        try:
            metrics = cast(_Metrics, importlib.import_module("opentelemetry.metrics"))
            meter = metrics.get_meter("nanobot.harness")
            self._events = meter.create_counter("nanobot.harness.events")
            self._context_chars = meter.create_histogram(
                "nanobot.harness.context.chars", unit="chars"
            )
            self._durations = meter.create_histogram(
                "nanobot.harness.operation.duration", unit="s"
            )
        except ImportError:  # pragma: no cover
            pass

    def event(self, name: str, *, role: str = "none", outcome: str = "ok") -> None:
        self.counts[f"{name}.{outcome}"] += 1
        if self._events is not None:
            self._events.add(1, {"event": name, "role": role, "outcome": outcome})

    def context(self, *, role: str, chars: int, truncated: bool) -> None:
        self.event("context", role=role, outcome="truncated" if truncated else "ok")
        if self._context_chars is not None:
            self._context_chars.record(chars, {"role": role})

    def duration(
        self,
        name: str,
        seconds: float,
        *,
        role: str = "none",
        outcome: str = "ok",
    ) -> None:
        self.values[f"duration_seconds.{name}.last"] = seconds
        if self._durations is not None:
            self._durations.record(
                seconds,
                {"operation": name, "role": role, "outcome": outcome},
            )

    def gauge(self, name: str, value: int | float) -> None:
        self.values[f"gauge.{name}"] = float(value)

    def snapshot(self) -> dict[str, int | float]:
        return {**self.counts, **self.values}
