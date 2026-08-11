"""Reranker interface and CrossEncoder implementation."""

# sentence-transformers is an optional lazy import with model-dependent return types.
# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any

from nanobot.memory_system.models import RankedResult


class Reranker(ABC):
    @abstractmethod
    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_n: int,
    ) -> list[RankedResult]: ...


class IdentityReranker(Reranker):
    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_n: int,
    ) -> list[RankedResult]:
        del query
        return [
            RankedResult(text=text, score=float(len(candidates) - index), source_index=index)
            for index, text in enumerate(candidates[:top_n])
        ]


class CrossEncoderReranker(Reranker):
    def __init__(
        self,
        model: str,
        *,
        max_batch_size: int = 32,
        encoder: Any | None = None,
    ) -> None:
        self._model_name = model
        self._max_batch_size = max_batch_size
        self._encoder = encoder
        self._load_lock = asyncio.Lock()

    async def _load(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        async with self._load_lock:
            if self._encoder is None:
                try:
                    from sentence_transformers import CrossEncoder
                except ImportError as exc:
                    raise RuntimeError(
                        "CrossEncoder memory requires the 'memory-reranker' extra"
                    ) from exc
                self._encoder = await asyncio.to_thread(CrossEncoder, self._model_name)
        return self._encoder

    async def rerank(
        self,
        query: str,
        candidates: list[str],
        top_n: int,
    ) -> list[RankedResult]:
        if not candidates:
            return []
        encoder = await self._load()
        scores: list[float] = []
        for start in range(0, len(candidates), self._max_batch_size):
            batch = candidates[start : start + self._max_batch_size]
            pairs = [(query, candidate) for candidate in batch]
            predicted = await asyncio.to_thread(encoder.predict, pairs)
            scores.extend(float(score) for score in predicted)
        ranked = sorted(
            (
                RankedResult(text=text, score=scores[index], source_index=index)
                for index, text in enumerate(candidates)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_n]
