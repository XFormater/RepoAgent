"""Embedding provider interface and implementations."""

from __future__ import annotations

import asyncio
import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

from nanobot.memory_system.storage.lexical import lexical_tokens


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbeddingProvider(EmbeddingProvider):
    """Deterministic local embedding used by tests and zero-service demos."""

    def __init__(self, dimension: int = 256) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            values = [0.0] * self._dimension
            normalized = " ".join(text.lower().split())
            tokens = normalized.split() or [normalized]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "big") % self._dimension
                values[index] += -1.0 if digest[4] & 1 else 1.0
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors


class LexicalHashEmbeddingProvider(EmbeddingProvider):
    """Sparse lexical feature hashing for deterministic local semantic recall.

    Unlike :class:`HashEmbeddingProvider`, features are non-negative so shared
    words cannot cancel each other. Word bigrams preserve a small amount of
    phrase structure without downloading an embedding model.
    """

    def __init__(self, dimension: int = 2048) -> None:
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            tokens = lexical_tokens(text)
            features = [*tokens, *(f"{left}\x1f{right}" for left, right in zip(tokens, tokens[1:]))]
            values = [0.0] * self._dimension
            for feature in features:
                digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
                index = int.from_bytes(digest, "big") % self._dimension
                values[index] += 1.0
            norm = math.sqrt(sum(value * value for value in values)) or 1.0
            vectors.append([value / norm for value in values])
        return vectors


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI-compatible async embedding adapter."""

    def __init__(
        self,
        *,
        model: str,
        dimension: int,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._client = client
        self._model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self._model,
            input=texts,
            dimensions=self._dimension,
        )
        ordered = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in ordered]


async def embed_in_batches(
    provider: EmbeddingProvider,
    texts: list[str],
    *,
    batch_size: int = 64,
) -> list[list[float]]:
    """Embed a large input without issuing an unbounded provider request."""
    result: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        result.extend(await provider.embed(texts[start : start + batch_size]))
        await asyncio.sleep(0)
    return result
