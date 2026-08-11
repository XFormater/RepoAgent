"""LLM semantic extraction with strict source-ID validation."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable

import json_repair
from pydantic import TypeAdapter, ValidationError

from nanobot.memory_system.models import ExtractedFact
from nanobot.memory_system.observability import logger, tracer

LLMExtractCallable = Callable[[list[dict[str, str]]], Awaitable[str]]

_FACT_LIST = TypeAdapter(list[ExtractedFact])
_SYSTEM_PROMPT = """You extract durable user memory from a conversation.
Return JSON only: an array of objects with fact_text, fact_type, entities,
relations, confidence, and source_message_ids. fact_type is preference, fact,
hypothesis, or goal. relations is an optional array of
{source,target,relation_type}; relation_type is INTERESTED_IN, WORKS_ON, USES,
PREFERS, or RELATES_TO. Keep only stable preferences, technology stack, work
domain, project context, important user facts, and durable goals. Drop greetings,
small talk, confirmations, assistant claims, transient questions, one-time
requests, secrets, credentials, and low-information text. Never invent a
source_message_id: use only IDs shown in the input. Hypotheses are allowed but
will be held for review. Entities and relations must be directly supported by
the cited user messages."""


class MemoryExtractor(ABC):
    @abstractmethod
    async def extract(self, messages: list[dict[str, str]]) -> list[ExtractedFact]: ...


class LLMMemoryExtractor(MemoryExtractor):
    def __init__(self, invoke: LLMExtractCallable) -> None:
        self._invoke = invoke

    async def extract(self, messages: list[dict[str, str]]) -> list[ExtractedFact]:
        if not messages:
            return []
        allowed_ids = {message["id"] for message in messages}
        user_ids = {
            message["id"]
            for message in messages
            if message.get("role") == "user"
        }
        serialized = json.dumps(messages, ensure_ascii=False)
        with tracer.start_as_current_span("memory.record.extract"):
            raw = await self._invoke(
                [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": serialized},
                ]
            )
        try:
            value = json_repair.loads(raw)
            if isinstance(value, dict):
                value = value.get("facts", [])
            facts = _FACT_LIST.validate_python(value)
        except (ValueError, TypeError, ValidationError) as exc:
            logger.warning("memory_extraction_invalid", error=str(exc))
            return []

        accepted: list[ExtractedFact] = []
        for fact in facts:
            source_ids = [
                source_id
                for source_id in fact.source_message_ids
                if source_id in allowed_ids
            ]
            if not source_ids or not user_ids.intersection(source_ids):
                logger.warning("memory_fact_has_no_valid_user_sources")
                continue
            accepted.append(fact.model_copy(update={"source_message_ids": source_ids}))
        return accepted


class StaticMemoryExtractor(MemoryExtractor):
    """Deterministic extractor used by tests and custom integrations."""

    def __init__(self, facts: list[ExtractedFact]) -> None:
        self._facts = facts

    async def extract(self, messages: list[dict[str, str]]) -> list[ExtractedFact]:
        del messages
        return [fact.model_copy(deep=True) for fact in self._facts]
