from __future__ import annotations

import json
from typing import Any

import pytest

from nanobot.memory_system.config import ShortTermConfig
from nanobot.memory_system.models import (
    ConversationMessage,
    ReasoningState,
    SessionContextState,
    TaskState,
    ToolExecutionState,
)
from nanobot.memory_system.short_term.redis_memory import (
    RedisShortTermMemory,
    ShortTermConflictError,
)


class _Pipeline:
    def __init__(self, redis: "_Redis") -> None:
        self.redis = redis
        self.operations: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    async def __aenter__(self) -> "_Pipeline":
        return self

    async def __aexit__(self, *_args: object) -> None:
        return None

    def __getattr__(self, name: str):
        def queue(*args: Any, **kwargs: Any) -> "_Pipeline":
            self.operations.append((name, args, kwargs))
            return self

        return queue

    async def execute(self) -> None:
        for name, args, kwargs in self.operations:
            await getattr(self.redis, name)(*args, **kwargs)


class _Redis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.lists: dict[str, list[str]] = {}
        self.ttls: dict[str, int] = {}

    def pipeline(self, transaction: bool = True) -> _Pipeline:
        del transaction
        return _Pipeline(self)

    async def set(
        self,
        key: str,
        value: str,
        *,
        ex: int | None = None,
        nx: bool = False,
    ) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        if ex is not None:
            self.ttls[key] = ex
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self.values.pop(key, None)
            self.lists.pop(key, None)

    async def rpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).append(value)

    async def ltrim(self, key: str, start: int, end: int) -> None:
        values = self.lists.get(key, [])
        normalized_start = max(len(values) + start, 0) if start < 0 else start
        normalized_end = len(values) + end if end < 0 else end
        self.lists[key] = values[normalized_start : normalized_end + 1]

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        normalized_start = max(len(values) + start, 0) if start < 0 else start
        normalized_end = len(values) + end if end < 0 else end
        return values[normalized_start : normalized_end + 1]

    async def expire(self, key: str, ttl: int) -> None:
        self.ttls[key] = ttl

    async def eval(
        self,
        _script: str,
        _numkeys: int,
        key: str,
        expected: int,
        payload: str,
        ttl: int,
    ) -> int:
        current = self.values.get(key)
        actual = TaskState.model_validate_json(current).version if current else 0
        if actual != int(expected):
            return actual
        await self.set(key, payload, ex=int(ttl))
        return -1

    async def scan_iter(self, *, match: str, count: int = 100):
        import fnmatch

        del count
        for key in [*self.values, *self.lists]:
            if fnmatch.fnmatch(key, match):
                yield key


@pytest.fixture
def memory() -> RedisShortTermMemory:
    return RedisShortTermMemory(
        "redis://unused",
        ShortTermConfig(
            enabled=True,
            max_messages=2,
            conversation_ttl=100,
            task_ttl=200,
            reasoning_ttl=300,
            tool_ttl=400,
        ),
        client=_Redis(),
    )


async def test_conversation_is_bounded_and_refreshes_ttl(
    memory: RedisShortTermMemory,
) -> None:
    for index in range(3):
        await memory.append_message(
            "session-1",
            ConversationMessage(
                message_id=f"m{index}",
                role="user",
                content=str(index),
            ),
        )

    messages = await memory.get_messages("session-1")

    assert [message.message_id for message in messages] == ["m1", "m2"]
    assert memory._client.ttls[memory.messages_key("session-1")] == 100


async def test_task_uses_optimistic_version(memory: RedisShortTermMemory) -> None:
    initial = await memory.put_task(
        "session-1",
        TaskState(task_id="task-1", goal="ship"),
    )
    updated = await memory.put_task(
        "session-1",
        initial.model_copy(update={"current_step": "test"}),
        expected_version=1,
    )

    assert updated.version == 2
    with pytest.raises(ShortTermConflictError):
        await memory.put_task(
            "session-1",
            updated,
            expected_version=1,
        )


async def test_reasoning_and_tool_dedup_round_trip(
    memory: RedisShortTermMemory,
) -> None:
    reasoning = ReasoningState(
        constraints=["read only"],
        current_requirements=["inspect"],
    )
    await memory.put_reasoning("session-1", reasoning)
    tool = ToolExecutionState(
        execution_id="exec-1",
        tool_name="search",
        arguments={"q": "memory"},
        result={"count": 1},
        status="succeeded",
    )
    await memory.put_tool_execution("session-1", tool)

    assert await memory.get_reasoning("session-1") == reasoning
    assert await memory.find_tool_execution(
        "session-1",
        "search",
        {"q": "memory"},
    ) == tool


async def test_complete_short_term_context_is_readable(memory: RedisShortTermMemory) -> None:
    await memory.append_message(
        "session-1",
        ConversationMessage(message_id="m1", role="user", content="continue task"),
    )
    await memory.put_task(
        "session-1",
        TaskState(task_id="task-1", goal="ship", current_step="test"),
    )
    await memory.put_context(
        "session-1",
        SessionContextState(current_goal="ship", active_files=["app.py"]),
    )
    await memory.put_run_state("session-1", {"status": "running"})

    rendered = await memory.build_runtime_context("session-1")

    assert "<short_term_memory>" in rendered
    assert "continue task" in rendered
    assert '"tasks"' in rendered
    assert "app.py" in rendered
    assert '"agent_state":{"status":"running"}' in rendered


async def test_bounded_runtime_context_remains_valid_json_and_escapes_key_globs() -> None:
    memory = RedisShortTermMemory(
        "redis://unused",
        ShortTermConfig(enabled=True, runtime_context_max_chars=512),
        client=_Redis(),
    )
    session_id = "tenant:*?[session]"
    for index in range(20):
        await memory.append_message(
            session_id,
            ConversationMessage(
                message_id=f"m{index}",
                role="user",
                content="large content " * 100,
            ),
        )

    rendered = await memory.build_runtime_context(session_id)
    encoded = rendered.splitlines()[2]

    assert len(encoded) <= 512
    assert json.loads(encoded)["truncated"] is True
    assert "*" not in memory.messages_key(session_id)
