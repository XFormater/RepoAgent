from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nanobot.agent.loop import AgentLoop
from nanobot.bus.events import OUTBOUND_META_AGENT_UI, InboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.providers.base import GenerationSettings, LLMResponse
from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.models import RepositoryRecord
from nanobot.repo_agent.repository import RepositoryManager


def _provider() -> MagicMock:
    provider = MagicMock()
    provider.get_default_model.return_value = "test-model"
    provider.generation = GenerationSettings(max_tokens=1024, temperature=0.0)
    provider.estimate_prompt_tokens.return_value = (100, "test")
    provider.can_resume_conversation_state.return_value = False
    provider.supports_progress_deltas = False
    provider.runtime = SimpleNamespace()
    return provider


@pytest.mark.asyncio
async def test_loop_denies_forbidden_github_action_before_clone_or_model(
    tmp_path,
    monkeypatch,
) -> None:
    provider = _provider()
    provider.chat_with_retry = AsyncMock()
    load = AsyncMock()
    monkeypatch.setattr(RepositoryManager, "load", load)
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        repository_agent_config=RepositoryAgentConfig(
            enabled=True,
            intent_strategy="rules",
        ),
    )
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)

    result = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content=(
                "https://github.com/owner/repo "
                "强制 push main 分支并绕过分支保护"
            ),
        )
    )

    assert result is not None
    assert "L4" in result.content
    provider.chat_with_retry.assert_not_awaited()
    load.assert_not_awaited()


@pytest.mark.asyncio
async def test_loop_loads_url_and_routes_question_to_repo_expert(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / ".nanobot" / "repositories" / "owner" / "repo" / "source"
    source.mkdir(parents=True)
    (source / "main.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    record = RepositoryRecord(
        repository_id="repo",
        url="https://github.com/owner/repo",
        owner="owner",
        name="repo",
        default_branch="main",
        commit_sha="a" * 40,
        local_path=str(source),
        file_count=1,
        size_bytes=30,
    )
    monkeypatch.setattr(RepositoryManager, "load", AsyncMock(return_value=record))
    provider = _provider()
    provider.chat_with_retry = AsyncMock(
        return_value=LLMResponse(
            content="入口位于 main.py:1。",
            usage={"prompt_tokens": 10, "completion_tokens": 8},
        )
    )
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        repository_agent_config=RepositoryAgentConfig(
            enabled=True,
            intent_strategy="rules",
        ),
    )
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)

    result = await loop._process_message(
        InboundMessage(
            channel="cli",
            sender_id="user",
            chat_id="direct",
            content="https://github.com/owner/repo 这个项目的入口在哪里？",
        )
    )

    assert result is not None
    assert "main.py:1" in result.content
    assert f"项目本地保存路径：`{source}`" in result.content
    call = provider.chat_with_retry.await_args
    tool_names = {item["function"]["name"] for item in call.kwargs["tools"]}
    assert {"repo_tree", "repo_search", "repo_read"} <= tool_names


@pytest.mark.asyncio
async def test_delete_approval_slash_command_reaches_repository_workflow(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / ".nanobot" / "repositories" / "owner" / "repo" / "source"
    source.mkdir(parents=True)
    record = RepositoryRecord(
        repository_id="repo",
        url="https://github.com/owner/repo",
        owner="owner",
        name="repo",
        default_branch="main",
        commit_sha="a" * 40,
        local_path=str(source),
        file_count=1,
        size_bytes=30,
    )
    cache_path = source.parent
    resolve = AsyncMock(return_value=record)
    delete = AsyncMock(return_value=cache_path)
    monkeypatch.setattr(RepositoryManager, "resolve_cached_repository", resolve)
    monkeypatch.setattr(RepositoryManager, "cache_path", lambda _self, _record: cache_path)
    monkeypatch.setattr(RepositoryManager, "delete_cached_repository", delete)
    provider = _provider()
    provider.chat_with_retry = AsyncMock()
    loop = AgentLoop(
        bus=MessageBus(),
        provider=provider,
        workspace=tmp_path,
        model="test-model",
        repository_agent_config=RepositoryAgentConfig(
            enabled=True,
            intent_strategy="rules",
        ),
    )
    loop.consolidator.maybe_consolidate_by_tokens = AsyncMock(return_value=None)

    prepared = await loop._process_message(
        InboundMessage(
            channel="websocket",
            sender_id="user",
            chat_id="chat",
            content="删除本地下载的 owner/repo 项目",
        )
    )

    assert prepared is not None
    agent_ui = prepared.metadata[OUTBOUND_META_AGENT_UI]
    approval_id = agent_ui["data"]["approval_id"]

    completed = await loop._process_message(
        InboundMessage(
            channel="websocket",
            sender_id="user",
            chat_id="chat",
            content=f"/approve-delete {approval_id}",
        )
    )

    assert completed is not None
    assert "Unknown command" not in completed.content
    assert "本地项目已删除" in completed.content
    delete.assert_awaited_once_with(record)
    provider.chat_with_retry.assert_not_awaited()
