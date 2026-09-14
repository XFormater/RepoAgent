"""Thin AgentLoop adapter for repository multi-agent turns."""

# pyright: reportPrivateUsage=false

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nanobot.agent.tools.context import RequestContext
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.repo_agent.executor import RepositoryAgentRun
from nanobot.utils.llm_runtime import LLMRuntime

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop, TurnContext


async def run_repository_turn(loop: AgentLoop, ctx: TurnContext) -> None:
    orchestrator = loop.repository_orchestrator
    if orchestrator is None:
        raise RuntimeError("repository orchestrator is not configured")
    runtime = ctx.require_runtime()

    async def invoke_repository_agent(
        messages: list[dict[str, object]],
        *,
        runtime: LLMRuntime,
        tools: ToolRegistry,
        max_iterations: int | None,
        ephemeral: bool,
        stream: bool,
        workspace: Path,
        repository_full_name: str,
    ) -> RepositoryAgentRun:
        metadata = {
            **ctx.msg.metadata,
            "repository_full_name": repository_full_name,
            "repository_workspace": str(workspace),
        }
        request_context = RequestContext(
            channel=ctx.delivery.route.channel,
            chat_id=ctx.delivery.route.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            session_key=ctx.session_key,
            original_user_text=ctx.original_user_text,
            runtime=runtime,
            metadata=metadata,
            sender_id=ctx.msg.sender_id,
            turn_id=ctx.turn_id,
            workspace=workspace,
            tenant_id=str(ctx.msg.metadata.get("tenant_id", "default")),
            user_id=str(
                ctx.msg.metadata.get("user_id")
                or ctx.msg.sender_id
                or ctx.delivery.route.chat_id
            ),
        )
        result = await loop._run_agent_loop(
            cast(list[dict[str, Any]], messages),
            runtime=runtime,
            on_progress=ctx.on_progress,
            on_stream=ctx.on_stream if stream else None,
            on_stream_end=ctx.on_stream_end if stream else None,
            on_retry_wait=ctx.on_retry_wait,
            session=None if ephemeral else ctx.session,
            channel=ctx.delivery.route.channel,
            chat_id=ctx.delivery.route.chat_id,
            message_id=ctx.msg.metadata.get("message_id"),
            metadata=metadata,
            session_key=ctx.session_key,
            original_user_text=ctx.original_user_text,
            pending_queue=None if ephemeral else ctx.pending_queue,
            ephemeral=ephemeral,
            run_extra_hooks_for_ephemeral=ctx.run_extra_hooks_for_ephemeral,
            hooks=ctx.hooks,
            hook_factories=ctx.hook_factories,
            turn_scopes=ctx.turn_scopes,
            tools=tools,
            request_context=request_context,
            provider_state=None if ephemeral else ctx.provider_state,
            max_iterations=max_iterations,
            workspace_override=workspace,
        )
        content, used, all_messages, stop_reason, had_injections = result
        return RepositoryAgentRun(
            final_content=content or "",
            tools_used=used,
            messages=cast(list[dict[str, object]], all_messages),
            stop_reason=stop_reason,
            had_injections=had_injections,
        )

    result = await orchestrator.run(
        request=ctx.original_user_text or ctx.msg.content,
        actor=ctx.msg.sender_id,
        session_key=ctx.session_key,
        initial_messages=ctx.initial_messages,
        runtime=runtime,
        invoke=invoke_repository_agent,
        on_update=ctx.on_progress,
        on_stream=ctx.on_stream,
        on_stream_end=ctx.on_stream_end,
        tenant_id=str(ctx.msg.metadata.get("tenant_id", "default")),
        trace_id=(
            str(ctx.msg.metadata["trace_id"])
            if isinstance(ctx.msg.metadata.get("trace_id"), str)
            else None
        ),
    )
    ctx.final_content = result.final_content
    ctx.all_messages = result.messages
    ctx.stop_reason = result.stop_reason
    ctx.had_injections = result.had_injections
    ctx.agent_ui = result.agent_ui
    # A /rere POOL turn returns immediately.  The Gateway-side durable-event
    # relay owns all visible progress and the eventual report, so do not emit
    # the legacy "task submitted" assistant message or persist an empty one.
    if result.deferred_to_worker:
        ctx.suppress_response = True
    if result.reset_provider_state and ctx.session is not None:
        ctx.session.provider_state = None
