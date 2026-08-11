"""Narrow AgentLoop integration helpers kept outside the critical core."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, cast

from nanobot.memory_system.models import (
    ConversationMessage,
    SessionContextState,
    ToolExecutionState,
)
from nanobot.memory_system.observability import logger
from nanobot.runtime_context import RuntimeContextBlock

if TYPE_CHECKING:
    from nanobot.agent.loop import AgentLoop, TurnContext


async def retrieve_for_turn(
    loop: AgentLoop,
    ctx: TurnContext,
) -> RuntimeContextBlock | None:
    service = loop.long_term_memory
    if service is None:
        return None
    blocks: list[str] = []
    if service.short_term is not None:
        try:
            short_term_context = await service.short_term.build_runtime_context(
                ctx.session_key
            )
            if short_term_context:
                blocks.append(short_term_context)
            await service.short_term.put_run_state(
                ctx.session_key,
                {
                    "status": "running",
                    "turn_id": ctx.turn_id,
                    "phase": "context_prepared",
                },
            )
        except Exception as exc:
            logger.warning(
                "short_term_memory_retrieve_failed",
                session_key=ctx.session_key,
                error=str(exc),
            )
    session = ctx.require_session()
    first_turn = not any(
        message.get("role") == "user" for message in session.messages
    )
    deep = ctx.attributes.get("memory_deep_retrieve") is True
    if service.should_retrieve(first_turn=first_turn, deep=deep):
        recent = [
            text
            for message in ctx.history
            if message.get("role") in {"user", "assistant"}
            if (text := _message_text(message.get("content")))
        ]
        prompt = await service.retrieve_prompt(
            user_id=_memory_user_id(loop, ctx),
            query=ctx.msg.content,
            recent_messages=recent,
            deep=deep,
        )
        if prompt:
            blocks.append(prompt)
    if not blocks:
        return None
    return RuntimeContextBlock(source="memory_system", content="\n\n".join(blocks))


async def record_for_turn(loop: AgentLoop, ctx: TurnContext) -> None:
    service = loop.long_term_memory
    if service is None:
        return
    if service.short_term is not None:
        try:
            session = ctx.require_session()
            goal_state = session.metadata.get("goal_state")
            goal_text = None
            if isinstance(goal_state, dict):
                raw_goal = cast(dict[str, Any], goal_state).get("objective")
                goal_text = raw_goal if isinstance(raw_goal, str) else None
            await service.short_term.put_context(
                ctx.session_key,
                SessionContextState(
                    current_goal=goal_text,
                    confirmed_context={
                        "channel": ctx.delivery.route.channel,
                        "sender_id": ctx.msg.sender_id,
                    },
                    last_summary=ctx.pending_summary,
                ),
            )
            await service.short_term.append_message(
                ctx.session_key,
                ConversationMessage(
                    message_id=str(
                        ctx.msg.metadata.get("message_id") or f"{ctx.turn_id}:user"
                    ),
                    role="user",
                    content=ctx.msg.content,
                ),
            )
            if ctx.final_content:
                await service.short_term.append_message(
                    ctx.session_key,
                    ConversationMessage(
                        message_id=f"{ctx.turn_id}:assistant",
                        role="assistant",
                        content=ctx.final_content,
                    ),
                )
            tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
            for message in ctx.all_messages:
                calls_value = message.get("tool_calls")
                if not isinstance(calls_value, list):
                    continue
                for call_value in cast(list[object], calls_value):
                    if not isinstance(call_value, dict):
                        continue
                    call = cast(dict[str, Any], call_value)
                    if not call.get("id"):
                        continue
                    function_value = call.get("function")
                    if not isinstance(function_value, dict):
                        continue
                    function = cast(dict[str, Any], function_value)
                    arguments = function.get("arguments")
                    if isinstance(arguments, str):
                        try:
                            parsed_value: object = json.loads(arguments)
                        except ValueError:
                            parsed_value = cast(object, {"raw": arguments})
                    else:
                        parsed_value = (
                            cast(object, arguments)
                            if isinstance(arguments, dict)
                            else cast(object, {})
                        )
                    parsed = (
                        cast(dict[str, Any], parsed_value)
                        if isinstance(parsed_value, dict)
                        else {}
                    )
                    tool_calls[str(call["id"])] = (
                        str(function.get("name") or "unknown"),
                        parsed,
                    )
            for message in ctx.all_messages:
                if message.get("role") != "tool" or not message.get("tool_call_id"):
                    continue
                call_id = str(message["tool_call_id"])
                tool_name, arguments = tool_calls.get(
                    call_id,
                    (str(message.get("name") or "unknown"), {}),
                )
                result_text = _message_text(message.get("content"))
                result_text = result_text[: service.settings.short_term.max_tool_result_chars]
                failed = bool(message.get("is_error"))
                await service.short_term.put_tool_execution(
                    ctx.session_key,
                    ToolExecutionState(
                        execution_id=call_id,
                        tool_name=tool_name,
                        arguments=arguments,
                        result=result_text,
                        status="failed" if failed else "succeeded",
                        error=result_text if failed else None,
                    ),
                )
            await service.short_term.put_run_state(
                ctx.session_key,
                {
                    "status": "completed",
                    "turn_id": ctx.turn_id,
                    "phase": "turn_saved",
                    "tools_used": sorted(
                        {tool_name for tool_name, _arguments in tool_calls.values()}
                    ),
                    "has_response": bool(ctx.final_content),
                },
            )
        except Exception as exc:
            logger.warning(
                "short_term_memory_record_failed",
                session_key=ctx.session_key,
                error=str(exc),
            )
    if not service.settings.record.enabled:
        return
    history_limit = service.settings.record.recent_turns * 2
    raw_messages = [
        message
        for message in ctx.history[-history_limit:]
        if message.get("role") in {"user", "assistant"}
    ]
    messages: list[dict[str, str]] = []
    for index, message in enumerate(raw_messages):
        text = _message_text(message.get("content"))
        if not text:
            continue
        timestamp = str(message.get("timestamp", ""))
        stable_id = hashlib.sha256(
            f"{ctx.session_key}\x1f{timestamp}\x1f{index}\x1f{text}".encode("utf-8")
        ).hexdigest()
        messages.append(
            {
                "id": stable_id,
                "role": str(message.get("role")),
                "content": text,
            }
        )
    user_message_id = str(
        ctx.msg.metadata.get("message_id") or f"{ctx.turn_id}:user"
    )
    messages.append(
        {
            "id": user_message_id,
            "role": "user",
            "content": ctx.msg.content,
        }
    )
    if ctx.final_content:
        messages.append(
            {
                "id": f"{ctx.turn_id}:assistant",
                "role": "assistant",
                "content": ctx.final_content,
            }
        )
    operation = service.record_turn(
        user_id=_memory_user_id(loop, ctx),
        messages=messages,
        batch_id=hashlib.sha256(
            f"{ctx.session_key}\x1f{ctx.turn_id}".encode("utf-8")
        ).hexdigest(),
    )
    if service.settings.record.mode == "synchronous":
        await operation
    else:
        loop.schedule_background(operation)


def _memory_user_id(loop: AgentLoop, ctx: TurnContext) -> str:
    if loop.unified_session:
        return ctx.msg.sender_id
    return f"{ctx.delivery.route.channel}:{ctx.msg.sender_id}"


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    blocks = cast(list[object], content)
    parts = [
        str(block_data.get("text", "")).strip()
        for block in blocks
        if isinstance(block, dict)
        for block_data in (cast(dict[str, Any], block),)
        if block_data.get("type") == "text"
    ]
    return "\n".join(part for part in parts if part)
