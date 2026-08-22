"""Model-message governance for agent runner requests.

This module owns model-facing message shaping and tool-result content normalization.
It may return copied messages or persisted-result placeholders, but it must not
mutate an existing session history list in place.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from loguru import logger

from nanobot.utils.helpers import (
    estimate_message_tokens,
    estimate_prompt_tokens_chain,
    find_legal_message_start,
    maybe_persist_tool_result,
    truncate_text,
)
from nanobot.utils.runtime import ensure_nonempty_tool_result

if TYPE_CHECKING:
    from nanobot.agent.tools.registry import ToolRegistry
    from nanobot.providers.base import LLMProvider

SNIP_SAFETY_BUFFER = 1024
MICROCOMPACT_MIN_CHARS = 500
INFLIGHT_COMPACT_TARGET_RATIO = 0.85
MAX_RECALLED_HISTORY_BLOCKS = 8
MAX_RECALLED_TOOL_BLOCKS = 3
COMPACTABLE_TOOLS = frozenset({
    "read_file", "exec", "grep", "find_files",
    "web_search", "web_fetch", "list_dir", "list_exec_sessions",
})
# read_file is the recovery path for persisted results; exempting it prevents persist->read->persist loops.
TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS = frozenset({"read_file"})
BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"
PLACEHOLDER_TEXTS = frozenset({
    "[Previous assistant message omitted.]",
})


def _tool_call_name_is_valid(tool_call: Any) -> bool:
    """Whether a persisted OpenAI-style tool_call carries a usable name.

    Mirrors ``ToolCallRequest.has_valid_name`` for the dict shape stored in
    message history: a degenerate call with ``name=None`` / ``""`` cannot be
    executed and is rejected by upstream APIs if replayed.
    """
    if not isinstance(tool_call, dict):
        return False
    tool_call_data = cast(dict[str, Any], tool_call)
    fn = tool_call_data.get("function")
    name = cast(dict[str, Any], fn).get("name") if isinstance(fn, dict) else tool_call_data.get("name")
    return isinstance(name, str) and bool(name)


@dataclass(slots=True)
class ContextGovernanceConfig:
    provider: LLMProvider
    model: str
    tools: ToolRegistry
    workspace: Path | None
    session_key: str | None
    max_tool_result_chars: int
    context_window_tokens: int | None = None
    context_block_limit: int | None = None
    max_tokens: int | None = None
    inflight_start_index: int = 0


class ContextGovernor:
    """Prepare model-copy messages while preserving persisted history."""

    def prepare_for_model(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        updated = self.strip_placeholder_assistant_messages(messages)
        updated = self.strip_malformed_tool_calls(updated)
        updated = self.drop_orphan_tool_results(updated)
        updated = self.backfill_missing_tool_results(updated)
        updated = self.apply_tool_result_budget(config, updated)
        updated = self.compact_inflight_overflow(config, updated, compacted_tool_call_ids)
        updated = self.snip_history(config, updated)
        updated = self.drop_orphan_tool_results(updated)
        return self.backfill_missing_tool_results(updated)

    @staticmethod
    def input_budget(config: ContextGovernanceConfig) -> int:
        if not config.context_window_tokens:
            return 0

        provider_max_tokens = getattr(
            getattr(config.provider, "generation", None),
            "max_tokens",
            4096,
        )
        max_output = config.max_tokens if isinstance(config.max_tokens, int) else (
            provider_max_tokens if isinstance(provider_max_tokens, int) else 4096
        )
        budget = config.context_block_limit or (
            config.context_window_tokens - max_output - SNIP_SAFETY_BUFFER
        )
        return budget if budget > 0 else 0

    @staticmethod
    def normalize_tool_result(
        config: ContextGovernanceConfig,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> Any:
        result = ensure_nonempty_tool_result(tool_name, result)
        if tool_name in TOOL_RESULT_OFFLOAD_EXEMPT_TOOLS:
            return result
        try:
            content = maybe_persist_tool_result(
                config.workspace,
                config.session_key,
                tool_call_id,
                result,
                max_chars=config.max_tool_result_chars,
            )
        except Exception:
            logger.exception(
                "Tool result persist failed for {} in {}; using raw result",
                tool_call_id,
                config.session_key or "default",
            )
            content = result
        if isinstance(content, str) and len(content) > config.max_tool_result_chars:
            return truncate_text(content, config.max_tool_result_chars)
        return content

    @staticmethod
    def strip_placeholder_assistant_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Remove assistant messages that are compaction placeholders.

        Messages like ``[Previous assistant message omitted.]`` carry no useful
        context for the model and can cause it to repeatedly attempt tool calls
        that previously failed, producing malformed responses in a loop.
        Consecutive same-role messages that result from removal are handled
        downstream by the provider's merge-consecutive logic. Only the
        model-facing copy is repaired; the persisted transcript is untouched
        (a copy is returned, or the same list object when nothing changes).
        """
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                if updated is not None:
                    updated.append(msg)
                continue
            content = msg.get("content", "")
            text = content if isinstance(content, str) else ""
            is_placeholder = text.strip() in PLACEHOLDER_TEXTS
            has_tool_calls = bool(msg.get("tool_calls"))
            if is_placeholder and not has_tool_calls:
                if updated is None:
                    updated = list(messages[:idx])
                logger.debug(
                    "Stripping placeholder assistant message from history: {!r}",
                    text[:60],
                )
                continue
            if updated is not None:
                updated.append(msg)
        if updated is None:
            return messages
        return updated

    @staticmethod
    def strip_malformed_tool_calls(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop persisted assistant tool_calls whose name is missing/non-string.

        A degenerate tool call (``name=None`` or ``""``) that slipped into the
        saved history before this guard existed gets replayed on every turn and
        makes upstream APIs reject the whole request
        (``messages.content.N.tool_use.name: Input should be a valid string``),
        permanently wedging the session. Removing the bad call here lets the
        existing orphan-result cleanup drop its now-dangling tool result, so a
        polluted session self-heals on its next turn. The persisted transcript
        is left untouched; only the model-facing copy is repaired (a copy is
        returned, or the same list object when nothing changes).
        """
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                if updated is not None:
                    updated.append(msg)
                continue
            calls = msg.get("tool_calls")
            if not calls:
                if updated is not None:
                    updated.append(msg)
                continue
            kept = [tc for tc in cast(list[Any], calls) if _tool_call_name_is_valid(tc)]
            if len(kept) == len(calls):
                if updated is not None:
                    updated.append(msg)
                continue
            if updated is None:
                updated = [dict(m) for m in messages[:idx]]
            logger.warning(
                "Stripping {} malformed tool_call(s) with missing/non-string "
                "name from assistant history before request",
                len(calls) - len(kept),
            )
            repaired = dict(msg)
            if kept:
                repaired["tool_calls"] = kept
            else:
                repaired.pop("tool_calls", None)
            # An assistant turn with neither content nor any valid tool call is
            # itself invalid upstream; drop it entirely in that case.
            has_content = bool(repaired.get("content"))
            if not kept and not has_content:
                continue
            updated.append(repaired)

        if updated is None:
            return messages
        return updated

    @staticmethod
    def drop_orphan_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Drop invalid tool results before history is sent back to providers."""
        declared: set[str] = set()
        fulfilled: set[str] = set()
        updated: list[dict[str, Any]] | None = None
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in cast(list[Any], msg.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        tool_call = cast(dict[str, Any], tc)
                        if tool_call.get("id"):
                            declared.add(str(tool_call["id"]))
            if role == "tool":
                tid = msg.get("tool_call_id")
                tid_str = str(tid) if tid else ""
                if not tid_str or tid_str not in declared or tid_str in fulfilled:
                    if updated is None:
                        updated = [dict(m) for m in messages[:idx]]
                    continue
                fulfilled.add(tid_str)
            if updated is not None:
                updated.append(dict(msg))

        if updated is None:
            return messages
        return updated

    @staticmethod
    def backfill_missing_tool_results(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Insert synthetic error results for assistant tool_calls with missing tool outputs."""
        declared: list[tuple[int, str, str]] = []
        fulfilled: set[str] = set()
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in cast(list[Any], msg.get("tool_calls") or []):
                    if isinstance(tc, dict):
                        name = ""
                        tool_call = cast(dict[str, Any], tc)
                        if tool_call.get("id"):
                            func = tool_call.get("function")
                            if isinstance(func, dict):
                                func_data = cast(dict[str, Any], func)
                                raw_name = func_data.get("name", "")
                                name = raw_name if isinstance(raw_name, str) else str(raw_name)
                            declared.append((idx, str(tool_call["id"]), name))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))

        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]
        if not missing:
            return messages

        updated = list(messages)
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": BACKFILL_CONTENT,
            })
            offset += 1
        return updated

    def apply_tool_result_budget(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        updated = messages
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self.normalize_tool_result(
                config,
                str(message.get("tool_call_id") or f"tool_{idx}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[idx]["content"] = normalized
        return updated

    def compact_inflight_overflow(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        """Compact in-flight tool results only when the request would overflow."""
        budget = self.input_budget(config)
        if budget <= 0:
            return messages

        tools = config.tools.get_definitions()
        updated = self._apply_recorded_compactions(messages, compacted_tool_call_ids)
        estimate, source = estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            updated,
            tools,
        )
        if estimate <= budget:
            return updated

        target = int(budget * INFLIGHT_COMPACT_TARGET_RATIO)
        candidates = self._inflight_compaction_candidates(
            config,
            updated,
            compacted_tool_call_ids,
        )
        if not candidates:
            return updated

        for candidate_idx, (idx, tool_call_id) in enumerate(candidates):
            is_newest_candidate = candidate_idx == len(candidates) - 1
            if is_newest_candidate and estimate <= budget:
                break
            if tool_call_id in compacted_tool_call_ids:
                continue
            if updated is messages:
                updated = [dict(m) for m in messages]
            compacted_tool_call_ids.add(tool_call_id)
            self._compact_tool_result_at(updated, idx)
            estimate, source = estimate_prompt_tokens_chain(
                config.provider,
                config.model,
                updated,
                tools,
            )
            if estimate <= target:
                break

        logger.debug(
            "In-flight context compaction for {}: prompt={} budget={} target={} via {}, ids={}",
            config.session_key or "default",
            estimate,
            budget,
            target,
            source,
            len(compacted_tool_call_ids),
        )
        return updated

    def snip_history(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not messages or not config.context_window_tokens:
            return messages

        budget = self.input_budget(config)
        if budget <= 0:
            return messages

        tools = config.tools.get_definitions()
        estimate, _ = estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            messages,
            tools,
        )
        if estimate <= budget:
            return messages

        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [
            (idx, dict(msg))
            for idx, msg in enumerate(messages)
            if msg.get("role") != "system"
        ]
        if not non_system:
            return messages

        system_tokens = sum(estimate_message_tokens(msg) for msg in system_messages)
        fixed_tokens, _ = estimate_prompt_tokens_chain(
            config.provider,
            config.model,
            system_messages,
            tools,
        )
        remaining_budget = max(0, budget - max(system_tokens, fixed_tokens))
        blocks = self._history_blocks(non_system)
        if not blocks:
            plain_non_system = [message for _, message in non_system]
            return system_messages + self._legal_history_tail([], plain_non_system)

        selected: set[int] = {len(blocks) - 1}
        used_tokens = self._block_tokens(blocks[-1][1])
        latest_user = next(
            (
                message
                for _, message in reversed(non_system)
                if message.get("role") == "user"
            ),
            None,
        )
        query_terms = self._relevance_terms(self._message_search_text(latest_user or {}))

        # Any messages produced during the current runner invocation are authoritative
        # in-flight state. They must not disappear merely because an old transcript
        # overflowed after a tool call.
        for block_index, (original_indexes, block) in enumerate(blocks[:-1]):
            if any(index >= config.inflight_start_index for index in original_indexes):
                block_tokens = self._block_tokens(block)
                if used_tokens + block_tokens <= remaining_budget:
                    selected.add(block_index)
                    used_tokens += block_tokens

        relevant = sorted(
            (
                (self._block_relevance(block, query_terms), block_index)
                for block_index, (_, block) in enumerate(blocks[:-1])
                if block_index not in selected
            ),
            key=lambda item: (item[0], item[1]),
            reverse=True,
        )
        recalled = 0
        for score, block_index in relevant:
            if score <= 0 or recalled >= MAX_RECALLED_HISTORY_BLOCKS:
                break
            block_tokens = self._block_tokens(blocks[block_index][1])
            if used_tokens + block_tokens <= remaining_budget:
                selected.add(block_index)
                used_tokens += block_tokens
                recalled += 1

        # Completed tool transactions are expensive to reproduce and may carry opaque
        # values that lexical relevance cannot connect to a later natural-language
        # question. Retain a bounded number of complete tool-containing user turns.
        tool_candidates = [
            block_index
            for block_index, (_, block) in enumerate(blocks[:-1])
            if block_index not in selected and self._block_has_tool_exchange(block)
        ][-MAX_RECALLED_TOOL_BLOCKS:]
        for block_index in tool_candidates:
            block_tokens = self._block_tokens(blocks[block_index][1])
            if used_tokens + block_tokens <= remaining_budget:
                selected.add(block_index)
                used_tokens += block_tokens

        # Spend the remaining budget on a contiguous recent tail. Relevant historical
        # blocks above displace irrelevant tail blocks instead of increasing the cap.
        for block_index in range(len(blocks) - 2, -1, -1):
            if block_index in selected:
                continue
            block_tokens = self._block_tokens(blocks[block_index][1])
            if used_tokens + block_tokens > remaining_budget:
                break
            selected.add(block_index)
            used_tokens += block_tokens

        kept = [
            message
            for block_index, (_, block) in enumerate(blocks)
            if block_index in selected
            for message in block
        ]
        plain_non_system = [message for _, message in non_system]
        return system_messages + self._legal_history_tail(kept, plain_non_system)

    @staticmethod
    def _history_blocks(
        messages: list[tuple[int, dict[str, Any]]],
    ) -> list[tuple[list[int], list[dict[str, Any]]]]:
        """Group history into user-owned turns so tool protocols stay intact."""
        blocks: list[tuple[list[int], list[dict[str, Any]]]] = []
        current_indexes: list[int] = []
        current: list[dict[str, Any]] = []
        for original_index, message in messages:
            if message.get("role") == "user" and current:
                blocks.append((current_indexes, current))
                current_indexes = []
                current = []
            current_indexes.append(original_index)
            current.append(message)
        if current:
            blocks.append((current_indexes, current))
        return blocks

    @staticmethod
    def _block_tokens(block: list[dict[str, Any]]) -> int:
        return sum(estimate_message_tokens(message) for message in block)

    @staticmethod
    def _message_search_text(message: dict[str, Any]) -> str:
        content = message.get("content")
        parts = [content] if isinstance(content, str) else []
        for tool_call in cast(list[Any], message.get("tool_calls") or []):
            if not isinstance(tool_call, dict):
                continue
            function = cast(dict[str, Any], tool_call).get("function")
            if isinstance(function, dict):
                parts.extend(str(value) for value in cast(dict[str, Any], function).values())
        return " ".join(parts)

    @classmethod
    def _relevance_terms(cls, text: str) -> set[str]:
        ascii_terms = {
            token.casefold()
            for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.:/#|-]{2,}", text)
            if token.casefold()
            not in {
                "the", "and", "for", "only", "current", "what", "from",
                "this", "that", "with", "unknown", "output",
            }
        }
        cjk_chunks = re.findall(r"[\u3400-\u9fff]+", text)
        cjk_terms = {
            chunk[start : start + width]
            for chunk in cjk_chunks
            for width in (2, 3, 4)
            for start in range(max(0, len(chunk) - width + 1))
        }
        return ascii_terms | cjk_terms

    @classmethod
    def _block_relevance(cls, block: list[dict[str, Any]], query_terms: set[str]) -> int:
        if not query_terms:
            return 0
        text = " ".join(cls._message_search_text(message) for message in block)
        return len(query_terms & cls._relevance_terms(text))

    @staticmethod
    def _block_has_tool_exchange(block: list[dict[str, Any]]) -> bool:
        return any(message.get("role") == "tool" or message.get("tool_calls") for message in block)

    @staticmethod
    def _tool_result_compaction_message(message: dict[str, Any]) -> str:
        name = message.get("name", "tool")
        return (
            f"Error: The previous {name} result was compacted to fit context because it was too "
            "large. Do not repeat the same call unchanged. Retry with a narrower path, query, "
            "range, or result limit, use another tool, or tell the user the task cannot fit in "
            "the available context."
        )

    def _legal_history_tail(
        self,
        kept: list[dict[str, Any]],
        non_system: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fallback = kept if kept else (non_system[-1:] if non_system else [])
        kept = self._user_tail(kept) or self._user_tail(non_system, last=True) or fallback

        start = find_legal_message_start(kept)
        return kept[start:] if start else kept

    @staticmethod
    def _user_tail(messages: list[dict[str, Any]], *, last: bool = False) -> list[dict[str, Any]]:
        indexes = range(len(messages) - 1, -1, -1) if last else range(len(messages))
        for idx in indexes:
            if messages[idx].get("role") == "user":
                return messages[idx:]
        return []

    def _apply_recorded_compactions(
        self,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[dict[str, Any]]:
        if not compacted_tool_call_ids:
            return messages
        updated = messages
        for idx, msg in enumerate(messages):
            if msg.get("role") != "tool":
                continue
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id or str(tool_call_id) not in compacted_tool_call_ids:
                continue
            compaction_message = self._tool_result_compaction_message(msg)
            if msg.get("content") == compaction_message:
                continue
            if updated is messages:
                updated = [dict(m) for m in messages]
            updated[idx]["content"] = compaction_message
        return updated

    def _inflight_compaction_candidates(
        self,
        config: ContextGovernanceConfig,
        messages: list[dict[str, Any]],
        compacted_tool_call_ids: set[str],
    ) -> list[tuple[int, str]]:
        compactable: list[tuple[int, str]] = []
        for idx, msg in enumerate(messages):
            if idx < config.inflight_start_index:
                continue
            if msg.get("role") != "tool" or msg.get("name") not in COMPACTABLE_TOOLS:
                continue
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id or str(tool_call_id) in compacted_tool_call_ids:
                continue
            content = msg.get("content")
            if not isinstance(content, str) or len(content) < MICROCOMPACT_MIN_CHARS:
                continue
            compactable.append((idx, str(tool_call_id)))

        return compactable

    def _compact_tool_result_at(self, messages: list[dict[str, Any]], idx: int) -> None:
        messages[idx]["content"] = self._tool_result_compaction_message(messages[idx])
