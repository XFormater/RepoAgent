"""On-demand deep semantic-memory retrieval tool."""

from __future__ import annotations

from typing import Any

from nanobot.agent.tools.base import Tool, ToolResult
from nanobot.agent.tools.context import current_request_context
from nanobot.memory_system.service import LongTermMemoryService


class MemoryRetrieveTool(Tool):
    def __init__(
        self,
        service: LongTermMemoryService,
        *,
        unified_session: bool,
    ) -> None:
        self._service = service
        self._unified_session = unified_session

    @property
    def name(self) -> str:
        return "memory_retrieve"

    @property
    def description(self) -> str:
        return (
            "Retrieve durable user memories relevant to a question. Use when the "
            "current prompt needs older preferences, goals, facts, or multi-hop "
            "relationships that are not already present in the conversation."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4000,
                    "description": "Focused semantic memory question.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, **kwargs: Any) -> str:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return ToolResult.error("query must not be empty")
        request = current_request_context()
        if request is None or not request.sender_id:
            return ToolResult.error("memory retrieval requires a user request context")
        user_id = (
            request.sender_id
            if self._unified_session
            else f"{request.channel}:{request.sender_id}"
        )
        prompt = await self._service.retrieve_prompt(
            user_id=user_id,
            query=query,
            recent_messages=[],
            deep=True,
        )
        return prompt or "<memory>No relevant durable memory found.</memory>"
