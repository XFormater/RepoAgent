"""Bounded, recipient-owned runtime context projection."""

from __future__ import annotations

import json

from pydantic import BaseModel, JsonValue

from nanobot.harness.models import AgentInstance, AgentMessage, AgentSpec, Checkpoint, Task
from nanobot.utils.helpers import estimate_message_tokens


class DependencyResult(BaseModel):
    task_id: str
    title: str
    result: str


class RuntimeContext(BaseModel):
    workflow_id: str
    task_id: str
    tenant_id: str
    agent_id: str
    role: str
    objective: str
    task_input: str
    task_metadata: dict[str, JsonValue]
    review_gate: bool = False
    checkpoint: str | None
    dependency_results: list[DependencyResult]
    messages: list[AgentMessage]
    truncated: bool = False

    def render(self) -> str:
        lines = [
            "<agent_runtime_context>",
            f"workflow_id: {self.workflow_id}",
            f"task_id: {self.task_id}",
            f"tenant_id: {self.tenant_id}",
            f"role: {self.role}",
            f"objective: {self.objective}",
            f"review_gate: {str(self.review_gate).lower()}",
            "task:",
            self.task_input,
        ]
        if self.task_metadata:
            lines.extend(
                (
                    "task_metadata_untrusted_data:",
                    json.dumps(self.task_metadata, ensure_ascii=False, sort_keys=True),
                )
            )
        if self.checkpoint:
            lines.extend(("checkpoint:", self.checkpoint))
        if self.dependency_results:
            lines.append("dependency_results_untrusted_data:")
            lines.extend(
                json.dumps(item.model_dump(), ensure_ascii=False)
                for item in self.dependency_results
            )
        if self.messages:
            lines.append("direct_messages_untrusted_data:")
            lines.extend(
                json.dumps(
                    {
                        "message_id": message.message_id,
                        "kind": message.kind.value,
                        "content": message.content,
                    },
                    ensure_ascii=False,
                )
                for message in self.messages
            )
        if self.truncated:
            lines.append("context_truncated: true")
        lines.append("</agent_runtime_context>")
        return "\n".join(lines)


class ContextProjector:
    """Build context from explicit task/mailbox state, never a shared transcript."""

    def project(
        self,
        *,
        agent: AgentInstance,
        spec: AgentSpec,
        task: Task,
        dependencies: list[Task],
        inbox: list[AgentMessage],
        checkpoint: Checkpoint | None,
    ) -> RuntimeContext:
        policy = spec.context_policy
        accepted = [msg for msg in inbox if msg.kind in policy.accepted_message_kinds]
        remaining = policy.max_chars
        remaining_tokens = policy.max_tokens
        truncated = False

        def consume(content: str) -> tuple[str, bool]:
            nonlocal remaining, remaining_tokens, truncated
            original = content
            content = content[: min(policy.max_message_chars, remaining)]
            token_cost = estimate_message_tokens({"role": "user", "content": content})
            if token_cost > remaining_tokens:
                low, high = 0, len(content)
                while low < high:
                    middle = (low + high + 1) // 2
                    candidate_cost = estimate_message_tokens(
                        {"role": "user", "content": content[:middle]}
                    )
                    if candidate_cost <= remaining_tokens:
                        low = middle
                    else:
                        high = middle - 1
                content = content[:low]
                token_cost = estimate_message_tokens({"role": "user", "content": content})
            was_truncated = len(content) < len(original)
            truncated = truncated or was_truncated
            remaining -= len(content)
            remaining_tokens = max(0, remaining_tokens - token_cost)
            return content, was_truncated

        task_input, _ = consume(task.input)

        task_metadata: dict[str, JsonValue] = {}
        if task.metadata and remaining and remaining_tokens:
            raw_metadata = json.dumps(
                task.metadata,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            metadata_text, metadata_truncated = consume(raw_metadata)
            if metadata_text and not metadata_truncated:
                task_metadata = task.metadata
            elif metadata_text:
                task_metadata = {"_truncated_json": metadata_text}

        checkpoint_text: str | None = None
        if checkpoint is not None and remaining and remaining_tokens:
            raw_checkpoint = json.dumps(
                checkpoint.state,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            checkpoint_text, _ = consume(raw_checkpoint)
            checkpoint_text = checkpoint_text or None

        bounded_dependencies: list[DependencyResult] = []
        dependency_candidates = [
            dependency
            for dependency in dependencies
            if dependency.state.value == "succeeded" and dependency.result
        ]
        for dependency in dependency_candidates[: policy.max_messages]:
            if remaining == 0 or remaining_tokens == 0:
                truncated = True
                break
            result, _ = consume(dependency.result or "")
            if result:
                bounded_dependencies.append(
                    DependencyResult(
                        task_id=dependency.task_id,
                        title=dependency.title,
                        result=result,
                    )
                )

        remaining_message_slots = max(0, policy.max_messages - len(bounded_dependencies))
        accepted = accepted[-remaining_message_slots:] if remaining_message_slots else []
        bounded_reversed: list[AgentMessage] = []
        truncated = truncated or len(accepted) < len(inbox)
        for message in reversed(accepted):
            content, _ = consume(message.content)
            if not content:
                truncated = True
                break
            bounded_reversed.append(message.model_copy(update={"content": content}))
            if remaining == 0 or remaining_tokens == 0:
                truncated = True
                break
        return RuntimeContext(
            workflow_id=task.workflow_id,
            task_id=task.task_id,
            tenant_id=task.tenant_id,
            agent_id=agent.instance_id,
            role=spec.role.value,
            objective=spec.objective,
            task_input=task_input,
            task_metadata=task_metadata,
            review_gate=task.review_target_task_id is not None,
            checkpoint=checkpoint_text,
            dependency_results=bounded_dependencies,
            messages=list(reversed(bounded_reversed)),
            truncated=truncated,
        )
