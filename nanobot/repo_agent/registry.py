"""Fixed repository expert registry and deterministic dispatch."""

from __future__ import annotations

from dataclasses import dataclass

from nanobot.agent.tools.registry import ToolRegistry
from nanobot.repo_agent.config import RepositoryAgentConfig, RepositoryExpertConfig
from nanobot.repo_agent.models import RepositoryIntent


@dataclass(frozen=True, slots=True)
class RepositoryExpert:
    name: str
    description: str
    intents: frozenset[str]
    tools: tuple[str, ...]
    max_iterations: int


class RepositoryExpertRegistry:
    def __init__(self, config: RepositoryAgentConfig, tools: ToolRegistry):
        self._experts: dict[str, RepositoryExpert] = {}
        for name, value in config.experts.items():
            if not value.enabled:
                continue
            expert = self._from_config(name, value)
            if name == "test" and config.test_execution != "disabled":
                exec_tool = tools.get("exec")
                sandbox = getattr(exec_tool, "sandbox", None) if exec_tool is not None else None
                execution_allowed = (
                    config.test_execution == "host"
                    or (config.test_execution == "sandboxed" and bool(sandbox))
                )
                if execution_allowed:
                    expert = RepositoryExpert(
                        name=expert.name,
                        description=expert.description,
                        intents=expert.intents,
                        tools=tuple(dict.fromkeys([*expert.tools, "exec"])),
                        max_iterations=expert.max_iterations,
                    )
            self._experts[name] = expert

    @staticmethod
    def _from_config(name: str, value: RepositoryExpertConfig) -> RepositoryExpert:
        return RepositoryExpert(
            name=name,
            description=value.description,
            intents=frozenset(value.intents),
            tools=tuple(value.tools),
            max_iterations=value.max_iterations,
        )

    @property
    def names(self) -> list[str]:
        return sorted(self._experts)

    def get(self, name: str) -> RepositoryExpert | None:
        return self._experts.get(name)

    def dispatch(self, intent: RepositoryIntent) -> RepositoryExpert | None:
        if intent.preferred_agent:
            preferred = self.get(intent.preferred_agent)
            if preferred is not None and intent.intent.value in preferred.intents:
                return preferred
        candidates = sorted(
            (
                expert
                for expert in self._experts.values()
                if intent.intent.value in expert.intents
            ),
            key=lambda expert: expert.name,
        )
        return candidates[0] if candidates else self.get("repo_knowledge")

