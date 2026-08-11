"""Configuration for the optional repository-maintenance agent."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from nanobot.branding import AGENT_DISPLAY_NAME, AGENT_SLUG
from nanobot.config_base import Base


class RepositoryExpertConfig(Base):
    """One fixed expert available to the dispatcher."""

    enabled: bool = True
    description: str
    intents: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=5, ge=1, le=30)


def _default_experts() -> dict[str, RepositoryExpertConfig]:
    read_tools = ["repo_tree", "repo_search", "repo_read", "repo_git_history"]
    return {
        "repo_knowledge": RepositoryExpertConfig(
            description=(
                "Explain repository architecture, symbols, dependencies, configuration, "
                "and code paths using direct source evidence."
            ),
            intents=[
                "repository_overview",
                "architecture_analysis",
                "code_question",
                "symbol_location",
                "dependency_analysis",
            ],
            tools=read_tools,
        ),
        "issue_analyst": RepositoryExpertConfig(
            description=(
                "Analyze GitHub issues, reproduction details, affected code, related "
                "history, likely root causes, and implementation scope."
            ),
            intents=["issue_triage", "bug_diagnosis"],
            tools=[*read_tools, "github_get_issue"],
            max_iterations=7,
        ),
        "code": RepositoryExpertConfig(
            description=(
                "Implement a bounded code change in an isolated task workspace while "
                "following repository instructions and preserving unrelated work."
            ),
            intents=["code_change", "test_generation"],
            tools=[
                *read_tools,
                "read_file",
                "list_dir",
                "find_files",
                "grep",
                "write_file",
                "edit_file",
            ],
            max_iterations=12,
        ),
        "test": RepositoryExpertConfig(
            description=(
                "Inspect and, when a real sandbox is configured, execute the repository's "
                "tests and static checks. Never invent passing results."
            ),
            intents=["test_generation", "code_change", "bug_diagnosis"],
            tools=[*read_tools, "read_file", "list_dir", "find_files", "grep"],
            max_iterations=7,
        ),
        "review": RepositoryExpertConfig(
            description=(
                "Independently review the task diff for correctness, scope, security, "
                "compatibility, and test coverage."
            ),
            intents=["code_review", "code_change", "test_generation"],
            tools=[*read_tools, "repo_diff", "read_file", "grep"],
            max_iterations=7,
        ),
        "github": RepositoryExpertConfig(
            description=(
                "Prepare user-visible metadata for a controlled Fork, branch, Push, "
                "and Draft Pull Request workflow. External writes are deterministic."
            ),
            intents=["pull_request"],
            tools=["repo_diff"],
            max_iterations=3,
        ),
    }


class GitHubWorkflowConfig(Base):
    """GitHub write workflow settings."""

    auth_mode: Literal["gh"] = "gh"
    fork_owner: str | None = None
    default_branch_only: bool = True
    draft_pr_only: bool = True
    allow_merge: bool = False
    allow_release: bool = False
    allow_force_push: bool = False
    approval_ttl_seconds: int = Field(default=600, ge=60, le=86_400)
    commit_author_name: str = AGENT_DISPLAY_NAME
    commit_author_email: str = f"{AGENT_SLUG}@users.noreply.github.com"
    branch_prefix: str = AGENT_SLUG


class RepositoryResearchConfig(Base):
    """Academic-direction GitHub discovery and comparative analysis settings."""

    enabled: bool = True
    search_results_per_query: int = Field(default=10, ge=1, le=100)
    max_candidates: int = Field(default=20, ge=1, le=100)
    max_repositories: int = Field(default=5, ge=1, le=20)
    max_concurrent_downloads: int = Field(default=3, ge=1, le=10)
    max_concurrent_analyses: int = Field(default=2, ge=1, le=10)
    analysis_iterations: int = Field(default=5, ge=1, le=15)
    minimum_stars: int = Field(default=0, ge=0)
    require_license: bool = False
    github_token_env: str = "GITHUB_TOKEN"
    request_timeout_seconds: float = Field(default=30.0, ge=1.0, le=120.0)


class RepositoryAgentConfig(Base):
    """Top-level repository agent configuration."""

    enabled: bool = False
    cache_dir: str = ".nanobot/repositories"
    state_dir: str = ".nanobot/repository-agent"
    allowed_hosts: list[str] = Field(default_factory=lambda: ["github.com"])
    max_repository_size_mb: int = Field(default=500, ge=1, le=10_000)
    max_files: int = Field(default=50_000, ge=100, le=1_000_000)
    clone_depth: int = Field(default=1, ge=1, le=1_000)
    intent_strategy: Literal["llm", "hybrid", "rules"] = "hybrid"
    max_plan_steps: int = Field(default=8, ge=1, le=30)
    simple_max_iterations: int = Field(default=6, ge=1, le=20)
    deletion_approval_ttl_seconds: int = Field(default=600, ge=60, le=86_400)
    test_execution: Literal["disabled", "sandboxed", "host"] = "disabled"
    experts: dict[str, RepositoryExpertConfig] = Field(default_factory=_default_experts)
    research: RepositoryResearchConfig = Field(default_factory=RepositoryResearchConfig)
    github: GitHubWorkflowConfig = Field(default_factory=GitHubWorkflowConfig)
