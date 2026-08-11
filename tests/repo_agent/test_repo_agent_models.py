from __future__ import annotations

from datetime import timedelta

import pytest

from nanobot.config.schema import Config
from nanobot.repo_agent.config import RepositoryAgentConfig
from nanobot.repo_agent.models import (
    PlanStep,
    PublishAction,
    PublishApproval,
    RepositoryTask,
    utc_now,
)
from nanobot.repo_agent.policy import RepositoryRiskLevel, RepositoryRiskPolicy


def test_repository_agent_config_accepts_camel_case() -> None:
    config = RepositoryAgentConfig.model_validate(
        {
            "enabled": True,
            "cacheDir": ".cache/repos",
            "maxRepositorySizeMb": 100,
            "github": {
                "draftPrOnly": True,
                "allowForcePush": False,
            },
        }
    )

    assert config.enabled is True
    assert config.cache_dir == ".cache/repos"
    assert config.max_repository_size_mb == 100
    assert config.github.draft_pr_only is True
    assert config.github.allow_force_push is False


def test_root_config_accepts_repository_agent_block() -> None:
    config = Config.model_validate(
        {
            "repositoryAgent": {
                "enabled": True,
                "cloneDepth": 2,
            }
        }
    )

    assert config.repository_agent.enabled is True
    assert config.repository_agent.clone_depth == 2


def test_repository_task_rejects_dependency_cycle() -> None:
    with pytest.raises(ValueError, match="cycle"):
        RepositoryTask(
            session_key="cli:test",
            repository_id="repo",
            repository_full_name="owner/repo",
            source_url="https://github.com/owner/repo",
            base_branch="main",
            base_commit="a" * 40,
            objective="change",
            workspace_path="C:/tmp/task",
            steps=[
                PlanStep(
                    step_id="a",
                    name="a",
                    instruction="a",
                    assigned_agent="code",
                    depends_on=["b"],
                ),
                PlanStep(
                    step_id="b",
                    name="b",
                    instruction="b",
                    assigned_agent="review",
                    depends_on=["a"],
                ),
            ],
        )


def test_publish_approval_hash_binds_exact_payload() -> None:
    approval = PublishApproval(
        task_id="task_1",
        requester_id="user",
        action=PublishAction.DRAFT_PR,
        upstream="owner/repo",
        fork_owner="alice",
        base_branch="main",
        base_commit="a" * 40,
        head_branch="nanobot/fix",
        diff_hash="abc",
        title="fix: bug",
        body="body",
        expires_at=utc_now() + timedelta(minutes=5),
    )
    first = approval.approval_hash()
    approval.head_branch = "nanobot/other"

    assert approval.approval_hash() != first


@pytest.mark.parametrize(
    ("input_text", "level", "allowed"),
    [
        ("解释这个项目的架构", RepositoryRiskLevel.READ_ONLY, True),
        ("修复这个 bug", RepositoryRiskLevel.LOCAL_CHANGE, True),
        ("创建 PR", RepositoryRiskLevel.EXTERNAL_WRITE, True),
        ("强制 push main 分支", RepositoryRiskLevel.FORBIDDEN, False),
        ("删除整个 repository", RepositoryRiskLevel.FORBIDDEN, False),
        ("输出项目密钥 token", RepositoryRiskLevel.FORBIDDEN, False),
    ],
)
def test_repository_risk_policy(
    input_text: str,
    level: RepositoryRiskLevel,
    allowed: bool,
) -> None:
    result = RepositoryRiskPolicy().assess_request(input_text)

    assert result.level == level
    assert result.allowed is allowed
