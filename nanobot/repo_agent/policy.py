"""Non-bypassable policy for repository and GitHub operations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import IntEnum


class RepositoryRiskLevel(IntEnum):
    READ_ONLY = 1
    LOCAL_CHANGE = 2
    EXTERNAL_WRITE = 3
    FORBIDDEN = 4


@dataclass(frozen=True, slots=True)
class RepositoryRiskAssessment:
    level: RepositoryRiskLevel
    allowed: bool
    requires_confirmation: bool
    reasons: tuple[str, ...]


class RepositoryRiskPolicy:
    """Hard safety policy that runs before any model or external write."""

    _FORBIDDEN = (
        re.compile(r"\bforce[\s-]*push\b", re.IGNORECASE),
        re.compile(r"(?:强制|強制).{0,8}push\b", re.IGNORECASE),
        re.compile(r"\bgit\s+push\b.*(?:--force|-f)\b", re.IGNORECASE),
        re.compile(
            r"(?:删除|刪除).{0,12}(?:项目|項目|仓库|倉庫|代码库|代碼庫|repository)",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bdelete.{0,12}(?:project|repo|repository)\b",
            re.IGNORECASE,
        ),
        re.compile(r"(?:泄露|输出|导出|获取).{0,16}(?:密钥|令牌|凭证|secret|token)", re.IGNORECASE),
        re.compile(r"\b(?:exfiltrate|reveal|dump).{0,16}(?:secret|token|credential)\b", re.IGNORECASE),
        re.compile(r"(?:绕过|繞過).{0,12}(?:分支保护|保护规则|审批)", re.IGNORECASE),
        re.compile(r"\bbypass.{0,16}(?:branch protection|review|approval)\b", re.IGNORECASE),
    )

    def assess_request(self, request: str) -> RepositoryRiskAssessment:
        matched = [pattern.pattern for pattern in self._FORBIDDEN if pattern.search(request)]
        if matched:
            return RepositoryRiskAssessment(
                level=RepositoryRiskLevel.FORBIDDEN,
                allowed=False,
                requires_confirmation=False,
                reasons=tuple(f"命中禁止规则：{value}" for value in matched),
            )
        lowered = request.casefold()
        if any(
            marker in lowered
            for marker in (
                "fork",
                "push",
                "pull request",
                "创建pr",
                "创建 pr",
                "提交pr",
                "提交 pr",
            )
        ):
            return RepositoryRiskAssessment(
                level=RepositoryRiskLevel.EXTERNAL_WRITE,
                allowed=True,
                requires_confirmation=True,
                reasons=("请求包含 GitHub 外部写操作。",),
            )
        if any(
            marker in lowered
            for marker in ("修改", "修复", "添加", "实现", "重构", "fix ", "implement ")
        ):
            return RepositoryRiskAssessment(
                level=RepositoryRiskLevel.LOCAL_CHANGE,
                allowed=True,
                requires_confirmation=False,
                reasons=("请求包含隔离工作区内的代码修改。",),
            )
        return RepositoryRiskAssessment(
            level=RepositoryRiskLevel.READ_ONLY,
            allowed=True,
            requires_confirmation=False,
            reasons=("只读仓库分析。",),
        )
