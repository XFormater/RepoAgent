from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from nanobot.repo_agent.models import RepositoryDeletionApproval, RepositoryRecord, utc_now
from nanobot.repo_agent.store import RepositoryStateStore


def test_store_persists_and_clears_active_repository(tmp_path: Path) -> None:
    store = RepositoryStateStore(tmp_path, ".state")
    record = RepositoryRecord(
        repository_id="repo",
        url="https://github.com/owner/repo",
        owner="owner",
        name="repo",
        default_branch="main",
        commit_sha="a" * 40,
        local_path=str(tmp_path / "repo"),
        file_count=2,
        size_bytes=100,
    )

    store.set_active_repository("cli:user", record)
    loaded = store.active_repository("cli:user")
    store.clear_active_repository("cli:user")

    assert loaded is not None
    assert loaded.full_name == "owner/repo"
    assert store.active_repository("cli:user") is None
    assert store.audit_path.exists()


def test_store_persists_deletion_approval(tmp_path: Path) -> None:
    store = RepositoryStateStore(tmp_path, ".state")
    approval = RepositoryDeletionApproval(
        requester_id="user",
        repository_id="repo",
        repository_full_name="owner/repo",
        repository_url="https://github.com/owner/repo",
        source_path=str(tmp_path / "cache" / "source"),
        cache_path=str(tmp_path / "cache"),
        commit_sha="a" * 40,
        expires_at=utc_now() + timedelta(minutes=10),
    )
    approval.binding_hash = approval.approval_hash()

    store.save_deletion_approval(
        approval,
        event="repository.delete.approval_requested",
    )
    loaded = store.load_deletion_approval(approval.approval_id)

    assert loaded is not None
    assert loaded.binding_hash == loaded.approval_hash()
    assert loaded.repository_full_name == "owner/repo"
