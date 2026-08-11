import shlex
from pathlib import Path

from nanobot.config.schema import Config
from nanobot.harness.sandbox_executor import HarnessProcessSandboxExecutor, _redact


def _executor(config: Config, project: Path) -> HarnessProcessSandboxExecutor:
    executor = object.__new__(HarnessProcessSandboxExecutor)
    executor.config = config
    executor.settings = config.harness
    executor.project_root = project
    return executor


def test_sandbox_config_disables_stateful_services_and_nested_sandbox(tmp_path):
    config = Config.model_validate(
        {
            "agents": {"defaults": {"workspace": str(tmp_path)}},
            "tools": {"exec": {"sandbox": "bwrap", "allowedEnvKeys": ["SECRET"]}},
            "memory": {"enabled": True},
            "harness": {"enabled": True, "workerExecutionBackend": "bwrap"},
        }
    )
    executor = _executor(config, tmp_path)

    isolated = executor._sandbox_config(tmp_path / "repo")

    assert isolated["agents"]["defaults"]["workspace"] == str(tmp_path / "repo")
    assert isolated["agents"]["defaults"]["maxConcurrentSubagents"] == 1
    assert isolated["memory"] == {"enabled": False}
    assert isolated["harness"] == {"enabled": False, "backend": "memory"}
    assert isolated["repositoryAgent"] == {"enabled": False}
    assert isolated["tools"]["restrictToWorkspace"] is True
    assert isolated["tools"]["webuiAllowLocalServiceAccess"] is False
    assert isolated["tools"]["exec"]["sandbox"] == ""
    assert isolated["tools"]["exec"]["allowedEnvKeys"] == []


def test_bwrap_command_isolates_process_and_mounts_only_task_workspace_rw(tmp_path):
    project = tmp_path / "app"
    workspace = project / "research" / "repo"
    workspace.mkdir(parents=True)
    (project / ".env").write_text("SECRET=value")
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    config = Config.model_validate(
        {
            "agents": {"defaults": {"workspace": str(project)}},
            "harness": {
                "workerExecutionBackend": "bwrap",
                "sandboxNetwork": "none",
            },
        }
    )
    executor = _executor(config, project)

    command = executor._command(workspace, config_file)
    rendered = shlex.join(command)

    assert command[0] == "bwrap"
    assert "--unshare-pid" in command
    assert "--unshare-ipc" in command
    assert "--unshare-uts" in command
    assert "--unshare-net" in command
    assert "--clearenv" in command
    assert "--cap-drop ALL" in rendered
    assert f"--ro-bind {project} {project}" in rendered
    assert f"--bind {workspace} {workspace}" in rendered
    assert f"--ro-bind /dev/null {project / '.env'}" in rendered
    assert str(config_file) in command


def test_sandbox_error_redaction_hides_provider_keys():
    assert _redact("request failed for sk-1234567890abcdef") == "request failed for [REDACTED]"
