"""Process-isolated Harness task execution using bubblewrap."""

from __future__ import annotations

import asyncio
import json
import os
import re
import resource
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.agent.loop import AgentLoop
from nanobot.config.schema import Config
from nanobot.harness.agent_executor import HarnessAgentExecutor
from nanobot.harness.context import RuntimeContext
from nanobot.harness.models import AgentSpec, ReviewResult

_RESULT_PREFIX = "GITHUB_RESEARCH_AGENT_SANDBOX_RESULT="
_SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{12,}\b")


def _redact(value: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", value)


class HarnessProcessSandboxExecutor:
    """Run each projected child-agent context in a separate bwrap process.

    The polling Worker remains the durable control plane and keeps database
    credentials.  The child receives only its bounded RuntimeContext, its
    immutable AgentSpec, a task-scoped workspace mount, and a short-lived
    minimal runtime config.
    """

    isolation_mode = "bwrap_process"

    def __init__(self, loop: AgentLoop, config: Config) -> None:
        if shutil.which("bwrap") is None:
            raise RuntimeError("Harness bwrap execution requested but bwrap is not installed")
        self.loop = loop
        self.config = config
        self._workspace_resolver = HarnessAgentExecutor(loop)
        self.project_root = Path(__file__).resolve().parents[2]
        self.settings = config.harness

    def _sandbox_config(
        self,
        workspace: Path,
        *,
        repository_agent: bool = False,
    ) -> dict[str, Any]:
        """Return the smallest config required by an isolated child runtime."""
        data = self.config.model_dump(mode="json", by_alias=True, exclude_none=True)
        # A child only needs the provider selected for this runtime. Never copy
        # unrelated provider credentials into its short-lived config, even when
        # the parent has several fallback providers configured.
        active_provider = self.config.get_provider_name()
        providers = data.get("providers", {})
        if active_provider and isinstance(providers, dict):
            provider_value = providers.get(active_provider)
            if provider_value is None:
                provider_value = providers.get(
                    active_provider.replace("_", "-")
                )
            data["providers"] = (
                {active_provider: provider_value}
                if isinstance(provider_value, dict)
                else {}
            )
        else:
            data["providers"] = {}
        agents = data.get("agents", {})
        defaults = agents.get("defaults", {})
        defaults["workspace"] = str(workspace)
        defaults["maxConcurrentSubagents"] = 1
        # A Harness child is not allowed to create grandchildren.
        defaults["fallbackModels"] = defaults.get("fallbackModels", [])
        agents["defaults"] = defaults

        tools = data.get("tools", {})
        tools["restrictToWorkspace"] = True
        tools["webuiAllowLocalServiceAccess"] = False
        exec_config = tools.get("exec", {})
        # The complete child process is already inside bwrap.  A nested bwrap
        # would require restoring namespace capabilities to the child.
        exec_config["sandbox"] = ""
        exec_config["allowedEnvKeys"] = []
        tools["exec"] = exec_config

        # The child workspace is the checked-out repository, while the VFS
        # repository cache remains rooted at the parent service workspace.
        # Carry that root as an absolute path so repository read tools validate
        # the mounted checkout against the real cache rather than resolving
        # ``.nanobot/vfs`` relative to the child repository directory.
        vfs = dict(data.get("vfs", {}))
        if repository_agent:
            configured_root = Path(str(vfs.get("root", ".nanobot/vfs"))).expanduser()
            if not configured_root.is_absolute():
                configured_root = self.config.workspace_path / configured_root
            vfs["root"] = str(configured_root.resolve(strict=False))

        return {
            "agents": agents,
            "providers": data.get("providers", {}),
            "modelPresets": data.get("modelPresets", {}),
            "tools": tools,
            "memory": {"enabled": False},
            "harness": {"enabled": False, "backend": "memory"},
            "vfs": vfs,
            # Repository research analysis is read-only and its Harness
            # AgentSpec allowlist remains enforced in the child. It is enabled
            # only for a marked repository-research task so ordinary children
            # retain the minimal tool catalog.
            "repositoryAgent": (
                data.get("repositoryAgent", {"enabled": False})
                if repository_agent
                else {"enabled": False}
            ),
        }

    def _command(self, workspace: Path, config_path: Path) -> list[str]:
        project = self.project_root
        args = [
            "bwrap",
            "--unshare-pid",
            "--unshare-ipc",
            "--unshare-uts",
            "--hostname",
            "github-research-agent-task",
            "--new-session",
            "--die-with-parent",
            "--clearenv",
            "--setenv",
            "HOME",
            str(workspace),
            "--setenv",
            "LANG",
            os.environ.get("LANG", "C.UTF-8"),
            "--setenv",
            "PATH",
            "/usr/local/node/bin:/usr/local/bin:/usr/bin:/bin",
            "--setenv",
            "PYTHONPATH",
            str(project),
            "--setenv",
            "NANOBOT_WORKSPACE_SANDBOX_ENFORCED",
            "bwrap",
            "--cap-drop",
            "ALL",
        ]
        if self.settings.sandbox_network == "none":
            args.append("--unshare-net")
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        ):
            if value := os.environ.get(key):
                args.extend(("--setenv", key, value))

        args.extend(("--ro-bind", "/usr", "/usr"))
        for path in (
            "/bin",
            "/lib",
            "/lib64",
            "/etc/alternatives",
            "/etc/ssl/certs",
            "/etc/pki/tls/certs",
            "/etc/pki/ca-trust",
            "/etc/crypto-policies",
            "/etc/resolv.conf",
            "/etc/hosts",
            "/etc/nsswitch.conf",
            "/etc/ld.so.cache",
        ):
            args.extend(("--ro-bind-try", path, path))
        if Path("/usr/local/node").exists():
            args.extend(("--ro-bind-try", "/usr/local/node", "/usr/local/node"))

        args.extend(
            (
                "--proc",
                "/proc",
                "--dev",
                "/dev",
                "--tmpfs",
                "/tmp",
                "--tmpfs",
                str(project.parent),
                "--dir",
                str(project),
            )
        )
        if workspace == project:
            args.extend(("--bind", str(project), str(project)))
        else:
            args.extend(("--ro-bind", str(project), str(project)))
            args.extend(("--bind", str(workspace), str(workspace)))

        # The repository is visible, but service credentials stored alongside
        # it are over-mounted with an empty read-only file.
        for name in (".env", ".env.local", ".env.production", ".npmrc", ".pypirc"):
            target = project / name
            if target.exists():
                args.extend(("--ro-bind", "/dev/null", str(target)))

        args.extend(
            (
                "--dir",
                "/run",
                "--dir",
                "/run/github-research-agent",
                "--ro-bind",
                str(config_path),
                "/run/github-research-agent/config.json",
                "--chdir",
                str(workspace),
                "--",
                sys.executable,
                "-m",
                "nanobot.harness.sandbox_child",
                "--config",
                "/run/github-research-agent/config.json",
            )
        )
        return args

    def _limit_resources(self) -> None:
        memory = self.settings.sandbox_memory_mb * 1024 * 1024
        file_size = self.settings.sandbox_file_size_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(
            resource.RLIMIT_NPROC,
            (self.settings.sandbox_process_limit, self.settings.sandbox_process_limit),
        )
        resource.setrlimit(
            resource.RLIMIT_NOFILE,
            (self.settings.sandbox_open_files_limit, self.settings.sandbox_open_files_limit),
        )
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_size, file_size))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        os.umask(0o077)

    async def __call__(self, context: RuntimeContext, spec: AgentSpec) -> str | ReviewResult:
        workspace, _ = self._workspace_resolver._workspace(context)
        payload = json.dumps(
            {"context": context.model_dump(mode="json"), "spec": spec.model_dump(mode="json")},
            ensure_ascii=False,
        ).encode()
        with tempfile.TemporaryDirectory(prefix="github-research-agent-sandbox-") as raw_dir:
            config_path = Path(raw_dir) / "config.json"
            config_path.write_text(
                json.dumps(
                    self._sandbox_config(
                        workspace,
                        repository_agent=(
                            context.task_metadata.get("domain") == "repository_research"
                        ),
                    ),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            config_path.chmod(0o600)
            command = self._command(workspace, config_path)
            logger.info(
                "Harness task {} entering {} workspace={} network={}",
                context.task_id,
                self.isolation_mode,
                workspace,
                self.settings.sandbox_network,
            )
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                preexec_fn=self._limit_resources,
            )
            try:
                stdout, stderr = await process.communicate(payload)
            except asyncio.CancelledError:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    process.kill()
                    await process.wait()
                raise

        output = stdout.decode("utf-8", errors="replace")
        error = _redact(stderr.decode("utf-8", errors="replace"))
        if process.returncode != 0:
            detail = error[-4000:] or _redact(output[-4000:])
            raise RuntimeError(
                f"Harness sandbox exited with code {process.returncode}: {detail}"
            )
        line = next(
            (item for item in reversed(output.splitlines()) if item.startswith(_RESULT_PREFIX)),
            None,
        )
        if line is None:
            raise RuntimeError("Harness sandbox returned no structured result")
        envelope = json.loads(line[len(_RESULT_PREFIX) :])
        if envelope.get("kind") == "review":
            return ReviewResult.model_validate(envelope.get("value"))
        if envelope.get("kind") != "text" or not isinstance(envelope.get("value"), str):
            raise RuntimeError("Harness sandbox returned an invalid result envelope")
        return envelope["value"]
