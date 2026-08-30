# Repository Analysis, Maintenance, and Open-Source Research

The optional `repositoryAgent` subsystem adds repository analysis, controlled
code maintenance, and bounded open-source project research to nanobot's
general-purpose agent runtime. It does not replace nanobot's other capabilities
such as conversation, web research, files, media, memory, connected apps,
automations, sustained goals, or subagents.

GitHub search is used only for candidate discovery when a user asks to find
open-source projects. Supplied-repository analysis and maintenance do not need a
search stage, and general research should use the information sources best
suited to the user's question.

## Behavior

Send a public GitHub repository URL and a question:

```text
https://github.com/owner/repository

Explain the main request lifecycle and cite the relevant files.
```

The runtime validates the URL, creates a shallow local clone, binds the answer
to the cloned commit, and routes the question to a fixed expert. It does not
Fork the project and does not execute repository code during intake.
Every repository answer ends with the exact local source-cache path used for
that answer.

Subsequent messages in the same session use the active repository. Run
`/repo close` to leave repository mode.

## Academic project research

The repository agent can also start without a URL when the user explicitly
asks it to find, research, recommend, or compare open-source projects:

```text
寻找图神经网络用于药物发现的 GitHub 开源项目，分析代码并比较可复现性。

“下载并总结 5 个开源项目”、“整理/盘点 GitHub 仓库”等同类集合研究请求也会
直接路由到 Repository Research Orchestrator，而不会交给普通 Agent 自行克隆。
```

To bypass natural-language routing and force this workflow, use:

```text
/rere 下载并分析 3 个 2026 年自回归图像生成开源项目
```

`/rere` without a research request returns a usage hint and does not
fall back to the general-purpose agent.

An explicit project count in Chinese or English is honored up to the configured
`repositoryAgent.research.maxRepositories` safety ceiling. For example, `3 个`,
`三个`, `3 projects`, and `three repositories` select three projects even when
the configured ceiling is five. The WebUI publishes Harness initialization and
workflow-stage progress immediately; reasoning from repository control calls is
streamed while query planning and report synthesis are running.
Query planning returns English-language search phrases, even when the user's
request is Chinese, so discovery sees terms such as `RAG retrieval`, `vector
search`, and `retrieval augmented generation` instead of the raw Chinese
instruction.

The research workflow expands the topic into up to four search queries, runs
nanobot's native `web_search` to recall GitHub repository pages, resolves those
URLs through GitHub's read-only repository metadata API, also runs GitHub's
Repository Search API as a structured supplement, deduplicates and ranks
candidates by topic relevance, popularity, freshness, and license metadata,
then:

1. selects a bounded number of repositories;
2. shallow-clones them under the normal repository cache, which is the
   user-scoped VFS `projects` directory when VFS is enabled. Leave
   `repositoryAgent.research.repositoryCacheDir` unset/null for this default;
   set it only when you intentionally want a non-VFS cache override;
3. analyzes each clone with repository-only read tools;
4. produces a comparison table, per-project evidence, recommendations, gaps,
   task ID, and successful local paths;
5. persists the full research record under the repository-agent state directory.

Downloads and analyses use separate configurable concurrency limits. One failed
clone or analysis produces a partial report as long as another project succeeds.
The WebUI receives per-repository download start/success/failure progress
messages and streams the final report text as it is generated.
Set `GITHUB_TOKEN` (or change `research.githubTokenEnv`) to increase GitHub API
rate limits. The token is sent only to the fixed GitHub Search API endpoint and
is never included in clone URLs or model prompts.

## Agents

| Agent | Responsibility |
|---|---|
| Repo Intake | Validate the URL, clone, measure, cache, and bind the repository |
| Intent | Classify questions, issue analysis, review, and code-change requests |
| Dispatcher | Route only to experts registered in configuration |
| Repo Knowledge | Explain architecture and source paths with direct evidence |
| Issue Analyst | Read a real GitHub Issue and trace likely affected code |
| Code | Modify an isolated task clone, never the read-only analysis clone |
| Test | Inspect tests and execute them only when the configured execution policy allows |
| Review | Independently review the real diff and return PASS or BLOCKED |
| GitHub | Prepare metadata; deterministic code performs approved writes |

Simple read-only questions run as bounded ReAct turns. Code changes use a
persistent dependency plan. The default plan is issue analysis (when relevant),
implementation, validation, and independent review.

## Save and publish commands

After a successful code task:

```text
/repo save
/repo patch
/repo fork
/repo pr
```

- `save` keeps the isolated task workspace locally.
- `patch` writes a patch under the task workspace.
- `fork` requests approval to create/reuse a Fork and Push a new task branch.
- `pr` requests approval for the same actions plus a Draft Pull Request.

`fork` and `pr` display the exact upstream, Fork owner, base commit, base/head
branches, title, and canonical Diff hash. Continue with:

```text
/approve <approval_id>
/reject <approval_id>
```

If the Diff changes after approval, the approval is invalidated. The first
version never merges, releases, force-pushes, or writes to an upstream default
branch.

## Local repository-cache deletion

Downloaded projects can be targeted with natural language or an explicit
command:

```text
删除本地下载的 owner/repository 项目
/repo delete owner/repository
```

The runtime resolves only existing cache entries and never clones a missing
repository during deletion. A short repository name must be unique; otherwise
the user must provide `owner/repository`. The system then shows the exact
repository, commit, source path, and cache root and waits for:

```text
/approve-delete <approval_id>
/reject-delete <approval_id>
```

The approval is bound to the requester, repository ID, URL, commit, source
path, cache root, and expiry. If any target field changes, deletion is refused.
Confirmed deletion removes that repository's source cache, isolated task
workspaces, and local patches, then invalidates active sessions and pending
publish approvals. It never deletes a GitHub repository or Fork.

## GitHub authentication

The first version is designed for a single-user portfolio deployment and uses
GitHub CLI. Install `git` and `gh`, then authenticate:

```bash
gh auth login
gh auth status
```

The runtime uses `gh auth git-credential` as an ephemeral Git credential helper;
tokens are not put into remote URLs or persisted by the repository agent.

For unauthenticated public Issue reads, the agent calls GitHub's public REST API.
Set `NANOBOT_GITHUB_TOKEN` only when a higher read rate limit is required.

## Test execution policy

Repository code and build scripts are untrusted. The default is:

```json
{
  "repositoryAgent": {
    "testExecution": "disabled"
  }
}
```

Supported values:

- `disabled`: the Test Agent inspects configuration but cannot execute commands.
- `sandboxed`: exposes `exec` only when nanobot has a configured system sandbox.
- `host`: executes through the host `exec` tool and should be used only for
  repositories you trust.

For public arbitrary repositories, use Linux with the existing `bwrap` sandbox.
Do not use `host` for untrusted repositories.

## Security boundaries

- Only configured GitHub hosts and HTTPS repository URLs are accepted.
- Credentials and non-GitHub URLs are rejected. Nested GitHub pages are
  normalized to the repository root so analysis still covers the full project.
- Clone is shallow, does not initialize submodules, and does not run builds.
- Repository size and file counts are bounded.
- Repository read tools resolve paths and reject symlink/path escapes.
- Analysis clones and code-task clones are separate.
- Repository files, Issues, comments, and commit messages are prompt-injection
  inputs and never become control instructions.
- Force-push, remote GitHub repository deletion, secret extraction, and
  branch-protection bypass requests are L4 and denied before model invocation.
- Local downloaded-cache deletion is L3, targets one revalidated cache root,
  and requires an exact expiring human approval.
- Fork, Push, and PR creation are L3 and require exact Diff-bound approval.

## Configuration

Merge the required fields from
[`repository-agent-config.example.json`](./repository-agent-config.example.json)
into `~/.nanobot/config.json`. `repositoryAgent.enabled` defaults to `false`, so
upgrading does not change normal nanobot behavior.

For a dedicated Fork organization, set `repositoryAgent.github.forkOwner`.
Otherwise the authenticated GitHub CLI user is used.

## Current first-version limits

- GitHub.com public repository intake and repository search only.
- Search ranking uses repository metadata before source inspection; the final
  report distinguishes that metadata from code evidence.
- Existing analysis clones are reused until the cache is removed through the
  approved local-deletion flow; automatic upstream refresh is not yet enabled.
- PRs are always Draft.
- CI monitoring and review-comment remediation are planned follow-up workflows.
- The existing chat window is the public surface; a dedicated repository card,
  Plan timeline, and approval panel are not yet included.
