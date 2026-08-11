---
name: repository-cache-management
description: "Explain where GitHub repositories are cached and safely manage natural-language requests to delete downloaded projects. Use when a user asks where analyzed source code is stored, asks to delete/remove a local downloaded repository, or needs to distinguish local cache deletion from deleting a GitHub repository."
---

# Repository Cache Management

Treat repository paths and deletion as controlled repository-management actions.

## Repository answers

End every repository analysis or code answer by reporting the exact project source
path supplied in the repository context. Label it `项目本地保存路径`. Never invent
or shorten the path.

## Local deletion

Interpret requests such as these as local cache management:

- `删除本地下载的 owner/repo 项目`
- `把 OpenHarness 的本地缓存删掉`
- `remove the downloaded owner/repo`
- `/repo delete owner/repo`

Never call filesystem or shell deletion tools for these requests. Let the
Repository Orchestrator resolve the cached repository and return an approval ID.
Display the resolved repository, commit, source path, and full cache deletion
scope to the user.

Only the exact requester may continue with:

```text
/approve-delete <approval_id>
```

Cancel with:

```text
/reject-delete <approval_id>
```

Do not describe a project as deleted until the orchestrator reports completion.
Deletion removes the repository cache root, including its analysis source,
isolated task workspaces, and local patches. It does not delete GitHub state.

## Safety boundary

Distinguish local cache deletion from requests to delete a GitHub repository,
Fork, branch, or remote. Remote repository deletion remains forbidden and must
be refused. If a short repository name matches multiple downloads, require
`owner/repo`; never guess.
