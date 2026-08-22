# GitHubResearchAgent 对话交接说明

项目路径：`/root/GitHubResearch_Agent`

## 当前目标

完成生产级多智能体 Harness 的逻辑 VFS，实现用户、租户、任务隔离，并接入 Agent Runtime、Memory、Repository Agent、API、Shell 和 Harness。

## 已完成工作

### 1. 逻辑 VFS

新增：

- `nanobot/security/logical_vfs.py`
- `tests/security/test_logical_vfs.py`

支持：

- 根据 `(tenant_id, user_id, task_id)` 派生隔离命名空间
- 项目目录：`projects/`
- 任务目录：`tasks/`
- 用户私有 Skill：`skills/`
- 共享 Skill 只读
- 路径穿越保护
- 用户级、任务级字节和文件数量配额
- 任务、项目 TTL 清理
- VFS 指标计数与快照
- 使用 `ContextVar` 绑定当前 VFS 和身份

默认配置：

```json
{
  "vfs": {
    "enabled": true,
    "root": ".nanobot/vfs",
    "maxUserBytes": 5368709120,
    "maxUserFiles": 100000,
    "maxTaskBytes": 1073741824,
    "maxTaskFiles": 50000,
    "taskTtlSeconds": 604800,
    "projectTtlSeconds": 2592000
  }
}
```

### 2. Agent Runtime 接入

已修改：

- `nanobot/agent/loop.py`
- `nanobot/agent/tools/context.py`
- `nanobot/agent/context.py`
- `nanobot/agent/context_orchestrator.py`
- `nanobot/agent/skills.py`
- `nanobot/agent/subagent.py`
- `nanobot/agent/tools/spawn.py`

实现：

- 主 Agent 运行期间绑定 VFS 身份
- 子 Agent 继承 tenant/user 身份
- Context Builder 按用户读取 MEMORY.md 和私有 Skill
- 统一 `RequestContext`
- Task State、Tool Execution、Runtime Context 可注入 Agent Context
- Agent 启动和运行期间周期清理过期 VFS namespace
- Context Orchestrator 保留指标统计

### 3. Repository Agent 接入

已修改：

- `nanobot/repo_agent/repository.py`
- `nanobot/repo_agent/orchestrator.py`
- `nanobot/repo_agent/loop_adapter.py`
- `nanobot/repo_agent/tools.py`

实现：

- GitHub 仓库缓存按 tenant/user 隔离
- 相同仓库被不同用户下载时使用不同 cache root
- 仓库工具只允许访问配置的安全 root
- Repository Agent 继承当前用户的 VFS identity
- 仓库工作区受配额控制

### 4. 文件和 Shell 工具

已修改：

- `nanobot/agent/tools/filesystem.py`
- `nanobot/agent/tools/shell.py`

实现：

- 写文件、编辑文件前执行 VFS quota 检查
- 禁止写入 shared/workspace Skill 目录
- Shell 执行后检查 VFS quota
- 原有 WorkspaceScope 和 Bubblewrap 机制继续保留

### 5. API 用户隔离

已修改：

- `nanobot/api/server.py`

支持：

```http
X-Nanobot-Tenant: tenant-a
X-Nanobot-User: user-123
```

这些身份用于：

- Session key
- VFS namespace
- 上传文件目录
- Memory namespace
- Repository cache namespace
- Agent Runtime metadata

注意：

- Header 应由可信网关或身份认证层注入
- Header 本身不是认证机制
- 未提供 Header 时，为兼容单用户模式，会使用 `session_id`

上传文件按用户隔离；multipart 上传会先进入临时目录，再移动到目标用户目录。

### 6. 可观测性

`GET /ready` 增加 VFS 状态和指标：

```json
{
  "vfs": {
    "enabled": true,
    "metrics": {
      "quota_exceeded": 0,
      "task_cleanup": 0,
      "cleanup_tasks": 0,
      "cleanup_projects": 0
    }
  }
}
```

Context Orchestrator 也保留阶段耗时、Token、Context block、压缩和错误指标。

### 7. 文档

已更新：

- `docs/configuration.md`
- `docs/multi-agent-harness.md`
- `docs/deployment.md`
- `docs/README.md`

文档说明了 Logical VFS、tenant/user/task 隔离、Skill 权限、API Header、Bubblewrap 和 Harness readiness 指标。

## 测试结果

专项测试：

```text
186 passed
```

排除当前环境相关的 xAI OAuth、WebSocket、WebUI 测试后：

```text
5540 passed, 18 skipped
```

全量测试此前结果：

```text
5839 passed, 18 skipped, 48 failed
```

剩余失败主要集中在：

- xAI OAuth 回调和代理环境
- WebSocket 集成测试
- WebUI Gateway smoke 测试

这些属于当前运行环境或网络代理问题，不是 VFS 相关失败。API model mismatch、API JSON 兼容性、idle compact 时间 mock 等由本轮修改引入的问题已修复。

## 重要注意事项

- 当前是“逻辑 VFS”，不是 FUSE 或 OverlayFS。
- 逻辑 VFS 负责路径、身份、权限、配额和生命周期。
- Bubblewrap 负责进程级和挂载级隔离。
- 生产环境仍应通过可信网关注入用户身份。
- 当前工作树包含此前已有的 Harness、Docker、Memory、Sandbox 等改动，不要执行 `git reset --hard` 或覆盖无关文件。
- `sessions/Y2xpOmRpcmVjdA.jsonl` 等文件存在已有修改，应保留。

## 建议下一步

1. 检查当前 Git diff，确认没有换行格式或无关文件污染。
2. 为 API tenant/user Header 增加端到端隔离测试。
3. 为 MemoryStore 的长期写入路径进一步实现用户级物理隔离。
4. 增加 VFS quota 超限时 Shell 命令执行前的预估限制。
5. 增加独立 Worker、Bubblewrap、Logical VFS 的部署测试。
6. 处理 WebSocket、xAI OAuth 和 WebUI smoke 测试环境问题。
7. 如需更强隔离，再实现 OverlayFS、FUSE 或 Bubblewrap 动态 mount namespace。
