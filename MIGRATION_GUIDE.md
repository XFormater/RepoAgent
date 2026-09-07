# GitHubResearch_Agent 服务器迁移指南

本文针对当前服务器上的实际部署编写，目标是在迁移
`GitHubResearch_Agent` 时完整保留应用文件、运行状态和它独占使用的四个数据库：
PostgreSQL、Redis、Neo4j、Qdrant。

文档基线日期：2026-09-07。

## 1. 当前部署结构

| 组件 | 当前用途 | 持久化位置 |
| --- | --- | --- |
| Gateway | WebSocket、WebUI、Agent 主进程 | 项目目录和 `/root/.nanobot` |
| Harness Worker | 多 Agent 任务执行器 | PostgreSQL、Redis、项目目录 |
| PostgreSQL 17 | 语义记忆元数据、Harness 工作流/任务/检查点 | Docker volume `postgres_data` |
| Redis 7.4.2 | 短期记忆、Harness 队列/租约/幂等状态 | Docker volume `redis_data` |
| Neo4j 5.25.1 | 图记忆 | Docker volume `neo4j_data` |
| Qdrant 1.15.4 | 向量记忆 | Docker volume `qdrant_storage` |

应用由以下 systemd 服务启动：

- `github-research-agent-gateway.service`
- `github-research-agent-worker.service`

当前关键端口：

- `127.0.0.1:18790`：Gateway health/readiness。
- `127.0.0.1:8765`：WebSocket 和 WebUI。
- `5432`：PostgreSQL。
- `6379`：Redis。
- `7474`、`7687`：Neo4j HTTP/Bolt。
- `127.0.0.1:6333`、`127.0.0.1:6334`：Qdrant REST/gRPC。

因为只有本项目使用四个数据库，新服务器应将所有数据库端口限制到
`127.0.0.1`。不要继续把 PostgreSQL、Redis 和 Neo4j 直接暴露到公网。

PostgREST 和 RedisInsight 不是项目运行所必需的组件，可以在核心迁移完成后按需恢复。

## 2. 必须迁移的文件

必须保留：

- `/root/GitHubResearch_Agent`：代码以及项目内的运行状态。
- `/root/GitHubResearch_Agent/.env`：数据库和模型密钥，权限应保持为 `0600`。
- `/root/GitHubResearch_Agent/.nanobot`：VFS、repository agent 状态和下载的仓库。
- `/root/GitHubResearch_Agent/sessions`：会话历史。
- `/root/GitHubResearch_Agent/memory`：文件记忆和 Dream 游标。
- `/root/GitHubResearch_Agent/cron`：定时任务。
- `/root/GitHubResearch_Agent/nanobot/web/dist`：当前 WebUI 构建产物；它未被 Git 跟踪。
- `/root/.nanobot`：实际运行配置、用户级状态和 WebUI 数据。
- 两个 systemd 主 unit 及其 `.d/hardening.conf` drop-in。
- 数据库相关 Docker volumes（Neo4j 分别使用 data、logs、plugins 三个卷）。

通常不需要迁移：

- `/root/GitHubResearch_Agent/.venv`：在新服务器重新创建。
- `/root/GitHubResearch_Agent/webui/node_modules`：需要重新构建 WebUI 时再安装。
- pip、npm、VS Code Server 缓存。

当前工作树并非干净状态，含运行时改动和未跟踪数据。不能只执行 `git clone`；必须复制实际项目目录。

迁移前先运行项目内的只读预检脚本。它不会读取或打印 `.env` 的值，但会检查
服务、数据库容器、Docker volumes、关键路径、权限、磁盘空间和仍在执行的 `/rere`：

```bash
cd /root/GitHubResearch_Agent
./scripts/migration_preflight.sh | tee /root/github-research-agent-preflight.txt
```

预检发现上述状态的研究记录，且 PostgreSQL 中对应的 Harness Task 仍为
`pending`、`ready`、`leased` 或 `running` 时，会以退出码 `2` 停止。这是预期保护：
先等待任务完成，或由操作员明确取消后再做最终备份。已进入终态但状态文件尚未更新的
旧记录只会告警，不会误阻迁移。不要为了绕过检查而用
`--allow-active-research` 进行正式切流；该参数只适合检查已有的演练快照。

## 3. 已有安全备份

服务器上已有一份经过校验的数据库安全副本：

```text
/root/server-migration-backup-20260907T011931Z.tar.gz
SHA-256: 9bc4e4e305af8b3995bb6e1baec6fbd10ead4178f53ca3345fd43b512f7bc330
```

它包含四个数据库的原生备份、完整 Docker volume、容器配置和恢复说明。

该备份制作时项目仍在运行，各数据库的快照时间相差几分钟。它适合作为安全副本，
但正式切流前仍应在停止 Gateway 和 Worker 后再制作一次最终静默备份，确保应用文件和四个数据库处于同一业务停写窗口。

## 4. 迁移前准备

新服务器建议满足：

- Linux x86_64。
- Python 3.12。
- Docker 26 或兼容版本。
- `bwrap`（当前 Harness Worker 的源码分析隔离后端）。
- 至少 10 GB 可用空间；若同时保存 Docker 镜像 tar，建议预留更多空间。
- 能够访问项目所需的模型/API 服务。

当前 Python 包使用 editable 方式安装，必要的可选依赖是 `api` 和 `memory`。

为避免 `latest` 标签漂移，建议在旧服务器保存正在使用的镜像层：

```bash
umask 077
docker save \
  -o /root/github-research-agent-db-images-20260907.tar \
  postgres:17 redis:7.4.2 neo4j:5.25.1 qdrant/qdrant:v1.15.4 alpine:3.20
sha256sum /root/github-research-agent-db-images-20260907.tar \
  > /root/github-research-agent-db-images-20260907.tar.sha256
```

如果不传输镜像，至少固定为当前实际版本：PostgreSQL 17、Redis 7.4.2、Neo4j 5.25.1、Qdrant 1.15.4。

## 5. 正式停写和最终备份

### 5.1 停止应用写入

先确认没有运行中的 POOL 任务。对 `/rere` 使用 WebUI 或
`/rere status <research-id>` 检查；同时记录 Harness 的最终任务分布，便于在新服务器
验收时比对：

```bash
docker exec postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc \
  "SELECT state || chr(124) || count(*) FROM harness_tasks GROUP BY state ORDER BY state"'
```

如果存在 `leased` 或 `running`，不要立刻停止 Worker。等待它们完成、重试进入终态，
或通过受控操作取消；直接 `systemctl stop` 会中断正在运行的进程隔离分析。

在维护窗口重新运行预检，确认它通过后，再停止 Worker 和 Gateway：

```bash
systemctl stop github-research-agent-worker.service
systemctl stop github-research-agent-gateway.service
systemctl is-active github-research-agent-worker.service
systemctl is-active github-research-agent-gateway.service
```

两条 `is-active` 应返回 `inactive`。从这一步开始，不要在旧服务器重新启动应用，也不要让新旧实例同时写入数据库。

### 5.2 备份应用和 systemd 配置

```bash
umask 077
MIGRATION_TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

tar --xattrs --acls --numeric-owner \
  --exclude='GitHubResearch_Agent/.venv' \
  --exclude='GitHubResearch_Agent/webui/node_modules' \
  -C /root \
  -czf "/root/github-research-agent-app-${MIGRATION_TIMESTAMP}.tar.gz" \
  GitHubResearch_Agent .nanobot

tar --xattrs --acls --numeric-owner \
  -C /etc/systemd/system \
  -czf "/root/github-research-agent-systemd-${MIGRATION_TIMESTAMP}.tar.gz" \
  github-research-agent-gateway.service \
  github-research-agent-gateway.service.d \
  github-research-agent-worker.service \
  github-research-agent-worker.service.d

sha256sum \
  "/root/github-research-agent-app-${MIGRATION_TIMESTAMP}.tar.gz" \
  "/root/github-research-agent-systemd-${MIGRATION_TIMESTAMP}.tar.gz" \
  > "/root/github-research-agent-files-${MIGRATION_TIMESTAMP}.sha256"
```

后续最终数据库备份应在同一个 shell 中执行；如果重新打开了 shell，先按相同格式重新设置
`MIGRATION_TIMESTAMP`。

不要打印 `.env` 或 `/root/.nanobot/config.json` 的内容。两个文件均包含密钥。

### 5.3 制作四库最终离线 volume 备份

应用停写后，正常停止数据库，使 PostgreSQL、Redis、Neo4j 和 Qdrant 完成落盘：

```bash
docker stop --time 60 postgres redis neo4j qdrant
docker ps --format '{{.Names}}|{{.Status}}'
```

创建最终备份目录并打包：

```bash
FINAL_DB_DIR="/root/github-research-agent-db-final-${MIGRATION_TIMESTAMP}"
install -d -m 700 "${FINAL_DB_DIR}/volumes" "${FINAL_DB_DIR}/config"

tar --numeric-owner \
  -C /var/lib/docker/volumes/postgres_data/_data \
  -czf "${FINAL_DB_DIR}/volumes/postgres_data.tar.gz" .

tar --numeric-owner \
  -C /var/lib/docker/volumes/redis_data/_data \
  -czf "${FINAL_DB_DIR}/volumes/redis_data.tar.gz" .

tar --numeric-owner \
  -C /var/lib/docker/volumes/neo4j_data/_data \
  -czf "${FINAL_DB_DIR}/volumes/neo4j_data.tar.gz" .

tar --numeric-owner \
  -C /var/lib/docker/volumes/neo4j_plugins/_data \
  -czf "${FINAL_DB_DIR}/volumes/neo4j_plugins.tar.gz" .

tar --numeric-owner \
  -C /var/lib/docker/volumes/neo4j_logs/_data \
  -czf "${FINAL_DB_DIR}/volumes/neo4j_logs.tar.gz" .

tar --numeric-owner \
  -C /var/lib/docker/volumes/qdrant_storage/_data \
  -czf "${FINAL_DB_DIR}/volumes/qdrant_storage.tar.gz" .

docker inspect postgres redis neo4j qdrant \
  > "${FINAL_DB_DIR}/config/docker-containers.json"
docker image inspect postgres:17 redis:latest neo4j:latest qdrant/qdrant:v1.15.4 \
  > "${FINAL_DB_DIR}/config/docker-images.json"
chmod -R go-rwx "${FINAL_DB_DIR}"

cd "${FINAL_DB_DIR}"
find . -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
```

最终切流时让旧数据库保持停止。若这里只是迁移演练，可以按
`postgres redis neo4j qdrant` 的顺序重启数据库，再启动 Gateway 和 Worker。

## 6. 将文件传到新服务器

至少传输：

- 应用归档及其 SHA-256 文件。
- systemd 归档。
- 最终数据库目录，或将其先打成一个总压缩包。
- 推荐同时传输 Docker 镜像 tar。
- 已有的 `server-migration-backup-20260907T011931Z.tar.gz` 作为第二份保险。

示例：

```bash
rsync -avP /root/github-research-agent-* root@NEW_SERVER_IP:/root/
rsync -avP /root/server-migration-backup-20260907T011931Z.tar.gz* \
  root@NEW_SERVER_IP:/root/
```

在新服务器立即执行 `sha256sum -c`。任何校验失败都必须重新传输，不能继续恢复。

## 7. 在新服务器恢复

### 7.1 恢复镜像

如已传输镜像 tar：

```bash
docker load -i /root/github-research-agent-db-images-20260907.tar
```

否则拉取固定版本镜像，不要直接采用迁移时未知的新 `latest`：

```bash
docker pull postgres:17
docker pull redis:7.4.2
docker pull neo4j:5.25.1
docker pull qdrant/qdrant:v1.15.4
docker pull alpine:3.20
```

### 7.2 恢复应用文件

```bash
tar -xzf /root/github-research-agent-app-TIMESTAMP.tar.gz -C /root
tar -xzf /root/github-research-agent-systemd-TIMESTAMP.tar.gz \
  -C /etc/systemd/system

chmod 600 /root/GitHubResearch_Agent/.env
chmod 600 /root/.nanobot/config.json
```

本指南按原路径 `/root/GitHubResearch_Agent` 编写。保持原路径可以避免修改 systemd、workspace 和 repository-agent 路径；如果更换路径，必须同步修改：

- systemd 的 `WorkingDirectory`、`ExecStart` 和 `ReadWritePaths`。
- `/root/.nanobot/config.json` 中的 workspace、cacheDir 和 stateDir。

### 7.3 重建 Python 环境

```bash
cd /root/GitHubResearch_Agent
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[api,memory]'
```

确认 WebUI 构建产物已经随应用归档恢复：

```bash
test -f /root/GitHubResearch_Agent/nanobot/web/dist/index.html
```

如果该文件不存在，需要安装 Bun 后重新构建：

```bash
cd /root/GitHubResearch_Agent/webui
bun install --frozen-lockfile
bun run build
```

### 7.4 恢复 Docker volumes

先创建空 volume：

```bash
docker volume create postgres_data
docker volume create redis_data
docker volume create neo4j_data
docker volume create neo4j_plugins
docker volume create neo4j_logs
docker volume create qdrant_storage
```

以下示例使用临时 Alpine 容器恢复，避免依赖宿主机 Docker volume 的内部路径。
将 `FINAL_DB_DIR` 改为新服务器上的实际目录：

```bash
FINAL_DB_DIR=/root/github-research-agent-db-final-TIMESTAMP

docker run --rm \
  -v postgres_data:/restore \
  -v "${FINAL_DB_DIR}/volumes:/backup:ro" \
  alpine:3.20 sh -c 'tar -xzf /backup/postgres_data.tar.gz -C /restore'

docker run --rm \
  -v redis_data:/restore \
  -v "${FINAL_DB_DIR}/volumes:/backup:ro" \
  alpine:3.20 sh -c 'tar -xzf /backup/redis_data.tar.gz -C /restore'

docker run --rm \
  -v neo4j_data:/restore \
  -v "${FINAL_DB_DIR}/volumes:/backup:ro" \
  alpine:3.20 sh -c 'tar -xzf /backup/neo4j_data.tar.gz -C /restore'

docker run --rm \
  -v neo4j_plugins:/restore \
  -v "${FINAL_DB_DIR}/volumes:/backup:ro" \
  alpine:3.20 sh -c 'tar -xzf /backup/neo4j_plugins.tar.gz -C /restore'

docker run --rm \
  -v neo4j_logs:/restore \
  -v "${FINAL_DB_DIR}/volumes:/backup:ro" \
  alpine:3.20 sh -c 'tar -xzf /backup/neo4j_logs.tar.gz -C /restore'

docker run --rm \
  -v qdrant_storage:/restore \
  -v "${FINAL_DB_DIR}/volumes:/backup:ro" \
  alpine:3.20 sh -c 'tar -xzf /backup/qdrant_storage.tar.gz -C /restore'
```

目标 volume 必须是新建的空卷。不要将备份覆盖到已有数据库文件上。

### 7.5 检查应用连接配置

数据库连接同时存在于项目 `.env` 和 `/root/.nanobot/config.json` 中。若四库与项目部署在同一台新服务器，连接主机统一使用 `127.0.0.1`。

需要核对的 `.env` 键名：

```text
POSTGRES_DB
POSTGRES_USER
POSTGRES_PASSWORD
REDIS_PASSWORD
NEO4J_AUTH
NEO4J_PASSWORD
QDRANT_API_KEY
NANOBOT_MEMORY__STORAGE__METADATA_DB__DSN
NANOBOT_MEMORY__STORAGE__REDIS__URL
NANOBOT_MEMORY__STORAGE__GRAPH_DB__URI
NANOBOT_MEMORY__STORAGE__GRAPH_DB__USERNAME
NANOBOT_MEMORY__STORAGE__GRAPH_DB__PASSWORD
NANOBOT_MEMORY__STORAGE__VECTOR_DB__HOST
NANOBOT_MEMORY__STORAGE__VECTOR_DB__PORT
NANOBOT_MEMORY__STORAGE__VECTOR_DB__REST_PORT
NANOBOT_MEMORY__STORAGE__VECTOR_DB__API_KEY
NO_PROXY
```

不要把这些变量的值提交到 Git、粘贴到迁移日志或发送到公开聊天中。

### 7.6 启动四个数据库

在受控的 root shell 中加载 `.env`，不要开启 shell trace：

```bash
set +x
set -a
. /root/GitHubResearch_Agent/.env
set +a
```

启动 PostgreSQL：

```bash
docker run -d \
  --name postgres \
  --restart unless-stopped \
  --env-file /root/GitHubResearch_Agent/.env \
  -p 127.0.0.1:5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:17
```

启动 Redis：

```bash
docker run -d \
  --name redis \
  --restart unless-stopped \
  -p 127.0.0.1:6379:6379 \
  -v redis_data:/data \
  redis:7.4.2 \
  redis-server --appendonly yes --requirepass "${REDIS_PASSWORD}"
```

启动 Neo4j：

```bash
docker run -d \
  --name neo4j \
  --restart unless-stopped \
  -e NEO4J_AUTH="${NEO4J_AUTH}" \
  -p 127.0.0.1:7474:7474 \
  -p 127.0.0.1:7687:7687 \
  -v neo4j_data:/data \
  -v neo4j_plugins:/plugins \
  -v neo4j_logs:/logs \
  neo4j:5.25.1
```

启动 Qdrant：

```bash
docker run -d \
  --name qdrant \
  --restart unless-stopped \
  -e QDRANT__SERVICE__API_KEY="${QDRANT_API_KEY}" \
  -e RUN_MODE=production \
  -e TZ=Etc/UTC \
  -p 127.0.0.1:6333:6333 \
  -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.15.4
```

完成后从当前 shell 清除敏感变量，或直接退出该 root shell：

```bash
unset POSTGRES_PASSWORD REDIS_PASSWORD NEO4J_AUTH NEO4J_PASSWORD QDRANT_API_KEY
```

## 8. 数据库验收

### PostgreSQL

```bash
docker exec postgres pg_isready
docker exec postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc "SELECT 1"'
```

可用下列查询比较表记录：

```bash
docker exec postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc \
  "SELECT schemaname || chr(46) || relname || chr(124) || n_live_tup
   FROM pg_stat_user_tables ORDER BY schemaname, relname"'
```

### Redis

```bash
docker exec redis redis-cli --no-auth-warning -a "${REDIS_PASSWORD}" PING
docker exec redis redis-cli --no-auth-warning -a "${REDIS_PASSWORD}" DBSIZE
```

Redis 中有带 TTL 的键，迁移后键数略少可能是正常过期，不能只用 `DBSIZE` 判断失败。

### Neo4j

```bash
docker exec neo4j bash -lc '
  neo_user="${NEO4J_AUTH%%/*}"
  neo_pass="${NEO4J_AUTH#*/}"
  /var/lib/neo4j/bin/cypher-shell -u "$neo_user" -p "$neo_pass" \
    "MATCH (n) RETURN count(n) AS nodes;
     MATCH ()-[r]->() RETURN count(r) AS relationships;"
'
```

### Qdrant

```bash
curl -fsS \
  -H "api-key: ${QDRANT_API_KEY}" \
  http://127.0.0.1:6333/collections | jq .
```

2026-09-07 检查时的参考基线如下。正式切换应在停写后重新记录一次，因为运行中的数据会继续变化：

| 存储 | 参考值 |
| --- | --- |
| PostgreSQL `agent_memory_records` | 26,357 |
| PostgreSQL Harness | 23 workflows、87 tasks、35 checkpoints |
| Redis | 98 keys，其中部分有 TTL |
| Neo4j | 906 nodes、82,335 relationships |
| Qdrant `github_research_agent_memory` | 0 points |
| Qdrant `locomo_hybrid_v2_1024` | 8,695 points |
| Qdrant `locomo_hybrid_v2_2_1024` | 8,967 points |
| Qdrant `locomo_hybrid_v2_1_1024` | 8,695 points |

## 9. 启动和验收应用

重新加载 systemd 并启动 Gateway、Worker：

```bash
systemctl daemon-reload
systemctl enable github-research-agent-gateway.service
systemctl enable github-research-agent-worker.service
systemctl start github-research-agent-gateway.service
systemctl start github-research-agent-worker.service
```

检查状态和日志：

```bash
systemctl --no-pager --full status github-research-agent-gateway.service
systemctl --no-pager --full status github-research-agent-worker.service
journalctl -u github-research-agent-gateway.service -n 100 --no-pager
journalctl -u github-research-agent-worker.service -n 100 --no-pager
```

Gateway 健康检查：

```bash
curl -fsS http://127.0.0.1:18790/health | jq .
curl -fsS http://127.0.0.1:18790/ready | jq .
curl -fsS http://127.0.0.1:8765/ >/dev/null
```

期望 `/ready` 至少包含：

```json
{
  "status": "ready",
  "harness": {
    "store": true,
    "postgres": true,
    "redis": true
  }
}
```

`/ready` 只直接覆盖 Harness 的 PostgreSQL 和 Redis，因此 Neo4j、Qdrant 仍必须执行上一节的独立检查。

迁移了 POOL/Harness 工作流时，再核验任务状态和审计数据已恢复：

```bash
docker exec postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc \
  "SELECT state || chr(124) || count(*) FROM harness_tasks GROUP BY state ORDER BY state"'

docker exec postgres sh -lc \
  'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc \
  "SELECT count(*) FROM harness_audit_events"'
```

任务总数和审计事件总数应不小于停写前记录的值；Redis 中过期租约、邮箱和幂等键
可因 TTL 自然减少。不要在旧服务器仍可能写入时同时启动新服务器的 Worker。

最后通过 WebUI 完成一次真实但低风险的读写测试：

1. 打开历史会话，确认 session 和附件可见。
2. 发起一次新对话，确认模型调用成功。
3. 执行一次会触发记忆写入的对话。
4. 再次核对 PostgreSQL、Redis、Neo4j 和 Qdrant 无认证或连接错误。
5. 确认 Harness Worker 已注册且没有持续重启。

## 10. 切流与回滚

完成全部验收后再更新反向代理、DNS 或客户端连接地址。数据库端口不需要开放公网。

建议至少保留旧服务器和全部备份 3～7 天，但旧服务器上的 Gateway、Worker 和数据库应继续保持停止，避免双写。

如果新服务器尚未接收真实写入，回滚步骤是：

1. 停止新服务器的 Gateway 和 Worker。
2. 停止新服务器四个数据库。
3. 在旧服务器按数据库、Gateway、Worker 的顺序重新启动。
4. 将流量切回旧服务器。

如果新服务器已经接收了真实写入，不要直接启动旧服务器；必须先备份新服务器数据并制定反向同步方案，否则会丢失迁移后的新数据。

## 11. 完成条件

只有同时满足以下条件，迁移才算完成：

- 所有传输文件通过 SHA-256 校验。
- `scripts/migration_preflight.sh` 在最终停写前通过，且没有未处置的 `/rere` 任务。
- 项目代码、`.env`、项目内 `.nanobot`、`sessions`、`memory`、`cron` 均已恢复。
- `/root/.nanobot/config.json` 已恢复且权限为 `0600`。
- 四个数据库容器使用持久化 volume 并通过独立检查。
- Gateway `/health` 和 `/ready` 正常。
- Worker 保持 active，没有重启循环。
- WebUI 历史会话及一次真实读写测试正常。
- 已保存可用的最终静默备份和回滚副本。
- 旧服务器尚未释放，或备份已至少保存到两个独立位置。
