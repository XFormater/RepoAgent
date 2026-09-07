#!/usr/bin/env bash
# Read-only migration preflight for the self-hosted GitHubResearchAgent deployment.
# It intentionally never loads .env or prints configuration values.

set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
USER_CONFIG="/root/.nanobot/config.json"
ALLOW_ACTIVE_RESEARCH=false

if [[ "${1:-}" == "--allow-active-research" ]]; then
  ALLOW_ACTIVE_RESEARCH=true
elif [[ $# -gt 0 ]]; then
  echo "Usage: $0 [--allow-active-research]" >&2
  exit 64
fi

info() { printf 'INFO  %s\n' "$*"; }
ok() { printf 'OK    %s\n' "$*"; }
warn() { printf 'WARN  %s\n' "$*" >&2; }
fail() { printf 'FAIL  %s\n' "$*" >&2; }

required_commands=(docker systemctl tar sha256sum rsync python3)
missing=0
for command in "${required_commands[@]}"; do
  if command -v "$command" >/dev/null 2>&1; then
    ok "command available: $command"
  else
    fail "required command is missing: $command"
    missing=1
  fi
done
if [[ $missing -ne 0 ]]; then
  exit 1
fi

for path in "$PROJECT_ROOT" "$PROJECT_ROOT/.env" "$PROJECT_ROOT/.nanobot" \
  "$PROJECT_ROOT/sessions" "$PROJECT_ROOT/memory" "$PROJECT_ROOT/cron" "$USER_CONFIG"; do
  if [[ -e "$path" ]]; then
    ok "required migration path exists: $path"
  else
    fail "required migration path is missing: $path"
    missing=1
  fi
done

for secret_file in "$PROJECT_ROOT/.env" "$USER_CONFIG"; do
  if [[ -f "$secret_file" ]]; then
    mode="$(stat -c '%a' "$secret_file")"
    if [[ "$mode" == "600" ]]; then
      ok "secret file mode is 0600: $secret_file"
    else
      warn "secret file should be 0600 before transfer: $secret_file (current $mode)"
    fi
  fi
done

for unit in github-research-agent-gateway.service github-research-agent-worker.service; do
  state="$(systemctl is-active "$unit" 2>/dev/null || true)"
  info "service $unit: ${state:-unknown}"
done

for container in postgres redis neo4j qdrant; do
  if docker container inspect "$container" >/dev/null 2>&1; then
    image="$(docker inspect --format '{{.Config.Image}}' "$container")"
    state="$(docker inspect --format '{{.State.Status}}' "$container")"
    ok "container $container: image=$image state=$state"
  else
    fail "required database container is missing: $container"
    missing=1
  fi
done

for volume in postgres_data redis_data neo4j_data neo4j_plugins neo4j_logs qdrant_storage; do
  if docker volume inspect "$volume" >/dev/null 2>&1; then
    ok "Docker volume exists: $volume"
  else
    fail "required Docker volume is missing: $volume"
    missing=1
  fi
done

available_kib="$(df -Pk /root | awk 'NR == 2 {print $4}')"
available_gib=$((available_kib / 1024 / 1024))
info "available disk under /root: ${available_gib} GiB"
if (( available_gib < 10 )); then
  warn "less than 10 GiB free; make space before creating migration archives"
fi

research_records="$({
  python3 - "$PROJECT_ROOT/.nanobot/repository-agent/research" <<'PY'
import json
import sys
from pathlib import Path

active = {"planning", "searching", "downloading", "analyzing", "synthesizing"}
root = Path(sys.argv[1])
if root.is_dir():
    for path in sorted(root.glob("research_*.json")):
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("status") in active:
            print(
                f"{row.get('research_id', path.stem)}|{row['status']}|"
                f"{row.get('workflow_id', '')}"
            )
PY
} || true)"

live_research=""
if [[ -n "$research_records" ]]; then
  while IFS='|' read -r research_id research_status workflow_id; do
    # The repository state file is a user-facing projection. Its last write can
    # predate a Worker cancellation, so use the durable Harness Task as the
    # migration safety authority when a valid workflow id is available.
    if [[ "$workflow_id" =~ ^workflow_[a-f0-9]{32}$ ]]; then
      task_states="$(docker exec postgres sh -lc \
        'psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Atc \
        "SELECT state FROM harness_tasks WHERE workflow_id = '\''"$workflow_id"'\''"' \
        2>/dev/null || true)"
      if grep -Eq '^(pending|ready|leased|running)$' <<<"$task_states"; then
        live_research+="${research_id}|${research_status}|${workflow_id}|${task_states//$'\n'/,}"$'\n'
      else
        warn "stale /rere projection is not blocking migration: ${research_id} (${research_status}; durable task=${task_states:-missing})"
      fi
    else
      # A legacy record without workflow identity cannot prove quiescence.
      live_research+="${research_id}|${research_status}|no-workflow-id"$'\n'
    fi
  done <<<"$research_records"
fi

if [[ -n "$live_research" ]]; then
  warn "active /rere tasks found; wait for completion or explicitly cancel them before final backup:"
  printf '%s' "$live_research" | sed 's/^/      /' >&2
  if [[ "$ALLOW_ACTIVE_RESEARCH" != true ]]; then
    fail "preflight stopped because active repository research exists"
    exit 2
  fi
fi

if [[ $missing -ne 0 ]]; then
  exit 1
fi

ok "preflight passed; proceed only after a planned maintenance window and final quiesced backup"
