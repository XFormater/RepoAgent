# Agent Self-Capability Offline Benchmark

## Purpose

This benchmark measures deterministic product behavior of the agent runtime,
independently of model quality, API availability, network access, and deployed
memory databases. Version 1 contains 30 equally weighted cases, balanced across
five capability domains with six cases in each domain.

| Capability domain | Observable contracts |
|---|---|
| Planning and state machine | tool-loop integrity, sustained-goal continuation, iteration boundaries, wall timeout, finalization, length recovery |
| Logical VFS and sandbox | namespace and tenant isolation, read-only shared skills, quotas, Bubblewrap mount policy, secret redaction |
| Delegation and subagents | child tool boundaries, admission control, slot accounting, cancellation, runtime snapshotting, cleanup ordering |
| Memory management | core-memory I/O, concurrent cursors, compaction safety, idempotent recording, bounded retrieval, restart recovery |
| Human intervention | post-tool injection, post-response injection, active-turn routing, boundary injection, stop, checkpoint restoration |

## Execution contract

- Every case runs as one exact pytest node in a fresh subprocess.
- Selected cases use fake providers, temporary workspaces, and local in-memory
  or temporary-file storage.
- The runner removes API credentials and proxy variables from child processes,
  sets `NO_PROXY=*`, and does not load the deployed database configuration.
- A case receives a binary score: the capability contract either passes or
  fails. Durations are diagnostic only and do not affect correctness.
- The v1 quality gate requires a 100% overall pass rate and 100% pass rate in
  every capability domain, preventing a strong category from hiding a weak one.

## 2026-08-13 result

| Run | Passed | Pass rate | Total isolated-process time |
|---|---:|---:|---:|
| Repetition 1 | 30 / 30 | 100% | 58.189 s |
| Repetition 2 | 30 / 30 | 100% | 57.721 s |

The two repetitions produced identical per-case outcomes. The retained JSON
report is from repetition 2.

| Capability domain | Passed | Pass rate | Mean case latency (run 2) |
|---|---:|---:|---:|
| Planning and state machine | 6 / 6 | 100% | 2,195 ms |
| Logical VFS and sandbox | 6 / 6 | 100% | 1,197 ms |
| Delegation and subagents | 6 / 6 | 100% | 2,221 ms |
| Memory management | 6 / 6 | 100% | 1,782 ms |
| Human intervention | 6 / 6 | 100% | 2,225 ms |

## Reproduce

```bash
.venv/bin/python scripts/run_agent_offline_benchmark.py
```

Run a single capability slice:

```bash
.venv/bin/python scripts/run_agent_offline_benchmark.py \
  --category planning_state_machine
```

Inputs and outputs:

- Dataset: `evals/suites/agent_self_capability_v1.json`
- Runner: `scripts/run_agent_offline_benchmark.py`
- Machine-readable report: `evals/results/agent_self_capability_v1_report.json`

## Interpretation and limitations

A 100% score means the committed deterministic scenarios satisfy their current
behavioral contracts. It does not claim that arbitrary user tasks will succeed,
that a particular LLM has a 100% success rate, or that production infrastructure
is available. Provider intelligence should remain covered by BFCL, AgentBench,
and SWE-bench; live database and channel availability require separate deployment
health checks. New regressions outside these 30 contracts require adding a new
versioned case rather than silently changing the v1 meaning.
