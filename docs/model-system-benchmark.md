# Model-in-the-Loop System Benchmark

This benchmark is the primary system evaluation for GitHubResearch Agent. It
uses the configured real model on every case. The older deterministic
self-capability suite remains a unit-level regression aid; its 30 pytest
contracts are not reported as model-in-the-loop agent performance.

## Evaluation layers

| Layer | Question | Primary measurements |
|---|---|---|
| Harness Regression | Can the model and runtime repeatedly complete fixed tool workflows? | exact task result, tool sequence, stop reason, latency, token use |
| Context Governance | Does governance reduce context while retaining answer quality? | paired on/off accuracy, input-token reduction |
| Memory Benefit | Does retrieved memory improve otherwise identical tasks? | paired memory on/off accuracy, retrieval hit rate, injected size |
| Recovery Correctness | Does execution resume at the safe boundary after faults? | recovered result, retry count, checkpoint phase, duplicate side effects |

Harness and recovery cases use simple opaque values so the model cannot solve a
task from prior knowledge. Context and memory cases are matched-pair
experiments: both variants use the same model, generation settings, task, and
scorer, and differ only in the module under evaluation. The memory experiment
uses the real retrieval and injection pipeline with temporary in-memory storage
to isolate module benefit from database availability; the LLM call remains an
online call to the configured provider.

## Three evaluation paths

The versioned suite at `evals/suites/agent_system_model_v1.json` is the fixed
benchmark path. Context and memory variants produce the controlled-experiment
path. All cases write into one run directory for the artifact-aggregation path:

```text
evals/runs/<run-id>/
├── manifest.json
├── events.jsonl
├── comparisons.json
├── report.json
└── cases/
    └── <case-id>__r<repetition>.json
```

The manifest pins the benchmark version, suite/config hashes, Git commit,
provider type, model, context window, and generation settings. It never records
resolved credentials. `events.jsonl` is an append-only case timeline;
`comparisons.json` contains paired deltas; `report.json` is the aggregate gate.

## Run

The default config references secrets through environment variables. Load the
same protected environment file used by the systemd services, then run:

```bash
set -a
. ./.env
set +a
.venv/bin/python scripts/run_agent_system_benchmark.py
```

Run one layer or case:

```bash
.venv/bin/python scripts/run_agent_system_benchmark.py \
  --layer context_governance

.venv/bin/python scripts/run_agent_system_benchmark.py \
  --case recovery-checkpoint-resume
```

Use `--repetitions 3` or more for a stability run. Repetitions are retained as
separate case artifacts so flaky model/runtime outcomes remain visible instead
of being overwritten by an average.

## Interpretation

The result is model-specific system evidence, not a model-independent proof.
Keep the provider, model, prompt suite, generation settings, code commit, and
feature configuration pinned when comparing runs. BFCL, AgentBench OS, and
SWE-bench should continue to be reported separately as model/task capability
benchmarks rather than merged into these system-layer scores.

## 2026-08-13 DeepSeek baseline

The authoritative v1 run used `deepseek-v4-pro` with temperature 0 and retained
all artifacts under `evals/runs/model-system-v1-deepseek-20260813-final/`.

| Layer | Passed | Result |
|---|---:|---|
| Harness Regression | 3 / 3 | fixed single-tool, multi-tool, and write/read workflows completed |
| Context Governance | 3 / 3 | accuracy stayed at 100%; estimated input tokens fell by 80.60% on average |
| Memory Benefit | 3 / 3 | control accuracy 0%, memory-on accuracy 100%, retrieval hit rate 100% |
| Recovery Correctness | 3 / 3 | transient retry, acknowledgement loss, and checkpoint resume recovered with zero duplicate side effects |

This is a small baseline of 12 fixed scenarios, not a general task-success
claim. Increase repetitions and expand the fault/task matrix before using it as
a release-grade statistical result.

## Memory Benefit v2 expansion

The dedicated v2 suite expands memory evaluation from 3 to 20 paired cases:
12 memory-required recall/composition probes, 4 current-message-overrides-old-
memory cases, and 4 irrelevant-memory non-regression cases. It is stored at
`evals/suites/memory_benefit_v2.json` and should be reported separately from the
original 12-case four-layer baseline.

The 2026-08-13 `deepseek-v4-pro` run produced 40 real-model variant calls. The
control group scored 40%, memory-on scored 100%, so the aggregate gain was 60
percentage points rather than the earlier 100-point result. Within the 12
deliberately memory-required cases the gain remained 100 points by construction;
the more important safety result was 0% harmful regression across the 8 conflict
and irrelevant-memory cases. Mean prompt tokens increased from 66.4 to 164.1,
an overhead of 97.7 tokens (147%). These synthetic opaque-fact probes measure
causal retrieval/injection behavior, not expected uplift on general user tasks.
An immediate second complete run reproduced all 20 pass decisions and all 40
normalized control/treatment answers exactly.

## LoCoMo long-conversation baseline

The separate LoCoMo-100 model-in-the-loop run evaluates official multi-session
conversation QA through the project memory retrieval/injection path. On the
fixed 100-question subset, `deepseek-v4-pro` improved from 16.03% Memory-Off F1
to 30.38% Memory-On F1 (+14.35 points). Excluding the official adversarial
category, the change was 7.83% to 12.18% (+4.35 points); evidence Recall@10 was
19.59%. See `docs/locomo-memory-benchmark.md`. This is explicitly a smoke subset,
not a full 1,986-question LoCoMo claim.

## Controlled Context Governance v2

The dedicated v2 suite replaces the three all-noise probes with 16 paired
information-graded cases: latest authority, recent dependency, system-summary
anchor, conflict override, early dependency, split early/recent dependency, and
recent/early historical tool results. Each pair shares one serialized source
history and source hash; only `ContextGovernor` is switched on or off. Variant
order is deterministic and recorded. The primary size metric is provider-
reported `prompt_tokens`; local estimates remain diagnostic only.

The authoritative 2026-08-13 `deepseek-v4-pro` run used 32 real-model calls.
Control accuracy was 100% (16/16), while governed accuracy was 62.5% (10/16), a
37.5-point regression. Mean prompt tokens fell from 4,265.8 to 1,382.6. The
mean per-case reduction was 65.07%, and the total-token weighted reduction was
67.59% (68,253 to 22,122 tokens). The suite therefore failed its 75% retention
gate despite substantial token savings.

All latest, recent, system-summary, conflict, and recent-tool cases survived.
All three early-fact cases, both split-dependency cases, and the early-tool-
result case failed after governance. Three complete runs, including the final
mixed variant order, produced identical normalized answers and pass decisions.
This is the current governance boundary: tail truncation is effective for
recent context but unsafe for unsummarized long-range dependencies.

### Relevance-preserving remediation

The governor now partitions history into complete user-owned turns, retains
query-relevant early turns and a bounded number of complete historical tool
transactions, and spends the remaining unchanged budget on the recent tail.
Relevant blocks displace noise; they do not raise `contextBlockLimit`.

After the change, two complete `deepseek-v4-pro` runs both scored 16/16 in the
same v2 controlled suite. Provider-reported prompt tokens fell from 68,253 to
22,172 per run: a total-token weighted reduction of 67.51%, with a 64.91% mean
per-case reduction. Control and governed accuracy were both 100%, and harmful
regression was 0%. All early-fact, split-dependency, and early-tool-result cases
that previously failed now pass. The committed gate now requires 100% case
pass rate and at least 60% mean provider prompt-token reduction.
