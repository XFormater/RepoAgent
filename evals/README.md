# Agent evaluation datasets

## Agent self-capability offline benchmark

`suites/agent_self_capability_v1.json` is a balanced, versioned benchmark for
five product capabilities: planning/state-machine recovery, logical VFS and
sandbox isolation, delegation/subagents, memory management, and live human
intervention. Its 30 cases run in isolated pytest processes with fake providers,
temporary workspaces, stripped API credentials, and no network dependency.

```bash
.venv/bin/python scripts/run_agent_offline_benchmark.py
```

The report is written to `results/agent_self_capability_v1_report.json`. The v1
gate requires 100% overall and 100% for every capability slice. This measures
deterministic product behavior, not model intelligence.

The files under `vendor/` are pinned evaluation inputs, not training data.

| Dataset | Coverage | Pinned revision | Local subset |
|---|---|---|---|
| BFCL V4 | Function selection and argument grounding | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | 400 simple Python cases and answers |
| AgentBench | OS/file/process tool use | `d1e4a10db08c87075c78972e48ecc182be03e2d5` | 26 OS dev cases |
| SWE-bench Lite | Real repository issue resolution | Hugging Face artifact SHA-256 `1e17ff5c...918bd8f` | 23 dev cases |
| LoCoMo | Very long-term conversational memory QA | `3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376` | Full 10-conversation input; fixed 100-QA default run |

Sources: BFCL from `ShishirPatil/gorilla`, AgentBench from `THUDM/AgentBench`, and
SWE-bench Lite from `SWE-bench/SWE-bench_Lite`. Upstream license files/cards are kept
beside each dataset. GAIA is intentionally absent because it requires accepting gated
access terms.

Generate a 20-case, side-effect-free BFCL smoke suite:

```bash
python scripts/prepare_harness_benchmarks.py
```

Each generated case includes `taskInput`. Run that prompt in a fresh isolated Agent
session and record `caseId`, `output`, `succeeded`, `latencyMs`, and `toolCalls`; then
score the records with `nanobot harness-eval`.

Do not execute AgentBench's `start` or `init` commands on a developer workstation.
They are designed for disposable benchmark containers and may create files or persistent
background processes. SWE-bench likewise requires a repository sandbox at the pinned
base commit for patch execution and test-based scoring.

## LoCoMo model-in-the-loop memory benchmark

The LoCoMo runner pre-indexes official dated dialogue turns through the project
memory service, executes paired Memory Off/On DeepSeek calls, applies the
official category-specific token-F1 rules, and aggregates retrieval evidence.
The default covers 100 fixed questions across all 10 conversations; pass
`--per-conversation 0` only when intentionally running all 1,986 questions.

```bash
set -a; source .env; set +a
.venv/bin/python scripts/run_locomo_memory_eval.py
```

See `docs/locomo-memory-benchmark.md` for the protocol, baseline, and reporting
limitations. A 100-question result must be labeled LoCoMo-100 smoke rather than
an official full-dataset score.
