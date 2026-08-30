# LoCoMo long-term-memory benchmark

## Scope and protocol

This project pins the official ACL 2024 LoCoMo release from
`snap-research/locomo` commit
`3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376`. The unchanged dataset SHA-256 is
`79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
It contains 10 conversations, 5,882 dialogue turns, and 1,986 QA items.

The fixed default `locomo-100-fixed-smoke` evaluates 100 questions, with 10
questions from each conversation and category quotas proportional to the full
dataset: 14 multi-hop, 16 temporal, 5 open-domain, 42 single-hop, and 23
adversarial. Stable SHA-256 ranking selects questions within every
conversation/category cell.

Each dated raw dialogue turn is pre-indexed as one `MemoryRecord`. Retrieval
uses the real `LongTermMemoryService`, vector store adapter, retrieval pipeline,
and bounded `MemoryInjector`. Both variants use the same DeepSeek model,
temperature 0, 64 output tokens, question, scorer, and deterministic alternating
call order:

- Memory Off receives no recalled entries.
- Memory On receives the top-10 entries selected by the memory pipeline.

The scorer reproduces the official LoCoMo rules: Porter-stemmed token F1 for
single-hop, temporal, and open-domain questions; comma-aware partial F1 for
multi-hop questions; and the official no-information decision for adversarial
questions. The released category-5 records use `adversarial_answer` rather than
the stale `answer` lookup in upstream `gpt_utils.py`; the runner treats that
field only as the incorrect multiple-choice distractor.

## DeepSeek LoCoMo-100 baseline — 2026-08-13

The authoritative run used `deepseek-v4-pro` and produced 200 real model calls.
Artifacts are under `evals/runs/locomo-100-deepseek-20260813-final/`.

| Metric | Memory Off | Memory On | Delta |
|---|---:|---:|---:|
| All-question mean F1 | 16.03% | 30.38% | +14.35 pp |
| Non-adversarial mean F1 | 7.83% | 12.18% | +4.35 pp |
| Multi-hop F1 (14) | 7.44% | 9.97% | +2.53 pp |
| Temporal F1 (16) | 0.00% | 8.95% | +8.95 pp |
| Open-domain F1 (5) | 32.76% | 10.35% | -22.41 pp |
| Single-hop F1 (42) | 7.97% | 14.36% | +6.39 pp |
| Adversarial accuracy (23) | 43.48% | 91.30% | +47.83 pp |

Paired outcomes were 24 wins, 68 ties, and 8 losses. Evidence Recall@10 was
19.59%, and evidence Hit Rate@10 was 21.00%. Mean injected context was 2,345
characters. Provider-reported prompt use rose from 8,848 tokens in Memory Off
to 73,256 tokens in Memory On (8.28x); completion use was 434 versus 451 tokens.

The aggregate uplift is real for this fixed run, but most of it comes from the
official adversarial slice. The answerable-question uplift is only 4.35 points,
and open-domain quality regressed. The low evidence recall identifies retrieval
as the main limitation of the current local profile.

## Persistent hybrid-memory result — 2026-08-20

The v2.2 run keeps the same dataset, fixed question IDs, paired call order, and
official scorer, but uses `deepseek-v4-flash` with a 2,048-token completion
budget. It indexes 8,967 records: 5,882 dated dialogue turns, 2,541 official
observations, 272 per-session observation bundles, and 272 official session
summaries. QA answers, QA evidence annotations, and `event_summary` are never
indexed. The records use four explicit storage roles:

- PostgreSQL is the authoritative record store and performs English full-text
  candidate recall.
- Qdrant stores 1,024-dimensional deterministic lexical-feature vectors.
- Neo4j stores the speaker/session provenance graph and expands related
  memories, using batched projection writes.
- Redis stores the bounded latest-session short-term context and idempotency
  state.

PostgreSQL and Qdrant candidates are combined with weighted reciprocal-rank
fusion, expanded through Neo4j, and BM25-reranked. Up to 56 long-term records
are bounded to 24,000 characters; Redis contributes at most 6,000 additional
characters. Relative dates are normalized from the source turn and session
date. Every database was verified by a real write/read or query for all ten
conversations before the paired QA calls ran.

Artifacts are under
`evals/runs/locomo-hybrid-100-v22-deepseek-flash-20260820/`.

| Metric | Memory Off | Memory On | Delta |
|---|---:|---:|---:|
| All-question mean F1 | 21.20% | **64.99%** | **+43.79 pp** |
| Non-adversarial mean F1 | 1.56% | 58.43% | +56.87 pp |
| Multi-hop F1 (14) | 0.00% | 52.93% | +52.93 pp |
| Temporal F1 (16) | 0.00% | 50.43% | +50.43 pp |
| Open-domain F1 (5) | 24.00% | 25.90% | +1.90 pp |
| Single-hop F1 (42) | 0.00% | 67.18% | +67.18 pp |
| Adversarial accuracy (23) | 86.96% | 86.96% | 0.00 pp |

Paired outcomes were 65 wins, 32 ties, and 3 losses. Evidence recall across the
56 long-term results plus bounded short-term context was 86.59%, with a 92.00%
hit rate. Mean injected context was 28,477 characters. All 100 Memory On calls
ended with `finish_reason=stop`; there were no empty answers. One Neo4j query
exceeded its five-second budget and degraded to PostgreSQL/Qdrant retrieval.

The 64.99% result clears the project LoCoMo-100 target, but it must not be
reported as an official full LoCoMo score. In particular, answerable-only F1 is
58.43%, open-domain inference remains weak, and three paired losses show that
recalled context can still turn a correct abstention into a plausible answer.

Run the persistent profile with:

```bash
set -a; source .env; set +a
.venv/bin/python scripts/run_locomo_hybrid_eval.py \
  --run-id locomo-hybrid-100-v22-deepseek-flash-20260820
```

## Interpretation boundaries

Both reported runs are 5.04% question subsets covering all conversations, not
full official LoCoMo scores. The v1 baseline uses a 384-dimensional hash
embedding, identity reranker, and isolated in-memory adapters. The v2.2 run
tests deployed Redis/PostgreSQL/Qdrant/Neo4j persistence and retrieval, but
holds memory extraction constant by indexing benchmark-provided memory corpora.
Neither run measures the quality of online LLM fact extraction.

For a stronger memory baseline, replace hash retrieval with a semantic embedding
model, enable a query-aware reranker, add temporal/session-aware retrieval, and
rerun the identical fixed selection before attempting the 1,986-question full
evaluation.
