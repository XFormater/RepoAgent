# Harness Quality Evaluation

Harness evaluation is an offline, deterministic quality gate for recorded agent runs.
It complements runtime Reviewer tasks: Reviewers decide whether one live result may
continue, while an evaluation suite detects regressions across a stable golden dataset.

## Dataset

Create `evals/harness.json`:

```json
{
  "thresholds": {
    "minCaseScore": 1.0,
    "minPassRate": 0.95,
    "minMeanScore": 0.98,
    "maxMeanScoreRegression": 0.01
  },
  "cases": [
    {
      "caseId": "repository-grounding-001",
      "tags": ["repository", "grounding"],
      "requiredTerms": ["src/main.py"],
      "requiredPatterns": ["line\\s+\\d+"],
      "forbiddenTerms": ["I guess", "probably"],
      "minOutputChars": 80,
      "maxOutputChars": 4000,
      "maxLatencyMs": 30000,
      "maxToolCalls": 12
    }
  ]
}
```

Each configured assertion contributes equally to the case score. The report retains
every failed criterion, so a score maps to a concrete failure. Tags produce slices
such as safety, grounding, repository, language, or task family.

Record provider-independent observations as JSON:

```json
[
  {
    "caseId": "repository-grounding-001",
    "output": "The implementation is in src/main.py at line 42.",
    "succeeded": true,
    "latencyMs": 8200,
    "toolCalls": 4
  }
]
```

## CI gate

```bash
nanobot harness-eval evals/harness.json artifacts/observations.json \
  --baseline evals/baseline-report.json \
  --output artifacts/evaluation-report.json
```

The command exits with status 1 when pass rate or mean score misses its threshold, or
when mean score regresses beyond `maxMeanScoreRegression`. Keep the golden dataset in
version control. Generate observations with isolated session keys and fixed model,
prompt, tool, and repository versions so changes are attributable.

Start with deterministic criteria for CI. An LLM judge can later emit an additional
criterion, but it should be calibrated against human labels, pinned to a model/prompt
version, repeated to measure variance, and never replace safety assertions.
