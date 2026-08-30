# Fixed-corpus token benchmark prompt

Analyze exactly these three 2026 autoregressive image-generation repositories:

1. `ShareLab-SII/UniAR` — local source: `research/autoregressive_image_generation_2026/UniAR`
2. `amazon-far/BAR` — local source: `research/autoregressive_image_generation_2026/BAR`
3. `stepfun-ai/NextStep-1` — local source: `research/autoregressive_image_generation_2026/NextStep-1`

Produce a Chinese technical report. Do not stop at README summaries. For every repository,
inspect implementation files and cite repository-relative paths. Cover:

- why it qualifies as a 2026 autoregressive image-generation project;
- visual token representation and tokenizer/decoder design;
- autoregressive factorization, prediction target, attention or decoding strategy;
- training pipeline, losses and important configuration;
- inference entry point and sampling procedure;
- released checkpoints/code completeness and practical hardware requirements;
- technical strengths, limitations and reproducibility risks.

Finish with a cross-project comparison of architecture, generation paradigm, training and
inference cost, openness, reproducibility, and suitable use cases. Clearly separate claims
verified in source code from claims found only in documentation.
