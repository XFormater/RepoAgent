# GitHubResearch_Agent BFCL V4 全量评测

## 结论

GitHubResearch_Agent 使用 DeepSeek V4 Flash 完成 BFCL V4 全量 agent-in-the-loop 评测。固定官方源码提交为 `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`，覆盖 22 个计分类别、5106 个计分样本，以及 Memory 类别所需的 111 个前置写入任务。最终完整性审计结果为：缺失 0、额外 0、重复 0、推理错误/错误占位符 0。

BFCL 官方评分器给出的系统总分为 **70.29%**。该结果衡量的是“GitHubResearch_Agent 运行时 + DeepSeek V4 Flash”的组合系统，不是 DeepSeek 模型直连结果，也不应作为官方排行榜的纯模型提交成绩。

## 系统接入边界

每次模型决策均经过项目的以下运行时组件：

- `AgentRunner`：执行共享状态机、迭代控制、provider 重试和工具结果管理。
- `ContextGovernor`：参与消息上下文治理。
- `ToolRegistry`：注册 BFCL 动态工具 schema，并执行参数校验及 capture 调用。
- 项目 OpenAI-compatible provider：调用 `deepseek-v4-flash`。

BFCL 仍负责官方数据加载、真实工具执行、状态后端、Memory 快照、Web Search 工具接口和评分。项目运行时负责决定调用什么工具及参数；BFCL 执行器负责产生权威环境状态和工具反馈，再将反馈送回下一次 `AgentRunner` 决策。由于环境未配置 SerpAPI，Web Search 接口使用适配器提供的 DDGS/DuckDuckGo 实现并复用已配置代理。

这一路径覆盖项目核心 agent runtime，但不覆盖完整产品层的子智能体、逻辑 VFS、人工介入和生产长期记忆注入。因此，本结果应描述为 **agent-in-the-loop 系统评测**。

## 数据范围与运行条件

| 项目 | 值 |
| --- | ---: |
| BFCL 版本 | V4 |
| 官方源码提交 | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` |
| 模型 | `deepseek-v4-flash` |
| 计分类别 | 22 |
| 计分样本 | 5106 |
| Memory 前置任务 | 111 |
| 基准所需生成任务总数 | 5217 |
| 温度 | 0.001 |
| 正式生成并发 | 8 |
| BFCL 估算成本 | $2.98 |
| 平均模型延迟 | 2.51 秒 |
| P95 模型延迟 | 6.86 秒 |

Format Sensitivity 对 function-calling 模型由当前官方运行器跳过，不属于上述 22 个计分类别。

5217 表示一轮完整基准所需的任务数，不包含额度故障、诊断探测和定向补跑产生的冗余 API 调用。$2.98 是 BFCL 根据最终有效结果工件汇总的估算成本，同样不表示整个故障恢复过程的实际账单。

## 官方评分结果

| 分层 | 准确率 |
| --- | ---: |
| BFCL Overall | **70.29%** |
| Non-Live | **85.17%** |
| Live | **77.79%** |
| Multi-Turn | **58.50%** |
| Agentic | **73.65%** |
| Web Search | **84.50%** |
| Memory | **62.80%** |

### Non-Live

| 类别 | 准确率 |
| --- | ---: |
| Simple 汇总 | 72.67% |
| Simple Python | 95.00% |
| Simple Java | 63.00% |
| Simple JavaScript | 60.00% |
| Multiple | 94.00% |
| Parallel | 89.00% |
| Parallel Multiple | 85.00% |
| Irrelevance Detection | 70.83% |

### Live

| 类别 | 准确率 |
| --- | ---: |
| Live Simple | 87.21% |
| Live Multiple | 75.59% |
| Live Parallel | 87.50% |
| Live Parallel Multiple | 66.67% |
| Live Irrelevance | 68.89% |
| Live Relevance | 81.25% |

### Multi-Turn

| 类别 | 准确率 |
| --- | ---: |
| Base | 73.00% |
| Missing Function | 50.50% |
| Missing Parameter | 48.50% |
| Long Context | 62.00% |

### Agentic

| 类别 | 准确率 |
| --- | ---: |
| Web Search Base | 84.00% |
| Web Search No Snippet | 85.00% |
| Memory KV | 69.68% |
| Memory Vector | 70.32% |
| Memory Recursive Summary | 48.39% |

## 记忆与数据库口径

BFCL Memory 测试要求严格使用其定义的三种可执行后端：KV、Vector 和 Recursive Summary，并依赖 111 个官方 prerequisite 会话建立状态。本次评分保持了这一标准，未将项目生产数据库替换为 BFCL 后端。

项目的短期对话状态、`AgentRunner` 消息历史和 `ContextGovernor` 在每次决策中生效。项目生产长期记忆服务所使用的 PostgreSQL、Qdrant、Neo4j 和 Redis 未注入本次 BFCL 得分，原因是跨样本检索或替换官方 Memory API 会改变基准定义并引入污染。它们的可用性与收益应由项目现有 LoCoMo、记忆收益和数据库后端验证单独报告。

因此，Memory 62.80% 表示智能体通过 BFCL 官方记忆工具管理和读取持久状态的能力，不等价于项目生产 `LongTermMemoryService` 的检索准确率。

## 故障与修复记录

评测中途 DeepSeek API 额度耗尽，导致 65 个 Multi-Turn 尾部样本、465 个 Memory 计分样本、111 个 Memory 前置任务及 200 个 Web 样本写入 `AgentRunner` 错误占位符。初版审计只识别 `Error during inference:`，曾错误地将这些占位结果统计为推理错误 0。

完成以下修复后，仅定向补跑无效 ID：

1. `AgentRunner.error` 不再被静默转换为正常模型响应，而是显式传递给 BFCL 生成器。
2. 审计器同时识别 BFCL 推理错误和项目持久化错误占位符。
3. 审计器可以生成按官方类别分组的 `--run-ids` 补跑清单，并将 Memory prerequisite ID 归入对应 Memory 类别。
4. BFCL 官方 `write(update_mode=True)` 是无锁读改写；8 线程补跑出现旧结果覆盖新结果。适配器增加进程内写锁后重新审计并补跑，最终错误数归零。

额度耗尽时生成的无效临时分数已保存在 `scores-quota-exhausted/`，不得用于能力结论。

## 可复核工件

- 运行清单：`evals/runs/bfcl-v4-agent-full-flash-20260821/manifest.json`
- 最终审计：`evals/runs/bfcl-v4-agent-full-flash-20260821/audit.json`
- 最终聚合分数：`evals/runs/bfcl-v4-agent-full-flash-20260821/scores/data_overall.csv`
- 分层分数：同目录下的 `data_non_live.csv`、`data_live.csv`、`data_multi_turn.csv` 和 `data_agentic.csv`
- 原始结果：`evals/runs/bfcl-v4-agent-full-flash-20260821/results/`
- 额度故障工件：`evals/runs/bfcl-v4-agent-full-flash-20260821/audit-quota-exhausted.json` 和 `scores-quota-exhausted/`
- 评测适配器：`scripts/run_bfcl_v4_agent_eval.py`

## 解释限制

- 使用了 BFCL 官方数据、状态执行与评分代码，但系统接入方式不是官方排行榜的纯模型 handler，因此不可直接与排行榜模型成绩作同口径排名。
- Web Search 使用 DDGS/DuckDuckGo 后端而非 SerpAPI；这也是与官方托管评测环境比较时需要保留的环境变量。
- 总分同时受 DeepSeek V4 Flash 的函数调用能力和项目运行时行为影响，不能单独归因于模型或某一个 harness 模块。
- 本轮没有做同模型直连对照实验，不能从 70.29% 推导 `AgentRunner` 或 `ContextGovernor` 的净增益。
- 生产长期记忆数据库未注入得分，Memory 分项只能说明对 BFCL 官方 Memory API 的使用能力。
