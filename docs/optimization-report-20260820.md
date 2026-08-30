# Harness 记忆优化与 BFCL 复测报告（2026-08-20）

## 结论

本轮在固定 LoCoMo 100 题、同一 DeepSeek V4 Flash、同一持久化混合记忆后端上，将上下文策略从固定 `ranked/24k` 优化为输入形态自适应路由：普通问答使用 `atomic-first/12k`，选择式证据核验使用 `ranked/24k`。自适应方案重复两次的 Memory On F1 分别为 **69.81%** 和 **65.77%**，均高于同口径固定 24k 基线 **64.38%**；两次均值为 **67.79%（+3.41pp）**。平均 Prompt Token 为 **580,490.5（-24.58%）**，没有检索降级，四个真实后端均通过逐会话验证。

本轮 BFCL（用户原文中的 BFCN 按仓库已有 BFCL 处理）V4 `simple_python` 400 题重测为 **372/400，93.00%**。模型错误和空工具调用均为 0；28 个失败中 27 个是参数不精确，1 个是调用数量错误，未出现函数名选择错误。

## 评测口径修正

旧报告把 reranker 返回的全部记录视为模型可见证据，但 `MemoryInjector` 会按字符预算截断，因此旧字段 `Evidence Recall@56 + STM = 86.59%` 高估了实际注入召回。本轮完成以下修正：

- `MemoryInjector.render_with_records()` 同时返回渲染文本和真正进入 Prompt 的记录。
- 每题同时记录 `candidateIds`、`injectedIds`、`injectedRecordCount` 和有效上下文策略。
- 聚合指标改为 `evidenceRecallInInjectedContextPlusShortTerm`，只计算最终进入长上下文的记录和 Redis STM。
- 增加检索并发信号量。首次全量运行因同一会话 10 个检索突发出现 4 次 Neo4j timeout；将检索并发固定为 1 后，三个正式 100 题运行均为 0 次降级。中止和失败工件保留并明确标记，未混入分数。

因此，本报告中的 85.09%、78.84% 和 81.51%–81.84% 是“模型/路由阶段实际收到的检索上下文召回”，不能与旧报告的 86.59% 直接比较。

## 固定诊断集对照

诊断集固定为 20 题：13 个 v2.2 非对抗失败题、3 个记忆引入错误的对抗题、4 个高分保护题。所有方案均重新调用模型，同时运行 Memory On/Off；它是有意加难的开发集，绝对分数不代表 LoCoMo 总体能力，只用于快速筛选。基线为 Flash、`ranked/24k`。

| 方案 | Memory On F1 | 相对基线 | 实际证据召回 | Prompt Token | Token 变化 | 结论 |
|---|---:|---:|---:|---:|---:|---|
| ranked/24k 基线 | 32.16% | — | 66.25% | 152,284 | — | 同期对照 |
| atomic-first/12k | 35.96% | +3.80pp | 57.50% | 105,317 | -30.84% | Pilot 最优，但召回下降 |
| atomic-first/24k | 34.44% | +2.28pp | 66.25% | 154,274 | +1.31% | 排序单独收益小，成本上升 |
| atomic-only/Top-64/12k | 30.82% | -1.34pp | 52.50% | 99,074 | -34.94% | 丢失 bundle/summary 的归纳收益 |
| LLM 二阶段证据压缩 | 25.58% | -6.58pp | 66.25%（压缩器输入） | 158,944 | +4.37% | 漏掉多跳证据；总 Token +15.15% |
| V4 Pro 路由 | 24.17% | -7.99pp | 66.25% | 150,704 | -1.04% | 模型切换无收益 |
| deliberate 回答契约 | 33.53% | +1.37pp | 66.25% | 154,564 | +1.50% | 小幅方向性收益，不足以升级 |

另用 `deepseek-reasoner` 对 3 个代表题做边界探测，Memory On 仅 1/3，计数和日期题仍失败，未继续扩大该分支。

## 固定 100 题全量结果

数据集和选择规则保持不变：LoCoMo 固定 10 个会话、每会话 10 题，共 100/1,986 题；模型为 `deepseek-v4-flash`，temperature 0，Memory On/Off 顺序按 case ID 固定交错。PostgreSQL、Qdrant、Neo4j、Redis 均是真实服务，不使用内存替身。

| 指标 | ranked/24k 基线 | atomic-first/12k 成本档 | adaptive 第 1 次 | adaptive 第 2 次 | adaptive 两次均值 |
|---|---:|---:|---:|---:|---:|
| Memory On F1 | 64.38% | 63.26% | 69.81% | 65.77% | **67.79%** |
| 相对基线 | — | -1.12pp | +5.43pp | +1.40pp | **+3.41pp** |
| 非对抗 F1 | 57.64% | 57.48% | 62.09% | 58.15% | **60.12%** |
| 对抗 F1 | 86.96% | 82.61% | 95.65% | 91.30% | **93.48%** |
| 实际证据召回 | 85.09% | 78.84% | 81.84% | 81.51% | **81.67%** |
| 实际证据命中率 | 90.00% | 85.00% | 88.00% | 88.00% | **88.00%** |
| Memory On Prompt Token | 769,676 | 527,323 | 580,488 | 580,493 | **580,490.5** |
| Prompt Token 相对基线 | — | -31.49% | -24.58% | -24.58% | **-24.58%** |
| 平均注入字符 | 28,485.09 | 17,126.71 | 19,594.83 | 19,595.08 | **19,594.96** |
| 检索降级查询 | 0 | 0 | 0 | 0 | **0** |
| 四后端全部验证 | 是 | 是 | 是 | 是 | **是** |

正式工件：

- 基线：`evals/runs/locomo-hybrid-100-v24-ranked24k-flash-20260820/report.json`
- 12k 成本档：`evals/runs/locomo-hybrid-100-v23-atomic12k-flash-r2-20260820/report.json`
- 自适应第 1 次：`evals/runs/locomo-hybrid-100-v25-adaptive-flash-20260820/report.json`
- 自适应第 2 次：`evals/runs/locomo-hybrid-100-v25-adaptive-flash-rep2-20260820/report.json`

### 如何解释自适应收益

固定 12k 把普通问答的重复 bundle/summary 压缩掉，非对抗 F1 相对 24k 基线几乎持平（单次成本档仅 -0.15pp），但它会伤害需要严格证据核验的选择题。自适应策略只读取输入形态：普通问答走原子事实优先 12k；有明确选项的请求保留 ranked/24k。它不读取 gold answer，也不使用 LoCoMo evidence 标注参与推理。

两次自适应运行相差 4.03pp，说明单次最高值 69.81% 不能当作稳定下界。可信表达应为：**两次均值 67.79%，范围 65.77%–69.81%，两次都高于同口径 64.38% 基线，同时 Prompt Token 稳定下降 24.58%。**

## BFCL V4 simple_python 重测

| 项目 | 2026-08-13 首次 | 2026-08-13 复测 | 2026-08-20 本轮 |
|---|---:|---:|---:|
| 通过数 | 371/400 | 375/400 | **372/400** |
| 准确率 | 92.75% | 93.75% | **93.00%** |
| 模型/API error | 0 | 0 | **0** |
| 空工具调用 | — | — | **0** |

本轮模型为当前配置主模型 `deepseek-v4-pro`，平均单题延迟 1,788.53ms。28 个失败中：

- 正确函数、参数不匹配：27；
- 工具调用数量不等于 1：1；
- 函数名选错：0；
- 三次运行都失败的稳定难题：24；
- 与上一次相比仅 3 题发生通过/失败翻转。

因此模型连接已不是 BFCL 瓶颈。当前 7% 误差集中在参数值、类型或可接受答案的精确一致性，以及 1 个多调用行为。

工件：`evals/results/bfcl_v4_simple_python_deepseek_v4_pro_20260820.json`

数据 SHA-256：

- `simple_python.json`: `82dd63ba502eb2520c6b5d1d9a5c4b590e03ff261565175561f6228a367d1991`
- `simple_python_answers.json`: `90cd5bc653690ee8e459b5b3f3fc9458606f7f3fcbf795bb51b7dc581f8c86dc`

## 已确认的瓶颈

1. **模型非确定性**：相同自适应配置两次相差 4.03pp；若需要更窄置信区间，应继续增加重复次数，而不是继续按单次失败调 Prompt。
2. **证据利用而非单纯召回**：ranked/24k 实际召回已达 85.09%，但仍有大量计数、跨日期聚合和严格实体对齐错误；继续扩大上下文会增加噪声和 Token。
3. **剩余检索缺口**：自适应实际召回均值 81.67%，仍有约 18.33% gold evidence 未进入最终上下文。词法哈希 embedding + BM25 对图像描述、同义改写和低词面重合题有限。
4. **对抗证据对齐**：模型容易把对话中真实但属于另一事件的短语错误迁移到问题事件上。仅增加“谨慎回答”提示没有稳定解决。
5. **数据库查询时延**：限流后没有超时，但平均检索仍约 5.7 秒；下一阶段应优化 Neo4j 扩展候选和索引，而不是提高本地并发。
6. **BFCL 参数精度**：连接和函数选择已经稳定，继续提升需要面向参数复制、类型归一化和单调用约束优化。

## 最终推荐配置

- LoCoMo 评测 profile 的准确率/成本平衡：`--context-policy adaptive --long-term-max-chars 12000 --multiple-choice-max-chars 24000 --retrieval-top-n 56`。
- 评测稳定性：本地单用户环境使用 `--retrieval-concurrency 1`；模型调用仍可保持 `--concurrency 4`。
- 极限召回排障：保留 `ranked/24k` 作为诊断基线。
- 低成本模式：可选 `atomic-first/12k`，Prompt Token -31.49%，但总体 F1 单次下降 1.12pp，不应宣称准确率提升。
- 不采用：atomic-only、LLM 二阶段压缩、V4 Pro/Reasoner 路由、单纯 deliberate Prompt。

`adaptive` 当前是 LoCoMo runner 的实验 profile，不是生产 agent 的全局默认值；原子/聚合记录类型来自该评测的分层索引语料。可直接进入生产核心的是实际注入记录口径与注入器 API。若要把自适应打包推广到普通用户记忆，必须先为生产记录增加稳定的粒度/来源元数据，并用非 LoCoMo 数据复验，不能把本报告的 +3.41pp 外推为所有应用的收益。

## 复现命令

```bash
set -a
. ./.env
set +a

.venv/bin/python scripts/run_locomo_hybrid_eval.py \
  --run-id locomo-hybrid-100-v25-adaptive-reproduction \
  --model deepseek-v4-flash --provider deepseek --max-tokens 2048 \
  --concurrency 4 --retrieval-concurrency 1 --retrieval-top-n 56 \
  --context-policy adaptive --context-transform direct \
  --answer-policy baseline --long-term-max-chars 12000 \
  --multiple-choice-max-chars 24000

.venv/bin/python scripts/run_bfcl_native_eval.py \
  --output evals/results/bfcl_v4_simple_python_deepseek_v4_pro_reproduction.json \
  --concurrency 4
```

## 代码与验证

本轮新增或修改的核心位置：

- `nanobot/memory_system/retrieve/injector.py`：返回实际注入记录；
- `scripts/run_locomo_hybrid_eval.py`：真实召回口径、多 case pilot、上下文策略/预算/模型参数、自适应路由、检索限流和完整工件字段；
- `tests/memory_system/test_retrieve_pipeline.py`：验证注入器只报告真正装入预算的记录。

验证结果：定向 memory retrieval 测试 **8/8 通过**，Ruff 与 BasedPyright 通过；临时取消 HTTP(S) 代理后完整项目回归 **5909 passed、18 skipped**。保留代理运行时曾有 44 个 localhost HTTP/WebSocket 测试失败，针对相同失败源文件取消代理后 **200/200 通过**，确认是回环流量被代理截断，而非本轮记忆代码回归。
