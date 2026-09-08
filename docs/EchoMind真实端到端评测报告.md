# EchoMind 真实 DeepSeek 端到端评测报告

## 1. 实验口径

- 模型：`deepseek-v4-flash`，通过项目 `.env` 中的 Anthropic 兼容地址真实调用；
- 请求集：`data/eval/campus_golden.json`；
- 数据规模：23 条意图样本、20 组业务对话，共 24 个实际对话轮次；
- 链路：意图识别 → Redis 会话 → ChromaDB/BGE 检索 → Jina Reranker → Agent/Skill/Tool → EvidenceVerifier → 单轮反思 → LLM-as-Judge/RAGAS；
- 数据性质：人工编写的校园业务黄金集，不是真实学校生产流量；
- RAG 语料：20 个校园演示知识片段；
- 运行环境：Windows 本地新版 API（端口 8002），复用本机 Redis 与 ChromaDB；模型缓存预热后运行。

## 2. 数据规模与总结果

报告共包含 25 项结果：1 项聚合意图评测和 24 个实际对话轮次。

| 指标 | 结果 |
|---|---:|
| 严格总通过率 | 16.00%（4/25） |
| 对话任务完成率 | 12.50%（3/24） |
| LLM Judge 质量通过 | 75.00%（18/24） |
| 路由正确率 | 87.50% |
| 有明确预期时 Tool 正确率 | 100.00% |
| 有明确预期时工单正确率 | 100.00% |
| 知识使用正确率 | 95.00% |
| 实际 Tool 调用量 | 27 次 |

这里的“任务完成”是严格离线门槛：回答质量、路由、Tool、工单、知识使用、RAGAS 和证据校验只要任一失败，本轮就失败。它不是用户满意度或线上自动解决率。

## 3. 意图识别

| Accuracy | 严格多标签命中率 | Macro-F1 |
|---:|---:|---:|
| 86.96% | 47.83% | 83.45% |

严格多标签命中率偏低的重要原因是标签口径不一致：当前模型同时输出“业务领域＋用户动作”，例如 `technical + query`，而多条黄金数据只标了 `technical`。因此该指标混合了真实误判和黄金标签欠标，后续应先统一标签 Schema，再重新评测。

## 4. 回答质量与 RAGAS

### LLM-as-Judge

| Relevance | Accuracy | Completeness | Helpfulness |
|---:|---:|---:|---:|
| 85.83% | 91.46% | 73.12% | 78.33% |

### RAGAS 与检索

| Faithfulness | Answer Relevancy | Context Precision | Context Recall | Answer Correctness |
|---:|---:|---:|---:|---:|
| 46.84% | 56.89% | 67.92% | 82.50% | 35.15% |

| Hit Rate | Recall@K | MRR | NDCG@K | 严格 RAGAS 通过率 |
|---:|---:|---:|---:|---:|
| 90.00% | 78.33% | 80.00% | 74.02% | 5.00%（1/20） |

当前主要问题不是“完全检索不到”：Hit Rate、MRR 和 Context Recall 已有一定基础；瓶颈是检索噪声、回答对证据的严格蕴含，以及回答与黄金答案的一致性。

## 5. 证据校验与真实反思

| 指标 | 结果 |
|---|---:|
| 进入证据校验 | 22 轮 |
| 首轮/反思后最终通过 | 16 轮 |
| 触发反思 | 7 轮 |
| 反思纠正成功 | 1 轮 |
| 条件纠正率 | 14.29%（1/7） |
| 安全降级 | 6 轮 |

失败主要集中在账单回答：模型重写后仍缺少知识库 Citation，部分反思轮没有提取到最终文本，最终被 `empty_response` 和 `missing_citation` 拦截并安全降级。这说明“规则能发现问题”已经成立，但 DeepSeek 的反思 Prompt、证据组织和兼容端点输出处理仍需优化。

## 6. 延迟

| 平均延迟 | P50 | P95（nearest-rank） | 最大值 |
|---:|---:|---:|---:|
| 8688 ms | 7344 ms | 17454 ms | 20562 ms |

本轮已预热 BGE 与 Jina 缓存，因此不把首次下载模型的冷启动时间计入以上结果。高延迟主要出现在 BillingAgent、复合 Agent、查询改写和 Judge/RAGAS 多次模型调用场景。

## 7. 可信结论

可以证明：真实 DeepSeek、Redis、ChromaDB、BGE、Jina Reranker、Agent 路由、业务 Tool、证据校验和离线评测已经形成可运行闭环；Tool、工单和知识使用的确定性检查能够执行。

不能声称：系统已有 80% 以上自动解决率、RAG 已全面达标或反思能稳定修复幻觉。当前最优先的三个改进点是：

1. 统一多标签黄金数据的领域/动作标注口径；
2. 修复账单回答的 Citation 约束与 DeepSeek 空最终文本问题；
3. 优化检索噪声、证据蕴含和 Answer Correctness，再用同一固定集复测。

原始结果：`data/eval/results/campus_end_to_end_live_large.json`。
