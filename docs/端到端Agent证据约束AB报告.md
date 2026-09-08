# EchoMind 端到端 Agent 证据约束 A/B 报告

## 实验边界

- 数据集：`data/eval/campus_end_to_end_compact.json`
- SHA-256：`08f7e72282f0c32b4803a68b36b728ef6a6188c8293d6bb03fea73619af75ca3`
- Baseline：`data/eval/results/campus_end_to_end_compact_latest.json`
- Upgrade：`data/eval/results/campus_end_to_end_compact_grounded_latest.json`
- 两轮使用完全相同的 6 条意图样本和 6 个真实对话轮次。
- 这是小规模代表性回归集，不等同于生产准确率或全量测试集结果。

## 改动

1. 对齐生产 Pattern 与离线网格搜索的口径：强业务关键词命中记为二值证据，再由融合权重限制其影响。
2. 明确校园卡消费、充值、余额、扣款属于 Billing；统一认证、密码、邮箱和账号锁定属于 Account。
3. 明确“转人工”必须包含 escalation 标签。
4. 知识库生效时注入证据约束：只能依据引用和成功 Tool 结果陈述事实，禁止补充证据外的时间、概率、政策和根因。
5. 查询改写结果按稳定文档 ID 去重，避免同一文档因不同相似度分数重复占用 Top-K。

## 实测结果

| 指标 | Baseline | Upgrade | 变化 |
|---|---:|---:|---:|
| 对话路由正确率（旧主意图口径） | 66.67% | 83.33% | +16.66 个百分点 |
| Tool 正确率 | 50.00% | 100.00% | +50.00 个百分点 |
| 工单正确率 | 100.00% | 100.00% | 持平 |
| 知识使用正确率 | 80.00% | 100.00% | +20.00 个百分点 |
| 任务完成率 | 50.00% | 83.33% | +33.33 个百分点 |
| 平均延迟 | 7203.17 ms | 6554.67 ms | -9.00% |

关键 401 RAG 用例：

| RAGAS 指标 | Baseline | Upgrade | 变化 |
|---|---:|---:|---:|
| Faithfulness | 10.53% | 84.62% | +74.09 个百分点 |
| Answer Relevancy | 83.91% | 85.63% | +1.72 个百分点 |
| Context Precision | 50.00% | 50.00% | 持平 |
| Context Recall | 100.00% | 100.00% | 持平 |
| Answer Correctness | 25.00% | 29.00% | +4.00 个百分点 |

## 结果解释

- 校园卡查询从 AccountAgent 修正为 BillingAgent，并真实调用 `query_campus_card`。
- 转人工请求从普通 request 修正为 escalation。
- 401 回答删除大量证据外推测，Faithfulness 明显提升。
- Answer Correctness 仍偏低，说明参考答案与生成答案的事实对齐仍需优化；不能宣称 RAGAS 已全面达标。
- 复测后又修正了多标签评测语义：`query + technical` 且路由到 TechnicalAgent 应判路由正确。该代码修复已通过单元测试和一次真实 `/chat` 验证，但没有回写上述已落盘的 Upgrade 指标。

## 简历可用表述

> 基于同一小规模端到端回归集进行错例驱动迭代，对齐在线 Pattern 与离线标定口径，并为 RAG 回答增加证据约束；实测 Tool 正确率由 50% 提升至 100%，任务完成率由 50% 提升至 83.33%，关键校园网 401 用例 Faithfulness 由 10.53% 提升至 84.62%。结果限定为代表性回归集，不作为生产准确率。

