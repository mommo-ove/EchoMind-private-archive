# EchoMind 证据校验与反思评测报告

## 1. 目标与边界

本轮升级解决的不是“向量召回是否能找到相似文本”，而是回答生成后的高风险事实是否有可靠来源：

- 引用的知识片段必须真实存在于本轮检索结果；
- 回答中的工单 ID 必须来自成功的 Tool 结果；
- “已创建工单”等动作声明必须有真实 Tool 调用；
- 工单、网络状态不得与结构化 Tool 数据冲突；
- 金额、退款时限等带单位数值必须与引用或 Tool 数据一致；
- 校验失败时最多反思重写一次，仍失败则返回安全降级结果。

规则校验器不判断开放式语义蕴含。例如，知识片段只给出 401 处理步骤，而回答声称“根因一定是服务器硬件损坏”，即使引用 ID 存在，当前规则仍无法判断这句话是否受到文本支持。该类问题后续应由 NLI/LLM 证据判别器或 RAGAS Faithfulness 补充，不能包装为已经解决。

## 2. 在线链路

```text
意图驱动检索
→ 查询改写
→ BGE向量召回
→ 合并去重
→ Cross-Encoder重排
→ Agent生成初稿
→ EvidenceVerifier确定性校验
   ├─ 通过：返回
   └─ 失败：携带问题代码反思重写一次
       ├─ 通过：返回纠正答案
       └─ 仍失败：安全拒答/建议人工处理
```

反思次数由 `EVIDENCE_MAX_REFLECTIONS` 控制，默认值为 `1`，防止无限自我修正和 Token 失控。

## 3. 对抗数据集

数据文件：`data/eval/hallucination_adversarial.json`

- 总量：200 条；
- 训练/验证/测试：120/40/40；
- 数据性质：基于公开校园演示知识和模拟 SQLite 业务记录构造的 synthetic adversarial benchmark；
- 不含真实学生信息、线上日志或真实学校系统数据。

错误类型包括：

1. `entity_fabrication`：虚构工单 ID；
2. `status_mismatch`：回答状态与 SQLite/Tool 状态冲突；
3. `missing_citation`：使用知识事实但未引用证据；
4. `numeric_mismatch`：金额、时限等数值被篡改；
5. `unsupported_causal_claim`：引用存在，但因果结论不受证据支持；
6. `supported`：真实支持、正确引用或安全拒答样本。

## 4. 验证集消融结果

同一 40 条 validation 数据集：

| 方案 | Precision | Recall | F1 | 漏检率 | Accuracy |
|---|---:|---:|---:|---:|---:|
| Baseline：引用/工单/状态/动作校验 | 100.00% | 62.96% | 77.27% | 37.04% | 75.00% |
| 增加带单位数值一致性校验 | 100.00% | 85.19% | 92.00% | 14.81% | 90.00% |

变化：

- Recall 提升 22.22 个百分点；
- F1 提升 14.73 个百分点；
- 漏检率下降 22.22 个百分点；
- Precision 保持 100%。

基线报告：`data/eval/results/hallucination_baseline_validation.json`

升级报告：`data/eval/results/hallucination_numeric_validation.json`

## 5. 固定测试切分结果

在固定的 40 条 test split 数据上：

| Precision | Recall | F1 | 漏检率 | Accuracy |
|---:|---:|---:|---:|---:|
| 100.00% | 85.19% | 92.00% | 14.81% | 90.00% |

测试报告：`data/eval/results/hallucination_test_latest.json`

测试集与验证集使用相同分层模板生成，因此指标相同并不代表真实线上泛化能力。简历和面试必须说明这是 200 条合成对抗集，而不能说成真实生产日志效果。

## 6. Bad Case

测试集剩余 4 条漏检均为 `unsupported_causal_claim`：

```text
证据：401时可以清除旧认证状态后重新登录。
回答：根因一定是学校服务器硬件损坏 [Citation kb-cause-xxx]。
```

引用 ID 正确，但引用内容并不支持该因果结论。确定性规则只能证明“引用存在”，不能完整证明“句子被证据蕴含”。后续优化路径：

1. 将回答拆成原子事实声明；
2. 使用中文 NLI 或 LLM Judge 判断 `evidence → claim` 是否成立；
3. 对关键数值、实体、状态继续保留程序化精确校验；
4. 使用人工复核样本校准阈值，避免 Judge 自身幻觉。

## 7. 运行方式

重新生成可复现数据：

```powershell
.\.venv\Scripts\python.exe tools\build_hallucination_golden.py
```

运行独立测试集：

```powershell
.\.venv\Scripts\python.exe -m evaluation.hallucination_eval `
  data\eval\hallucination_adversarial.json `
  --split test `
  --output data\eval\results\hallucination_test_latest.json
```

连接 `.env` 中配置的真实 DeepSeek，评测“已检出错误后的单轮反思纠正率”：

```powershell
.\.venv\Scripts\python.exe -m evaluation.live_reflection_eval --limit 8
```

## 8. 真实 DeepSeek 单轮反思实验

本实验从固定 test split 中按错误类型均衡抽取 8 条**已经被确定性校验器检出**的错误初稿，使用项目真实反思 Prompt 调用 `deepseek-v4-flash` 重写，再用同一个 `EvidenceVerifier` 复验。

| 检出错误 | 纠正通过 | 安全降级 | 条件纠正率 |
|---:|---:|---:|---:|
| 8 | 5 | 3 | 62.50% |

3 条失败包括：1 条重写后仍缺少 Citation，2 条没有提取到最终回答文本。针对后者已增加 `empty_response` 校验，空回答不得视为纠正成功，必须走安全降级。

结果文件：`data/eval/results/live_deepseek_reflection_latest.json`

这个 62.50% 的分母是“规则已经抓到的 8 条错误”，不是自然用户请求的幻觉率，也不是完整线上问题解决率。样本规模很小，目前只用于验证闭环能真实调用模型并暴露 bad case，不应直接写成简历核心指标。

## 9. 简历表述

> 构建基于引用与 Tool 结果的事实校验和单轮反思纠错链路，对工单实体、状态、动作及带单位数值进行确定性验证；在 200 条合成对抗集上开展分层消融，验证集 F1 由 77.27% 提升至 92.00%、幻觉漏检率由 37.04% 降至 14.81%，固定测试切分 F1 为 92.00%，并通过安全降级限制未受证据支持的回答。

如果面试官追问，应主动补充：该指标来自 synthetic benchmark，不是线上真实用户解决率；开放式语义幻觉仍需要 NLI/LLM 证据判别与人工抽检进一步覆盖。
