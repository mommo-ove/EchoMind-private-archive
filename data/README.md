# EchoMind 校园演示数据

本目录只提交可公开、可复现的合成数据，不保存真实学生信息、运行日志、Redis 数据或生产数据库。

## 数据内容

- `knowledge/campus_knowledge.json`：20 篇校园网、校园卡、工单和安全规范文档，用于 ChromaDB RAG。
- `eval/campus_golden.json`：23 条意图用例和 20 条端到端对话用例。
- SQLite 演示数据：由 `CampusStore` 和 `tools/seed_campus_demo.py` 写入 Docker `campus_data` 数据卷。

## 使用顺序

先从本 Worktree 重建并启动服务：

```powershell
docker compose up -d --build
docker compose ps
```

导入校园知识文档：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\import-campus-knowledge.ps1
```

向容器内 SQLite 写入可重复执行的 OPEN、PROCESSING、RESOLVED 工单：

```powershell
docker compose exec echomind python tools/seed_campus_demo.py
```

运行校园黄金集：

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run-campus-eval.ps1
```

`/eval/run` 当前仍缺少生产级管理员鉴权、评测数据隔离和请求限额，因此只应在本地演示环境执行。评测会调用真实模型并产生 Token 成本。

意图用例兼容两种标注：

- `expected_intent`：主意图，保留用于主意图 Accuracy。
- `expected_intents`：完整标签集合，用于多标签 Subset Accuracy 和 Macro-F1。

复合问题可以同时提供两者，例如：

```json
{
  "message": "校园网一直报401而且校园卡重复扣款",
  "expected_intent": "technical",
  "expected_intents": ["technical", "billing"]
}
```

评测报告只有在主意图 Accuracy 和多标签 Macro-F1 都达到 `0.75` 时，才把意图识别项判为通过。真实简历指标必须来自固定版本模型、固定评测集的实际运行报告，不能使用单元测试中的构造分数。

## 演示身份

- `demo_user_01`：两笔金额相同的模拟校园卡消费；拥有开放和处理中工单。
- `demo_user_02`：一笔普通消费；拥有已解决工单。
- `demo_user_03`：无校园卡消费记录，用于空结果场景。

所有商户、交易、网络状态和工单均为 synthetic demo data，不代表任何学校真实系统。
# Intent-axis evaluation

The current intent benchmark is `eval/intent_axes_golden.json`. It separates
business domains, user actions, and human-escalation state. See
`../docs/意图三轴重构与评测.md` before interpreting its metrics.
