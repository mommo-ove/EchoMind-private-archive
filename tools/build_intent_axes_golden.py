"""Build a three-axis intent dataset with a legacy held-out challenge split."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.intent_axes import validate_case
from tools.build_intent_calibration_golden import build_payload as build_legacy_payload


DOMAIN_TOPICS = {
    "technical": [
        "校园网认证失败", "宿舍WiFi频繁掉线", "认证页提示401", "网页提示403",
        "校园网延迟很高", "宿舍突然断网", "认证页面打不开", "连接后无法访问网页",
        "网络一直丢包", "代理关闭后仍不能联网", "无线网找不到", "认证状态反复失效",
        "教学楼网络很慢", "客户端提示连接超时", "校园网服务显示异常",
    ],
    "billing": [
        "校园卡重复扣款", "充值一直没到账", "余额无故减少", "食堂出现陌生消费",
        "退款迟迟没到账", "同一分钟扣了两笔", "支付成功但余额不对", "消费流水有异常",
        "挂失后仍然扣款", "退款审核状态不明", "充值记录查不到", "校园卡账单金额不对",
        "缴费订单重复支付", "退款金额少了", "消费记录时间不对",
    ],
    "account": [
        "校园账号被锁定", "忘记登录密码", "密码重置后仍不能登录", "账号提示停用",
        "认证设备数量超限", "绑定手机号需要修改", "账号出现陌生设备", "邮箱信息填错了",
        "统一认证无法通过", "账号资料需要更新", "密码重置入口找不到", "账号被误冻结",
        "登录验证码收不到", "学号绑定失败", "账号权限突然消失",
    ],
    "general": [
        "工单状态含义", "校园服务办理时间", "信息中心值班时间", "报修流程",
        "校园助手使用方式", "服务大厅地址", "工单处理进度", "常见问题入口",
        "人工服务时间", "问题反馈渠道", "报修需要的材料", "工单状态更新规则",
        "校园服务联系方式", "服务评价方式", "问题处理时限",
    ],
}

ACTION_TEMPLATES = {
    "query": ["我想了解{topic}", "请问{topic}怎么查询", "能说明一下{topic}吗"],
    "request": ["请帮我处理{topic}", "麻烦帮我提交{topic}的工单", "请协助我解决{topic}"],
    "report": ["{topic}", "我遇到了{topic}", "现在出现了{topic}"],
    "complaint": ["{topic}一直没解决，我很不满意", "{topic}拖了很久，我要投诉", "{topic}的处理体验太差了"],
    "feedback": ["建议优化{topic}", "希望增加{topic}的提醒功能", "我对{topic}有一条改进建议"],
}

GREETINGS = [
    "你好，校园助手", "早上好", "下午好，校园助手", "晚上好，请问在线吗", "hello，在吗", "嗨，能聊聊吗",
    "你好呀，校园助手", "有人在线吗", "哈喽校园助手", "早安", "午安", "晚上好呀",
    "你好，第一次使用", "嗨嗨", "请问有人吗", "在吗", "您好", "你好同学", "hello", "早上好呀",
]

OTHER = [
    "上海明天天气如何", "讲一个程序员笑话", "写一首关于秋天的诗", "推荐一部科幻电影", "计算二十三乘以七",
    "把早上好翻译成法语", "推荐学校附近的餐馆", "解释生成式人工智能", "推荐一首歌", "讲讲宇宙",
    "帮我写一副春联", "明天会下雨吗", "世界杯谁赢了", "给猫起个名字", "推荐一本小说",
    "怎么学习吉他", "解释一下量子力学", "做一道数学题", "帮我规划旅游", "今天是什么节日",
]

ESCALATION_CASES = [
    ("宿舍断网，请马上给我转人工", ["technical"], "request"),
    ("校园网401一直没解决，我要找负责人", ["technical"], "complaint"),
    ("重复扣款了，立刻接人工客服", ["billing"], "complaint"),
    ("退款一直没到，请找真人处理", ["billing"], "request"),
    ("账号被盗，请立即升级处理", ["account"], "request"),
    ("账号锁定又有陌生扣款，马上联系负责人", ["account", "billing"], "complaint"),
    ("校园网断了还重复扣费，给我转人工", ["technical", "billing"], "complaint"),
    ("别让机器人回答，请接真人客服", ["general"], "request"),
    ("这个问题拖太久了，我要联系值班主管", ["general"], "complaint"),
    ("认证失败影响考试，请立即通知老师", ["technical"], "request"),
    ("校园卡被盗刷，必须找负责人处理", ["billing"], "complaint"),
    ("账号停用影响选课，请马上转人工", ["account"], "request"),
    ("网络故障非常严重，请接人工客服", ["technical"], "complaint"),
    ("退款问题反复出现，我要找真人客服", ["billing"], "complaint"),
    ("账号和网络都无法使用，请升级处理", ["account", "technical"], "request"),
    ("我要人工服务，不要自动回复", ["general"], "request"),
    ("工单一直没人管，请联系负责人", ["general"], "complaint"),
    ("校园卡扣款异常，请马上找真人", ["billing"], "request"),
    ("认证页持续报错，立刻转人工", ["technical"], "request"),
    ("账户资料被改了，请立即联系老师", ["account"], "request"),
]


def _review(rationale: str, *, ambiguous: bool = False) -> dict:
    return {
        "status": "reviewed",
        "reviewer": "codex-assisted",
        "rationale": rationale,
        "requires_human_signoff": ambiguous,
    }


def _validation_cases() -> list[dict]:
    cases = []
    index = 1
    for domain, topics in DOMAIN_TOPICS.items():
        for action, templates in ACTION_TEMPLATES.items():
            for position, topic in enumerate(topics):
                message = templates[position % len(templates)].format(topic=topic)
                cases.append({
                    "case_id": f"axis-val-{index:03d}",
                    "split": "validation",
                    "source": "scenario_matrix",
                    "message": message,
                    "expected": {"domains": [domain], "action": action, "escalated": False},
                    "review": _review(f"{domain}领域的{action}表达"),
                })
                index += 1
    for message in GREETINGS:
        cases.append({
            "case_id": f"axis-val-{index:03d}", "split": "validation", "source": "boundary_bank",
            "message": message,
            "expected": {"domains": ["general"], "action": "greeting", "escalated": False},
            "review": _review("仅寒暄且没有业务诉求"),
        })
        index += 1
    for message in OTHER:
        cases.append({
            "case_id": f"axis-val-{index:03d}", "split": "validation", "source": "out_of_scope_bank",
            "message": message,
            "expected": {"domains": ["general"], "action": "other", "escalated": False},
            "review": _review("校园服务范围外问题"),
        })
        index += 1
    for message, domains, action in ESCALATION_CASES:
        cases.append({
            "case_id": f"axis-val-{index:03d}", "split": "validation", "source": "safety_boundary_bank",
            "message": message,
            "expected": {"domains": domains, "action": action, "escalated": True},
            "review": _review("包含明确人工接管或升级表达"),
        })
        index += 1
    return cases


def _legacy_action(case: dict) -> tuple[str, bool]:
    labels = set(case["expected_intents"])
    primary = case["expected_intent"]
    message = case["message"]
    complaint_markers = ("投诉", "不满意", "太差", "糟糕", "推诿", "没人处理", "没解决")
    feedback_markers = ("建议", "希望", "满意", "好用", "好评")
    query_markers = ("请问", "怎么", "什么", "如何", "哪里", "几点", "什么时候", "想查", "查询")
    request_markers = ("帮我", "请帮", "麻烦", "协助", "创建", "提交", "办理", "更新")
    if primary == "greeting" and not any(marker in message for marker in query_markers + request_markers):
        return "greeting", False
    if primary == "feedback" or any(marker in message for marker in feedback_markers):
        return "feedback", False
    if primary == "complaint" or any(marker in message for marker in complaint_markers):
        return "complaint", False
    if primary == "request" or any(marker in message for marker in request_markers):
        return "request", False
    if primary == "query" or any(marker in message for marker in query_markers) or "?" in message or "？" in message:
        return "query", False
    if primary == "other":
        return "other", False
    return "report", False


def _legacy_domains(case: dict) -> list[str]:
    labels = set(case["expected_intents"])
    message = case["message"]
    domains = [name for name in ("technical", "billing", "account") if name in labels]
    if not domains:
        if any(word in message for word in ("校园网", "WiFi", "网络", "断网", "掉线", "401", "403", "认证页", "故障")):
            domains.append("technical")
        if any(word in message for word in ("校园卡", "扣款", "退款", "充值", "消费", "余额", "账单", "支付")):
            domains.append("billing")
        if any(word in message for word in ("账号", "账户", "密码", "挂失", "设备数量", "登录验证码")):
            domains.append("account")
    return domains or ["general"]


def _test_cases() -> list[dict]:
    transformed = []
    for position, legacy in enumerate(build_legacy_payload()["cases"], start=1):
        labels = set(legacy["expected_intents"])
        domains = _legacy_domains(legacy)
        action, ambiguous = _legacy_action(legacy)
        escalated = any(marker in legacy["message"] for marker in (
            "转人工", "接人工", "真人", "负责人", "主管", "通知老师", "联系老师", "升级处理"
        ))
        transformed.append({
            "case_id": f"axis-test-{position:03d}",
            "split": "test",
            "source": "legacy_challenge",
            "message": legacy["message"],
            "expected": {
                "domains": domains,
                "action": action,
                "escalated": escalated,
            },
            "review": _review(
                "将旧的混合标签拆分为领域、动作和升级状态",
                ambiguous=ambiguous,
            ),
            "legacy": {
                "expected_intent": legacy["expected_intent"],
                "expected_intents": legacy["expected_intents"],
            },
        })
    return transformed


def build_payload() -> dict:
    cases = [*_validation_cases(), *_test_cases()]
    for case in cases:
        validate_case(case)
    if len({case["message"] for case in cases}) != len(cases):
        raise ValueError("dataset contains duplicate messages")
    return {
        "schema_version": 2,
        "description": "Independent domain/action/escalation intent evaluation. Test cases are held-out legacy challenges.",
        "case_count": len(cases),
        "label_schema": {
            "domains": sorted(["general", "technical", "billing", "account"]),
            "actions": sorted(["greeting", "query", "request", "complaint", "feedback", "other"]),
            "escalated": "boolean; true only for explicit human handoff/escalation or reviewed high-risk conditions",
        },
        "cases": cases,
    }


def main() -> None:
    output = Path("data/eval/intent_axes_golden.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(build_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
