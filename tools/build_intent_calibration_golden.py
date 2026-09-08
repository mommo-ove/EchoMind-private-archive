"""Build the fixed 70/30 intent calibration and held-out test dataset."""

from __future__ import annotations

import json
from pathlib import Path


SINGLE = {
    "greeting": [
        "你好，我想咨询校园服务", "早上好，有人在吗", "嗨，能帮我一下吗", "晚上好",
        "hello校园助手", "你好呀", "在吗，想问个事", "下午好同学",
    ],
    "technical": [
        "校园网登录一直提示401", "宿舍WiFi连不上", "网页打不开还总掉线", "校园网报403错误",
        "网络延迟高而且丢包", "认证页面不自动弹出", "开着代理后上不了网", "整栋宿舍突然断网了",
    ],
    "billing": [
        "食堂好像重复扣款了", "校园卡充值一直没到账", "这笔异常消费能退款吗", "想查最近七天校园卡流水",
        "退款什么时候能到账", "校园卡余额无故少了", "支付成功但卡里没钱", "同一分钟被扣了两笔费用",
    ],
    "escalation": [
        "马上给我转人工客服", "这个问题很严重我要找负责人", "别让机器人处理，请接真人", "紧急情况立刻升级处理",
        "我要联系值班主管", "现在就转人工", "情况非常严重请立刻通知老师", "必须找真人客服解决",
    ],
    "account": [
        "校园账号被锁定了", "我忘记校园网密码了", "密码重置后还是登录不了", "账号提示已停用",
        "怎么修改登录密码", "我的账户无法认证", "官方密码重置入口在哪里", "账号设备数量超限了",
    ],
    "complaint": [
        "我要投诉校园网服务", "这个处理效率太差了", "问题拖了一周都没人解决", "客服态度让我很不满意",
        "校园卡问题一直推诿", "我要反馈严重的服务问题", "报修几次都没结果", "这个服务体验实在糟糕",
    ],
    "request": [
        "请帮我创建网络报修工单", "帮我查询这张工单", "请记录我的校园卡问题", "麻烦帮我提交故障信息",
        "帮我检查校园网状态", "请创建一个账单工单", "能否帮我办理挂失", "帮我把工单更新为处理中",
    ],
    "query": [
        "工单OPEN是什么意思", "校园网维护到几点结束", "退款流程是什么", "校园卡挂失需要哪些材料",
        "工单状态有哪些", "为什么认证会出现403", "如何判断是不是重复扣款", "现在的服务状态怎么样",
    ],
    "feedback": [
        "建议增加夜间客服", "希望回答步骤再清楚一点", "这个功能挺好用的", "建议支持语音输入",
        "希望工单能主动提醒", "我对这次处理结果满意", "页面操作可以再简单些", "建议增加校园卡余额提醒",
    ],
    "other": [
        "今天天气怎么样", "给我讲个笑话", "你会写诗吗", "推荐一部电影",
        "一加一等于多少", "帮我翻译一句英语", "附近有什么好吃的", "你觉得人工智能是什么",
    ],
}


MULTI = [
    ("校园网401而且食堂重复扣款", "technical", ["technical", "billing"]),
    ("网断了还被扣了两次钱", "technical", ["technical", "billing"]),
    ("账号锁了，马上给我转人工", "escalation", ["account", "escalation"]),
    ("校园网坏了一周，我要投诉", "complaint", ["technical", "complaint"]),
    ("充值没到账，请帮我创建工单", "request", ["billing", "request"]),
    ("查一下异常消费，不行就转人工", "escalation", ["billing", "escalation"]),
    ("网络频繁掉线，帮我提交报修", "request", ["technical", "request"]),
    ("账号403而且我要找负责人", "escalation", ["account", "technical", "escalation"]),
    ("退款流程是什么，我还要投诉处理太慢", "complaint", ["query", "billing", "complaint"]),
    ("建议增加网络状态查询功能", "feedback", ["feedback", "technical"]),
    ("校园卡丢了还发现陌生消费", "billing", ["account", "billing"]),
    ("宿舍断网，情况紧急请转人工", "escalation", ["technical", "escalation"]),
    ("帮我查工单并告诉我OPEN的含义", "request", ["request", "query"]),
    ("密码忘了，请帮我提交账号工单", "request", ["account", "request"]),
    ("重复扣款一直没解决，我很不满意", "complaint", ["billing", "complaint"]),
    ("你好，我想查询校园网维护时间", "query", ["greeting", "query", "technical"]),
    ("网络慢，建议增加实时状态提醒", "feedback", ["technical", "feedback"]),
    ("退款没到，立刻给我接真人", "escalation", ["billing", "escalation"]),
    ("账号被锁并且认证一直报401", "account", ["account", "technical"]),
    ("帮我创建重复扣款工单，问题很紧急", "request", ["billing", "request", "escalation"]),
]


def build_payload():
    cases = []
    index = 1
    for label, messages in SINGLE.items():
        for position, message in enumerate(messages):
            cases.append({
                "case_id": f"intent-{index:03d}",
                "split": "validation" if position < 6 else "test",
                "message": message,
                "expected_intent": label,
                "expected_intents": [label],
            })
            index += 1
    for position, (message, primary, labels) in enumerate(MULTI):
        cases.append({
            "case_id": f"intent-{index:03d}",
            "split": "validation" if position < 10 else "test",
            "message": message,
            "expected_intent": primary,
            "expected_intents": labels,
        })
        index += 1
    return {
        "schema_version": 1,
        "description": "Intent calibration uses validation only; held-out test is evaluated once after selection.",
        "case_count": len(cases),
        "cases": cases,
    }


def main():
    output = Path("data/eval/intent_calibration_golden.json")
    output.write_text(json.dumps(build_payload(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
