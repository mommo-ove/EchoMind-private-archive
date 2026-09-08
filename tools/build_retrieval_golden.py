"""Build EchoMind's token-free retrieval benchmark from curated questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SINGLE_QUESTIONS: Mapping[str, Sequence[str]] = {
    "校园网401认证失败处理": (
        "校园网登录一直提示401该怎么处理？",
        "认证页说登录状态失效，我应该先清理什么？",
        "宿舍WiFi能连上但认证报401，能帮我排查吗？",
        "换了设备之后校网401，是账号密码错了吗？",
        "校园网又401了，我退出认证再重连还是不行。",
    ),
    "校园网403权限异常处理": (
        "校园网认证后出现403是什么意思？",
        "账号能识别但提示没有访问权限怎么办？",
        "连接访客网后403，是不是网络选错了？",
        "设备数量超限会导致校园网403吗？",
        "校网显示403，反复改密码有用吗？",
    ),
    "校园网DNS与代理排查": (
        "WiFi连上了但所有网页都打不开怎么排查？",
        "开着VPN时校园网访问不了网页要怎么办？",
        "只有一个网站打不开，是DNS坏了吗？",
        "浏览器代理扩展会影响校园网访问吗？",
        "校园网能连接但域名都解析失败咋整？",
    ),
    "宿舍区域断网排查": (
        "我们整层宿舍都断网了应该提供哪些信息？",
        "怎么判断断网只影响我还是整栋楼？",
        "同寝室几个人同时没网，先查设备还是服务状态？",
        "宿舍楼大面积断网需要直接建工单吗？",
        "三号宿舍楼从昨晚开始没网，该怎样报修？",
    ),
    "校园网速度慢与频繁掉线": (
        "校园网特别慢还总掉线要记录哪些指标？",
        "晚上网速变慢，怎么判断是不是高峰拥堵？",
        "只有我的电脑网速慢，应该先检查什么？",
        "同一区域很多人都卡顿，要怎样反馈给维护人员？",
        "校网延迟高丢包严重，报障时要带什么数据？",
    ),
    "新设备接入校园网": (
        "新买的电脑第一次怎么连校园网？",
        "新手机连接WiFi后不弹认证页怎么办？",
        "可以借同学账号给我的新设备认证吗？",
        "物联网设备接校园网要遵守什么规则？",
        "首次接入时怎样确认认证页面是官方的？",
    ),
    "校园网维护窗口说明": (
        "校园网维护期间认证失败正常吗？",
        "怎么确认现在是不是学校计划维护？",
        "公告结束后网络还没恢复应该怎么办？",
        "没有实时状态时能直接说校园网正在维护吗？",
        "维护窗口可能影响哪些校园网络服务？",
    ),
    "校园卡重复扣款核验": (
        "食堂好像重复扣了两次款，怎么核验？",
        "两笔金额相同就一定是重复扣款吗？",
        "校园卡重复消费需要比较哪些交易字段？",
        "发现疑似重复扣款后可以直接承诺退款吗？",
        "同一分钟被扣两笔钱，我该走什么处理流程？",
    ),
    "校园卡消费记录查询": (
        "怎么查我最近七天的校园卡消费记录？",
        "消费明细应该显示哪些字段？",
        "没查到校园卡交易时系统应该怎么回答？",
        "能不能帮我查另一个同学的校园卡流水？",
        "校园卡账单查询为什么必须限制为当前用户？",
    ),
    "校园卡充值未到账": (
        "校园卡充值成功但余额没变怎么办？",
        "充值未到账需要准备哪些交易信息？",
        "支付平台已经扣款，学校卡里还没有钱怎么处理？",
        "充值异常时客服可以索要短信验证码吗？",
        "怎样区分充值处理中、成功未入账和支付失败？",
    ),
    "校园卡挂失与补办": (
        "校园卡丢了应该先挂失还是先补办？",
        "Agent能直接替我把校园卡挂失吗？",
        "补办校园卡一般需要做身份核验吗？",
        "为了补卡可以在聊天里发送身份证照片吗？",
        "校园卡遗失后怎样降低被冒用风险？",
    ),
    "校园卡退款与冲正说明": (
        "校园卡异常扣款的退款和冲正有什么流程？",
        "核验前能保证退款多久到账吗？",
        "扣款撤销、冲正和余额返还有什么区别？",
        "申请校园卡退款需要保留哪些交易凭证？",
        "异常消费处理为什么还要核对商户结算状态？",
    ),
    "创建校园服务工单": (
        "创建校园服务工单前需要确认哪些内容？",
        "报修描述怎样写才方便技术人员复现？",
        "用户没明确同意时可以自动创建工单吗？",
        "重复点击提交为什么不应该生成两张工单？",
        "工单描述里能不能附上密码和完整Token？",
    ),
    "工单状态与流转": (
        "OPEN、PROCESSING和RESOLVED分别表示什么？",
        "校园工单允许按照什么顺序更新状态？",
        "已解决的工单能直接退回OPEN吗？",
        "谁有权限把工单改成处理中？",
        "工单现在显示PROCESSING说明到哪一步了？",
    ),
    "工单查询与用户隔离": (
        "为什么查工单必须使用后端认证得到的用户ID？",
        "知道别人的工单编号就能查看内容吗？",
        "查不到工单时为什么统一返回不存在或无权限？",
        "LLM参数里传入的user_id可以直接信任吗？",
        "系统如何防止用户越权查看他人的报修单？",
    ),
    "人工升级与紧急事件": (
        "哪些情况需要标记escalated=true？",
        "标记需要人工介入就等于已经接通客服了吗？",
        "用户强烈要求转人工时系统应该怎么处理？",
        "遇到CRITICAL紧急事件可以创建高优先级工单吗？",
        "演示环境的人工升级和生产环境有什么差别？",
    ),
    "校园服务隐私与凭据安全": (
        "校园服务Agent绝对不能索取哪些敏感信息？",
        "用户身份为什么必须来自后端可信上下文？",
        "日志和评测数据里应该怎样保护个人隐私？",
        "可以把完整银行卡号和验证码发给客服核验吗？",
        "演示系统能不能导入真实学生的交易流水？",
    ),
    "账号锁定与密码重置": (
        "校园账号连续登录失败被锁了怎么办？",
        "客服可以帮我找回旧密码吗？",
        "密码重置后仍无法登录应该记录什么信息？",
        "账号锁定时可以把验证码发给Agent处理吗？",
        "怎么通过学校官方渠道重置校园网密码？",
    ),
    "投诉与服务反馈处理": (
        "我想投诉校园服务，系统应该先问什么？",
        "处理投诉时能在没有核验前承诺赔偿吗？",
        "一般服务反馈应该由哪个Agent说明？",
        "投诉涉及持续网络故障时要不要转专业Agent？",
        "用户坚持要求真人处理投诉时该怎么办？",
    ),
    "技术与账单复合问题处理": (
        "校园网登录失败还被重复扣款，应该由谁处理？",
        "一句话同时包含技术和账单问题时能并行处理吗？",
        "TechnicalAgent和BillingAgent的结果要怎样合并？",
        "又连不上网又有异常消费，是否要识别两个意图？",
        "复合问题里还要求马上转人工时哪个流程优先？",
    ),
}


COMPOSITE_QUESTIONS: Sequence[Tuple[str, Sequence[str]]] = (
    ("宿舍整层断网且正值维护时间，我该先看公告还是直接报修？", ("宿舍区域断网排查", "校园网维护窗口说明")),
    ("校园网401反复出现并且账号可能被锁，应该怎样处理？", ("校园网401认证失败处理", "账号锁定与密码重置")),
    ("新电脑不弹认证页而且浏览器开着代理，如何排查？", ("新设备接入校园网", "校园网DNS与代理排查")),
    ("整栋楼网络很慢又频繁掉线，报障要准备哪些证据？", ("宿舍区域断网排查", "校园网速度慢与频繁掉线", "创建校园服务工单")),
    ("校园网403又想查看当前服务状态，应分别怎么处理？", ("校园网403权限异常处理", "校园网维护窗口说明")),
    ("校园卡充值未到账后如何创建一张不重复的账单工单？", ("校园卡充值未到账", "创建校园服务工单")),
    ("发现重复扣款后应该怎样核验并申请冲正？", ("校园卡重复扣款核验", "校园卡退款与冲正说明")),
    ("校园卡丢失后发现陌生消费，先做什么并怎样查流水？", ("校园卡挂失与补办", "校园卡消费记录查询")),
    ("查校园卡账单时如何确保不会看到别人的交易？", ("校园卡消费记录查询", "校园服务隐私与凭据安全")),
    ("退款工单从OPEN到解决要经过哪些状态？", ("校园卡退款与冲正说明", "工单状态与流转")),
    ("用户知道别人的工单号还要求查询，系统应如何鉴权和回复？", ("工单查询与用户隔离", "校园服务隐私与凭据安全")),
    ("创建技术工单时如何避免写入密码，并避免重复提交？", ("创建校园服务工单", "校园服务隐私与凭据安全")),
    ("网络故障持续且用户强烈要求真人，应该怎样升级并保留工单？", ("人工升级与紧急事件", "创建校园服务工单", "投诉与服务反馈处理")),
    ("投诉校园卡重复扣款时，系统该如何记录问题又不能提前承诺退款？", ("投诉与服务反馈处理", "校园卡重复扣款核验", "校园卡退款与冲正说明")),
    ("校园网连不上同时校园卡充值也没到账，系统应该如何拆分处理？", ("技术与账单复合问题处理", "校园卡充值未到账", "校园网401认证失败处理")),
    ("账号锁定导致认证失败，重置后仍然401要如何继续排查？", ("账号锁定与密码重置", "校园网401认证失败处理")),
    ("维护结束后宿舍仍断网且多人受影响，什么时候该创建技术工单？", ("校园网维护窗口说明", "宿舍区域断网排查", "创建校园服务工单")),
    ("工单状态由谁更新，普通用户又只能查看哪些工单？", ("工单状态与流转", "工单查询与用户隔离")),
    ("新设备接入校园网时怎样避免进入钓鱼认证页并保护凭据？", ("新设备接入校园网", "校园服务隐私与凭据安全")),
    ("既有403权限问题又要求投诉转人工，路由和升级顺序是什么？", ("校园网403权限异常处理", "投诉与服务反馈处理", "人工升级与紧急事件")),
)


UNANSWERABLE_QUESTIONS: Sequence[str] = (
    "学校图书馆周末几点关门？",
    "今天二食堂的午餐菜单是什么？",
    "本学期高等数学考试具体是哪一天？",
    "帮我查询明天第一节课在哪间教室。",
    "学校游泳馆办年卡多少钱？",
    "研究生奖学金什么时候发放？",
    "能帮我预约校医院牙科门诊吗？",
    "今年校园招聘会有哪些公司参加？",
    "宿舍空调电费每度多少钱？",
    "快递站今晚几点停止取件？",
    "请告诉我辅导员的私人手机号。",
    "帮我修改本学期选修课。",
    "校园巴士下一班还有几分钟到？",
    "能查询我的英语四级成绩吗？",
    "学校附近哪家火锅店最好吃？",
    "替我申请明天的宿舍晚归。",
    "实验室打印机还剩多少张纸？",
    "帮我开一张在读证明电子版。",
    "这周末学校会不会下雨？",
    "帮我给室友发送一条微信消息。",
)


DIFFICULTIES = ("direct", "paraphrase", "colloquial", "noisy", "boundary")


def _context_id(title: str, content: str) -> str:
    return hashlib.md5(f"{title}_0_{content[:50]}".encode()).hexdigest()


def build_payload(documents: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    title_to_id = {
        item["title"]: _context_id(item["title"], item["content"])
        for item in documents
    }
    cases: List[Dict[str, Any]] = []
    for doc_index, (title, questions) in enumerate(SINGLE_QUESTIONS.items(), start=1):
        for variant_index, question in enumerate(questions, start=1):
            cases.append({
                "case_id": f"single-{doc_index:02d}-{variant_index}",
                "question": question,
                "category": "single_document",
                "difficulty": DIFFICULTIES[variant_index - 1],
                "answerable": True,
                "reference_context_relevance": {title_to_id[title]: 3.0},
                "reference_titles": [title],
            })
    for index, (question, titles) in enumerate(COMPOSITE_QUESTIONS, start=1):
        cases.append({
            "case_id": f"composite-{index:02d}",
            "question": question,
            "category": "composite",
            "difficulty": "multi_evidence",
            "answerable": True,
            "reference_context_relevance": {
                title_to_id[title]: 3.0 for title in titles
            },
            "reference_titles": list(titles),
        })
    for index, question in enumerate(UNANSWERABLE_QUESTIONS, start=1):
        cases.append({
            "case_id": f"unanswerable-{index:02d}",
            "question": question,
            "category": "unanswerable",
            "difficulty": "out_of_scope",
            "answerable": False,
            "reference_context_relevance": {},
            "reference_titles": [],
        })

    payload = {
        "schema_version": 1,
        "description": "EchoMind校园知识库检索评测集；不调用LLM，避免与端到端生成评测混用。",
        "corpus_document_count": len(documents),
        "case_count": len(cases),
        "corpus": [
            {"context_id": title_to_id[item["title"]], "title": item["title"]}
            for item in documents
        ],
        "cases": cases,
    }
    validate_payload(payload, documents)
    return payload


def validate_payload(payload: Mapping[str, Any], documents: Sequence[Mapping[str, str]]) -> None:
    cases = list(payload["cases"])
    if len(cases) != 140 or payload["case_count"] != 140:
        raise ValueError("retrieval benchmark must contain exactly 140 cases")
    if len({case["case_id"] for case in cases}) != len(cases):
        raise ValueError("case_id values must be unique")
    if len({case["question"] for case in cases}) != len(cases):
        raise ValueError("questions must be unique")
    expected_titles = {item["title"] for item in documents}
    if set(SINGLE_QUESTIONS) != expected_titles:
        raise ValueError("single-document questions must cover the complete corpus")
    context_ids = {item["context_id"] for item in payload["corpus"]}
    for case in cases:
        relevance = case["reference_context_relevance"]
        if not set(relevance).issubset(context_ids):
            raise ValueError(f"unknown context ID in {case['case_id']}")
        if case["answerable"] != bool(relevance):
            raise ValueError(f"answerability mismatch in {case['case_id']}")


def write_payload(payload: Mapping[str, Any], output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--knowledge",
        type=Path,
        default=Path("data/knowledge/campus_knowledge.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/eval/campus_retrieval_golden.json"),
    )
    args = parser.parse_args()
    documents = json.loads(args.knowledge.read_text(encoding="utf-8"))
    print(write_payload(build_payload(documents), args.output))


if __name__ == "__main__":
    main()
