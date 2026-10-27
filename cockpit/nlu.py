"""规则 NLU 基线：只匹配完整命令，不对句子里的关键词直接执行车控。"""

import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Intent:
    name: str
    tool: str
    arguments: dict[str, Any]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))


def non_local_reason(text: str) -> str:
    """保守的本地执行边界；咨询可交给Agent，但不能变成车控授权。"""
    text = normalize_text(text)
    patterns = {
        "negated_or_hypothetical": r"不|别|禁止|取消|停止|如果|假如|假设|要是|倘若",
        "quoted_or_reported": r"[\"'“”‘’「」]|他说|她说|听说|有人说|例如|比如|举例|复述|翻译|解释|讨论|想象|假装|研究|学习|说明|分析|比较|含义|后果|影响|为什么",
        "compound_or_trip": r"[,，;；]|然后|顺便|同时|并且|接着|以及|或者|还是|机场|路线|导航|充电|音乐|车窗|座椅|播放|预约",
        "deferred_action": r"明天|等会|稍后|以后|一会|到家|下次|定时|提醒|分钟|小时|点钟",
        "unsupported_domain": r"手机|电脑|笔记本|平板|耳机|轮胎|车胎|胎压|发动机|机油|价格|售价",
    }
    for reason, pattern in patterns.items():
        if re.search(pattern, text):
            return reason
    return ""


_TEMP = re.compile(r"(?P<number>[+-]?\d+(?:\.\d+)?|[负零〇一二两三四五六七八九十百点]+)(?:摄氏度|摄氏|度|°[cC])")
_DIGITS = dict(zip("零〇一二两三四五六七八九", [0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9]))


def parse_number(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        pass
    sign = -1 if value.startswith("负") else 1
    value = value.removeprefix("负")
    integer, *fraction = value.split("点")
    if len(fraction) > 1 or not integer or (fraction and not fraction[0]):
        raise ValueError("无法识别中文温度数字")
    if integer.count("十") > 1 or integer.count("百") > 1 or ("十" in integer and "百" in integer and integer.index("十") < integer.index("百")):
        raise ValueError("无法识别中文温度数字")
    if "十" not in integer and "百" not in integer:
        whole = int("".join(str(_DIGITS[c]) for c in integer))
    else:
        whole, digit = 0, 0
        for char in integer:
            if char in _DIGITS:
                digit = _DIGITS[char]
            elif char in {"十", "百"}:
                whole += (digit or 1) * {"十": 10, "百": 100}[char]
                digit = 0
            else:
                raise ValueError("无法识别中文温度数字")
        whole += digit
    decimal = float("0." + "".join(str(_DIGITS[c]) for c in fraction[0])) if fraction and fraction[0] else 0
    return sign * (whole + decimal)


def extract_temperature(text: str) -> list[float]:
    try:
        return [parse_number(m["number"]) for m in _TEMP.finditer(normalize_text(text))]
    except (ValueError, KeyError):
        return []


def mask_temperature(text: str) -> str:
    return _TEMP.sub("<温度>", normalize_text(text))


def temperature_request_check(text: str) -> tuple[str, float | None, str]:
    """返回 execute/clarify/fallback、槽位和原因；阈值通过不能替代这个检查。"""
    text = normalize_text(text)
    reason = non_local_reason(text)
    if reason:
        return "fallback", None, reason
    if re.search(r"电量|电池|车辆状态|开灯|关灯|打开|关闭|已经|目前|当前|原来|之前|过去|曾经|刚才|上次|设成了|调到了|被设|我喜欢|我偏好", text):
        return "fallback", None, "compound_or_observation"
    if re.search(r"[?？]|吗|么|呢|吧|是否|能否|能不能|可不可以|会不会|怎么样|多少|几度|什么|怎么|如何|应该|建议|会怎样|会如何|省电|华氏|°F", text, re.I):
        return "fallback", None, "question_or_consultation"
    if re.search(r"查|看|显示|读取|读出|报告|报一下|了解|告诉|知道|信息|设定值", text):
        return "fallback", None, "read_request_not_write"
    if re.search(r"再|一点|一些|[升降调][高低](?!到|至)", text):
        return "clarify", None, "relative_temperature_unsupported"
    slots = extract_temperature(text)
    if not slots:
        return "clarify", None, "missing_temperature"
    if "合适" in text:
        return "fallback", None, "question_or_consultation"
    remainder = _TEMP.sub("", text).replace("一下", "")
    if len(slots) != 1 or re.search(r"\d|[零〇一二两三四五六七八九十百]", remainder):
        return "fallback", None, "multiple_or_ambiguous_numbers"
    # 既需要温度上下文，也需要明确的操作表达，防止“今天24度”获得写权限。
    command = re.sub(r"空调|温度|车内|车里|座舱|驾驶室", "", text)
    if not re.search(r"空调|温度|车内|车里|座舱|驾驶室", text) or not re.search(
        r"设|调|改|弄|开到|保持|控制|定在|定到|升到|降到|变成", command
    ):
        return "fallback", None, "not_an_explicit_temperature_command"
    return "execute", slots[0], "validated_absolute_temperature"


def parse_intent(text: str) -> Intent | None:
    # fullmatch 是关键：“不要把空调调到24度”不能因为包含数字就触发写操作。
    original = text
    text = normalize_text(text).rstrip("。！!？?")
    if re.fullmatch(r'(?:请|帮我)?(?:(?:当前|现在|这辆车|汽车|车辆)?(?:电量)?还能(?:行驶|开|跑)(?:多远|多少公里)|(?:计算|估算|查询|查看)(?:一下)?(?:剩余)?续航(?:里程)?)', text):
        return Intent('estimate_driving_range', 'estimate_driving_range', {})
    if re.fullmatch(r'(?:请|帮我)?(?:查看|查询)(?:一下)?(?:历史行驶记录|行驶历史|长期行驶记忆)', text):
        return Intent('get_driving_history', 'get_driving_history', {})
    if re.fullmatch(r"(?:请|帮我)?(?:查看|查询)(?:一下)?车辆状态", text):
        return Intent("query_vehicle", "get_vehicle_state", {})
    control = parse_vehicle_control(original)
    if control is not None:
        return control
    match = re.fullmatch(
        r"(?:请|帮我)?(?:把)?空调(?:温度)?(?:调到|设为|设置为|设置到)"
        r"(-?\d+(?:\.\d+)?)(?:度|℃|°C)", text, re.IGNORECASE,
    )
    if match:
        if temperature_request_check(original)[0] != "execute":
            return None
        # 范围校验由共享工具层完成，规则和模型不各自维护一套规则。
        return Intent("set_temperature", "set_temperature", {"temperature": float(match[1])})
    return None


def parse_vehicle_control(text: str) -> Intent | None:
    """新增车控只做完整规则匹配；不删除问号，也不复用温度向量分数授予写权限。"""
    text = normalize_text(text).rstrip("。！!")
    prefix = r"(?:请|帮我|请帮我)?(?:把|将)?"
    for enabled, verb in [(True, r"(?:打开|开启|开)"), (False, r"(?:关闭|关掉|关)")]:
        for tool, subject in [("set_headlights", r"(?:前照灯|大灯|车灯)"),
                              ("set_wipers", r"(?:雨刮器?|雨刷)")]:
            if re.fullmatch(prefix + rf"(?:{verb}(?:一下)?{subject}|{subject}{verb}(?:一下)?)", text):
                arguments = {"enabled": enabled} if tool == "set_headlights" else {"mode": "low" if enabled else "off"}
                return Intent(tool, tool, arguments)
    match = re.fullmatch(prefix + r"(?:雨刮器?|雨刷)(?:调到|设为|设置为|设置到)(低速|高速|关闭)", text)
    if match:
        return Intent("set_wipers", "set_wipers", {"mode": {"低速": "low", "高速": "high", "关闭": "off"}[match[1]]})
    match = re.fullmatch(prefix + r"(?:驾驶员)?座椅(?:位置)?(?:调到|设为|设置为|设置到)"
                         r"(?:(-?\d+(?:\.\d+)?)%|百分之([负零〇一二两三四五六七八九十百点]+))", text)
    if match:
        try:
            value = float(match[1]) if match[1] is not None else parse_number(match[2])
        except (ValueError, KeyError):
            return None
        return Intent("adjust_seat", "adjust_seat", {"position_percent": value})
    return None
