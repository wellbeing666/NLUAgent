"""应用侧有限语法授权；不信任Planner声称的权限或改写的用户原句。"""

import re

from cockpit.nlu import parse_intent, normalize_text
from cockpit.safety import WRITE_TOOLS


def split_clauses(text):
    connector = r'[,，;；、]|并且|同时|然后|接着|顺便|以及|还有(?=打开|关闭|把|将|查询|查看|设置|调整|导航|空调|座椅|雨刮|前照灯)|并|和(?=打开|关闭|把|将|查询|查看|设置|调整|导航|空调|座椅|雨刮|前照灯)|再(?=打开|关闭|把|将|查询|查看|设置|调整|导航)'
    return [re.sub(r'^先', '', part.strip(' 。！!')) for part in re.split(connector, text) if part.strip(' 。！!')]


def write_grants(text):
    normalized = normalize_text(text)
    # 条件/否定的作用域可能跨分句，第一版保守地不授予这类混合表达的写能力。
    unsafe = r'[?？“”‘’「」"\']|吗|能否|能不能|可不可以|如果|假如|假设|要是|倘若|除非|只有|当.*(?:时|后)|不要|别|禁止|取消|停止|不想|不需要|不用|停车后|稍后|等会|明天|定时|他说|她说|翻译|解释|复述|假装'
    if re.search(unsafe, normalized):
        return []
    grants = []
    for clause in split_clauses(text):
        intent = parse_intent(clause)
        if intent and intent.tool in WRITE_TOOLS:
            grants.append({'source': clause, 'tool': intent.tool, 'arguments': intent.arguments, 'used': False})
    return grants
