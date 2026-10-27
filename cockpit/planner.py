"""LLM只生成声明式任务图；通过校验前不执行任何业务工具。"""

import json
import re
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cockpit.tools import build_langchain_tools

MAX_TASKS = 12
TOOL_NAMES = Literal['get_vehicle_state', 'get_vehicle_location', 'set_temperature', 'set_headlights',
                     'adjust_seat', 'set_wipers', 'get_route_options', 'find_chargers',
                     'search_navigation_places', 'plan_navigation_route', 'navigate_to_destination',
                     'estimate_driving_range', 'get_driving_history']
REF_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,31}\.data(?:\.[a-zA-Z0-9_-]+)*$')


class PlanningError(ValueError):
    pass


class Condition(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    ref: str = Field(max_length=200)
    operator: Literal['eq', 'ne', 'lt', 'lte', 'gt', 'gte']
    value: str | int | float | bool | None


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str = Field(pattern=r'^[a-zA-Z][a-zA-Z0-9_-]{0,31}$')
    tool: TOOL_NAMES
    arguments: dict[str, Any]
    depends_on: list[str] = Field(default_factory=list, max_length=MAX_TASKS)
    condition: Condition | None = None


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    tasks: list[Task] = Field(max_length=MAX_TASKS, description='可执行任务列表；只有缺少执行必需信息才为空。')
    clarification: str = Field(default='', max_length=500, description='仅用于缺少必需信息时向用户提问；tasks非空时必须是空字符串。')


def references(value, depth=0):
    if depth > 8:
        raise PlanningError('任务参数嵌套过深。')
    if isinstance(value, dict):
        if '$ref' in value:
            if set(value) != {'$ref'} or not isinstance(value['$ref'], str) or not REF_PATTERN.fullmatch(value['$ref']):
                raise PlanningError('任务结果引用格式不正确。')
            yield value['$ref']
        else:
            for child in value.values():
                yield from references(child, depth + 1)
    elif isinstance(value, list):
        for child in value:
            yield from references(child, depth + 1)


def validate_plan(value, catalog, max_tasks=MAX_TASKS):
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(raw) > 64000:
            raise PlanningError('任务计划过长。')
        plan = TaskPlan.model_validate(value)
    except (ValidationError, TypeError, ValueError, OverflowError, RecursionError):
        # 不回传模型原始响应或验证器中的输入片段。
        raise PlanningError('模型没有返回合法的任务计划；本轮未执行工具，请简化请求后重试。') from None
    if len(plan.tasks) > max_tasks:
        raise PlanningError(f'超过任务数量上限（{max_tasks}项），本轮未执行。')
    if bool(plan.tasks) == bool(plan.clarification.strip()):
        raise PlanningError('计划必须包含任务，或仅包含需要澄清的问题。')
    ids = {task.id for task in plan.tasks}
    if len(ids) != len(plan.tasks):
        raise PlanningError('任务ID不能重复。')
    for task in plan.tasks:
        if task.id in task.depends_on or len(set(task.depends_on)) != len(task.depends_on) or not set(task.depends_on) <= ids:
            raise PlanningError('任务依赖无效。')
        fields = catalog[task.tool].args_schema.model_fields
        required = {name for name, field in fields.items() if field.is_required()}
        if set(task.arguments) - set(fields) or not required <= set(task.arguments):
            raise PlanningError('工具参数缺失或包含不允许的字段，本轮未执行。')
        refs = list(references(task.arguments))
        if task.condition is not None:
            if not REF_PATTERN.fullmatch(task.condition.ref):
                raise PlanningError('条件引用格式不正确。')
            refs.append(task.condition.ref)
        if any(ref.split('.')[0] not in task.depends_on for ref in refs):
            raise PlanningError('结果引用必须声明对应的前置依赖。')
    # Kahn校验：循环或失效任务图一律在任何副作用发生前拒绝。
    done = set()
    while len(done) < len(ids):
        ready = {task.id for task in plan.tasks if task.id not in done and set(task.depends_on) <= done}
        if not ready:
            raise PlanningError('任务依赖存在环，本轮未执行。')
        done |= ready
    return plan


PLANNER_PROMPT = """你是中文智能座舱Planner，只通过submit_plan提交结构化任务图，不执行工具，不报告成功。
单个不明确意图和多个意图都可规划。保留所有明确任务，不擅自增加副作用。
任务字段：id、tool、arguments、depends_on、condition。任务最多12项；缺少目的地/目标温度等关键参数时返回tasks=[]及clarification，不猜测。
tasks非空时clarification必须为""；需要澄清时tasks必须为[]且clarification必须是非空问题；禁止两个都空或两个都非空。
你只负责规划，不负责给出最终建议。机场方案、电量、续航等查询有可用工具时先规划查询；执行后另有回答整理器综合建议，不要因为用户要求解释/建议就省略工具或返回空计划。
依赖任务参数用{"$ref":"t1.data.routes.0.route_id"}引用真实前置结果；引用的任务必须写进depends_on。
condition格式为{"ref":"t1.data.battery_percent","operator":"lt","value":30}；支持eq/ne/lt/lte/gt/gte。
根据实际语义声明依赖：先/然后/再/修改后的查询要等待前序任务；用户明确独立的查询可以无依赖。
同一执行器的多次写入必须按原请求顺序排列并声明依赖。并行任务不应互相读写同一执行器。
车控参数必须来自本轮用户明确命令；否定、疑问、引述、假设和不完整表达不能改成确定车控。只规划用户真正请求的动作，不用聊天历史授权新写入。
第一版写授权只支持原句中可完整规则识别的并列/顺序车控。条件、否定与车控混合的作用域不自动猜测；这类写需求应先澄清。条件读取任务可以正常规划。
应用会逐任务核对原句中的完整车控命令和精确参数，再校验车辆当前安全状态。你不能通过参数声明allow_write、车速、挡位或驻车状态。
set_temperature范围16~30；adjust_seat为0~100%的绝对模拟行程；set_wipers只支持off/low/high。
前照灯、座椅、关闭雨刮需0车速/P挡/已驻车，工具执行时再次检查；开启雨刮可以在行驶中执行。
默认失败策略为独立任务继续、依赖失败跳过，不自动重试写操作，不承诺停车后执行。
只给真实导航目的地时使用navigate_to_destination，它读取用户已设置的模拟当前位置并自动匹配，不猜坐标。默认recommended，可用avoid_congestion/less_fee/no_highway。
手动指定两地点时可分别search_navigation_places，再通过引用传{name,location,id}给plan_navigation_route；不要编造坐标，不随意从模糊同名结果选择地点。
get_route_options和find_chargers只用于模拟机场案例，不混入真实高德导航。机场电量咨询可查询车辆和机场模拟路线，必要时按结果条件查询模拟充电站。
剩余续航、当前电量能跑多远必须调用estimate_driving_range，让代码读取最新SOC、本车持久化容量和历史能耗，不能自行编造车辆容量/额定续航。
get_driving_history可解释人工行驶记忆；只读。写入历史仅由用户通过面板或CLI手动提交，不能把规划路线当成已行驶记录。
续航工具给的是能耗估算，不能保证高德路线实际可达；需要比较路线时可引用实际距离和续航字段，必须说明估算边界。
所有车辆状态和机场耗电是模拟数据；高德路线是真实服务快照，不能编造真实到达电量、已预约、已开车或已启动导航。
只需要进一步解释或信息不足时，用clarification说明需要补充什么，不编造已经完成的动作。
"""


class Planner:
    def __init__(self, service, model, max_tasks=MAX_TASKS):
        self.service = service
        self.max_tasks = max_tasks
        self.catalog = {tool.name: tool for tool in build_langchain_tools(service)}
        schema = {'type': 'function', 'function': {'name': 'submit_plan',
                  'description': '提交经校验后由调度器执行的座舱任务图。', 'parameters': TaskPlan.model_json_schema()}}
        # 只暴露规划提交工具；auto兼容部分开启thinking但不支持强制tool_choice的服务。
        # 返回值仍必须通过下方一次submit_plan/纯JSON校验，不能直接执行业务工具。
        self.model = model.bind_tools([schema], tool_choice='auto')

    def plan(self, text, history):
        self.service.trace('planner.started', {'max_tasks': self.max_tasks})
        catalog = [{'name': tool.name, 'description': tool.description,
                    'parameters': tool.args_schema.model_json_schema()} for tool in self.catalog.values()]
        messages = [SystemMessage(content=PLANNER_PROMPT + '\n可用业务工具：' + json.dumps(catalog, ensure_ascii=False)),
                    *history, HumanMessage(content=text)]
        response = self.model.invoke(messages)
        calls = getattr(response, 'tool_calls', [])
        if len(calls) == 1 and calls[0]['name'] == 'submit_plan':
            value = calls[0]['args']
        elif not calls:
            # 部分兼容服务忽略tool_choice，允许纯JSON，但仍完整校验，绝不执行散文或旧式工具循环。
            content = response.content
            if isinstance(content, list):
                content = '\n'.join(block['text'] for block in content if isinstance(block, dict) and block.get('type') == 'text')
            try:
                if not isinstance(content, str) or len(content) > 64000:
                    raise ValueError()
                content = re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip())
                value = json.loads(content)
            except (ValueError, TypeError, RecursionError):
                raise PlanningError('模型没有提交结构化任务计划；本轮未执行，请重试。') from None
        else:
            raise PlanningError('Planner只能提交一次submit_plan，不能直接调用车控工具。')
        plan = validate_plan(value, self.catalog, self.max_tasks)
        self.service.trace('planner.plan', plan.model_dump())
        return plan
