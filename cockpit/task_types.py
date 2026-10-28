"""业务目标、任务图与模型协议。权限范围由代码确定，不由模型指定。"""

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

MAX_TASKS = 12
REF_PATTERN = re.compile(r'^[a-zA-Z][a-zA-Z0-9_-]{0,31}\.data(?:\.[a-zA-Z0-9_-]+)*$')
KINDS = Literal['vehicle_control','vehicle_state','driving_range','driving_history','navigation','airport_advice']
CONTROL_NAMES = {'set_temperature','set_headlights','adjust_seat','set_wipers'}
KIND_LABELS = {'vehicle_control':'车辆控制','vehicle_state':'查询车辆状态','driving_range':'估算剩余续航',
               'driving_history':'读取行驶记忆','navigation':'导航与地点查询','airport_advice':'模拟机场方案'}


class PlanningError(ValueError):
    pass


class Condition(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    ref: str = Field(max_length=200)
    operator: Literal['eq','ne','lt','lte','gt','gte']
    value: str | int | float | bool | None


class Task(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    id: str = Field(pattern=r'^[a-zA-Z][a-zA-Z0-9_-]{0,31}$')
    kind: KINDS
    goal: str = Field(min_length=1, max_length=500)
    inputs: dict[str, Any]
    depends_on: list[str] = Field(default_factory=list, max_length=MAX_TASKS)
    condition: Condition | None = None


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    tasks: list[Task] = Field(max_length=MAX_TASKS)
    clarification: str = Field(default='', max_length=500, description='tasks非空时必须为空字符串；缺信息时tasks为空且这里为澄清问题。')


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


def allowed_tools(task):
    if task.kind == 'vehicle_control':
        return {task.inputs['tool'], 'get_vehicle_state'}
    if task.kind == 'vehicle_state':
        return {'get_vehicle_location'} if task.inputs.get('view') == 'location' else {'get_vehicle_state'}
    if task.kind == 'driving_range':
        return {'get_vehicle_state','get_driving_history','estimate_driving_range'}
    if task.kind == 'driving_history':
        return {'get_driving_history'}
    if task.kind == 'navigation':
        if task.inputs.get('mode') == 'search': return {'search_navigation_places'}
        return {'get_vehicle_location','search_navigation_places','plan_navigation_route','navigate_to_destination'}
    if task.inputs.get('view') == 'routes': return {'get_vehicle_state','get_route_options'}
    if task.inputs.get('view') == 'chargers': return {'get_route_options','find_chargers'}
    return {'get_vehicle_state','get_route_options','find_chargers','estimate_driving_range','get_driving_history'}


def primary_tools(task):
    if task.kind == 'vehicle_control': return {task.inputs['tool']}
    if task.kind == 'vehicle_state': return allowed_tools(task)
    if task.kind == 'driving_range': return {'estimate_driving_range'}
    if task.kind == 'driving_history': return {'get_driving_history'}
    if task.kind == 'navigation':
        return {'search_navigation_places'} if task.inputs.get('mode') == 'search' else {'navigate_to_destination','plan_navigation_route'}
    return {'find_chargers'} if task.inputs.get('view') == 'chargers' else {'get_route_options'}


def resource(task):
    if task.kind == 'vehicle_control': return 'actuator:' + task.inputs['tool']
    if task.kind == 'navigation' and task.inputs.get('mode') != 'search': return 'navigation'
    return None


def validate_inputs(task, catalog):
    fields = {
        'vehicle_control':({'tool','arguments'}, {'tool','arguments'}),
        'vehicle_state':({'view'}, set()), 'driving_range':(set(), set()),
        'driving_history':({'limit'}, set()),
        'navigation':({'mode','destination','origin','origin_name','city','keywords','strategy'}, set()),
        'airport_advice':({'destination','view','route_id'}, set()),
    }
    allowed, required = fields[task.kind]
    if set(task.inputs) - allowed or not required <= set(task.inputs):
        raise PlanningError('业务任务参数缺失或包含不允许的字段。')
    if task.kind == 'vehicle_control':
        name, args = task.inputs['tool'], task.inputs['arguments']
        if not isinstance(name, str) or name not in CONTROL_NAMES or not isinstance(args, dict):
            raise PlanningError('车控任务必须指定受支持的动作与参数。')
        fields = catalog[name].args_schema.model_fields
        if set(args) != set(fields):
            raise PlanningError('车控目标参数不完整或包含权限字段。')
    elif task.kind == 'vehicle_state' and task.inputs.get('view','state') not in {'state','location'}:
        raise PlanningError('不支持的状态查询类型。')
    elif task.kind == 'navigation':
        mode = task.inputs.get('mode','destination')
        if mode not in {'destination','between_places','search'}:
            raise PlanningError('不支持的导航目标类型。')
        needed = {'keywords','city'} if mode == 'search' else {'destination'}
        if not needed <= set(task.inputs) or (mode == 'between_places' and not {'origin','origin_name'} & set(task.inputs)):
            raise PlanningError('导航缺少目的地、搜索城市或明确起点。')
        if 'destination' in task.inputs:
            target = task.inputs['destination']
            if not isinstance(target,(str,dict)) or (isinstance(target,str) and not target.strip()):
                raise PlanningError('目的地必须为明确名称或前置结果引用。')
        for key in ['origin','destination']:
            value = task.inputs.get(key)
            if isinstance(value,dict) and not list(references(value)):
                raise PlanningError('规划器不能直接提供地点坐标，请引用前置地点查询结果。')
        for key in ['city','keywords','origin_name']:
            if key in task.inputs and (not isinstance(task.inputs[key], str) or not task.inputs[key].strip()):
                raise PlanningError('导航名称或城市参数不正确。')
        if task.inputs.get('strategy','recommended') not in {'recommended','avoid_congestion','less_fee','no_highway'}:
            raise PlanningError('不支持的导航偏好。')
    elif task.kind == 'airport_advice':
        if task.inputs.get('view','advice') not in {'advice','routes','chargers'} or task.inputs.get('destination','机场') != '机场':
            raise PlanningError('模拟机场任务参数不正确。')


def canonical_inputs(task):
    inputs = dict(task.inputs)
    if task.kind == 'vehicle_state': inputs.setdefault('view','state')
    if task.kind == 'driving_history': inputs.setdefault('limit',10)
    if task.kind == 'navigation':
        inputs.setdefault('mode','destination')
        if inputs['mode'] != 'search': inputs.setdefault('strategy','recommended')
    if task.kind == 'airport_advice':
        inputs.setdefault('view','advice')
        inputs.setdefault('destination','机场')
        if inputs['view'] == 'chargers': inputs.setdefault('route_id','')
    return task.model_copy(update={'inputs':inputs})


def validate_plan(value, catalog, max_tasks=MAX_TASKS, completed=None, allow_empty=False):
    completed = completed or {}
    try:
        if len(json.dumps(value, ensure_ascii=False, allow_nan=False)) > 64000:
            raise ValueError()
        plan = TaskPlan.model_validate(value)
    except (ValidationError, TypeError, ValueError, OverflowError, RecursionError):
        raise PlanningError('模型没有返回合法业务任务图；未开始执行新任务。') from None
    if len(plan.tasks) > max_tasks:
        raise PlanningError(f'超过任务数量上限（{max_tasks}项），未开始新任务。')
    if bool(plan.tasks) == bool(plan.clarification.strip()) and not (allow_empty and not plan.tasks and not plan.clarification.strip()):
        raise PlanningError('计划必须包含任务，或仅包含需要澄清的问题。')
    plan.tasks = [canonical_inputs(task) for task in plan.tasks]
    ids = {task.id for task in plan.tasks}
    if len(ids) != len(plan.tasks) or ids & set(completed):
        raise PlanningError('任务ID重复或覆盖了已完成任务。')
    known = ids | set(completed)
    for task in plan.tasks:
        if task.id in task.depends_on or len(set(task.depends_on)) != len(task.depends_on) or not set(task.depends_on) <= known:
            raise PlanningError('任务依赖无效。')
        try:
            validate_inputs(task, catalog)
        except PlanningError:
            raise
        except (ValueError, TypeError, KeyError, OverflowError):
            raise PlanningError('业务目标输入类型不正确。') from None
        refs = list(references(task.inputs))
        if task.condition:
            if not REF_PATTERN.fullmatch(task.condition.ref): raise PlanningError('条件引用格式不正确。')
            refs.append(task.condition.ref)
        if any(ref.split('.')[0] not in task.depends_on for ref in refs):
            raise PlanningError('结果引用必须声明对应前置依赖。')
    done = set(completed)
    while not ids <= done:
        ready = {task.id for task in plan.tasks if task.id not in done and set(task.depends_on) <= done}
        if not ready: raise PlanningError('任务依赖存在环。')
        done |= ready
    return plan


def submitted(response, name):
    """只读取明确的结构化输出，不记录模型内部思考或供应商原文。"""
    calls = getattr(response, 'tool_calls', [])
    if getattr(response, 'invalid_tool_calls', []): raise PlanningError('模型返回了无效的结构化调用。')
    if len(calls) == 1 and calls[0]['name'] == name:
        return calls[0]['args']
    if calls: raise PlanningError('该阶段只能提交规定的结构化结果，不能直接执行业务工具。')
    content = response.content
    if isinstance(content, list):
        content = '\n'.join(item['text'] for item in content if isinstance(item, dict) and item.get('type') == 'text' and isinstance(item.get('text'), str))
    try:
        if not isinstance(content, str) or len(content) > 64000: raise ValueError()
        return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', content.strip()))
    except (ValueError, TypeError, RecursionError):
        raise PlanningError('模型未提交合法结构化结果。') from None
