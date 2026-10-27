"""任务DAG调度、结果引用与逐任务授权。工具结果是最终回答的唯一执行事实来源。"""

import math
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from time import monotonic

from cockpit.planner import TaskPlan, validate_plan
from cockpit.safety import WRITE_TOOLS
from cockpit.task_auth import write_grants
from cockpit.tools import build_langchain_tools
from cockpit.driving_memory import range_text, history_text

TERMINAL = {'succeeded', 'failed', 'blocked', 'skipped'}
BLOCKED_CODES = {'SAFETY_BLOCKED', 'WRITE_NOT_AUTHORIZED'}
TASK_LABELS = {'get_vehicle_state': '查询车辆状态', 'get_vehicle_location': '读取模拟车辆位置',
               'set_temperature': '设置空调', 'set_headlights': '开关前照灯', 'adjust_seat': '调整座椅',
               'set_wipers': '设置雨刮', 'get_route_options': '查询模拟机场路线', 'find_chargers': '查询模拟充电站',
               'search_navigation_places': '搜索高德地点', 'plan_navigation_route': '规划高德路线',
               'navigate_to_destination': '就近导航规划', 'estimate_driving_range':'估算剩余续航', 'get_driving_history':'读取长期行驶记忆'}


def same_value(left, right):
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if type(left) in {int, float} and type(right) in {int, float}:
        return left == right
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(same_value(left[k], right[k]) for k in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(same_value(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def resolve_reference(ref, records):
    task_id, *path = ref.split('.')
    record = records.get(task_id)
    if record is None or record.status != 'succeeded':
        raise ValueError('引用的前置任务没有成功。')
    current = record.result
    for key in path:
        if isinstance(current, dict) and key in current:
            current = current[key]
        elif isinstance(current, list) and key.isdigit() and int(key) < len(current):
            current = current[int(key)]
        else:
            raise ValueError('前置结果缺少需要的字段或候选项。')
    return deepcopy(current)


def resolve_arguments(value, records):
    if isinstance(value, dict):
        if set(value) == {'$ref'}:
            return resolve_reference(value['$ref'], records)
        return {k: resolve_arguments(v, records) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_arguments(v, records) for v in value]
    return value


def condition_matches(condition, records):
    actual = resolve_reference(condition.ref, records)
    expected = condition.value
    if condition.operator in {'eq', 'ne'}:
        primitive = {str, int, float, bool, type(None)}
        compatible = type(actual) is type(expected) or (type(actual) in {int, float} and type(expected) in {int, float})
        if type(actual) not in primitive or not compatible:
            raise ValueError('条件字段类型不匹配。')
        equals = same_value(actual, expected)
        return equals if condition.operator == 'eq' else not equals
    if (type(actual) not in {int, float} or type(expected) not in {int, float}
            or not math.isfinite(actual) or not math.isfinite(expected)):
        raise ValueError('数值条件的前置结果类型不匹配。')
    return {'lt': actual < expected, 'lte': actual <= expected,
            'gt': actual > expected, 'gte': actual >= expected}[condition.operator]


@dataclass
class TaskRecord:
    id: str
    tool: str
    depends_on: list
    arguments: dict
    condition: dict | None = None
    order_only: list = field(default_factory=list)
    status: str = 'pending'
    attempts: int = 0
    result: dict | None = None
    error: dict | None = None
    authorization: dict = field(default_factory=dict)
    started_ms: float | None = None
    duration_ms: float = 0

    def to_dict(self):
        return asdict(self)


class Executor:
    def __init__(self, service, max_workers=4, budget_seconds=60):
        self.service = service
        self.max_workers = max(1, min(4, max_workers))
        self.budget_seconds = budget_seconds
        self.last_plan = None
        self.last_records = []

    def _effective_plan(self, plan):
        raw = plan.model_dump()
        catalog = {tool.name: tool for tool in build_langchain_tools(self.service)}
        validate_plan(raw, catalog)
        last_resource = {}
        order_only = {task['id']: [] for task in raw['tasks']}
        # 模型的数组顺序不一定是拓扑序，先满足显式依赖再补资源互斥边，避免人为引入循环。
        ordered, remaining, placed = [], list(raw['tasks']), set()
        while remaining:
            ready = next(task for task in remaining if set(task['depends_on']) <= placed)
            ordered.append(ready)
            remaining.remove(ready)
            placed.add(ready['id'])
        for task in ordered:
            resource = ('actuator:' + task['tool']) if task['tool'] in WRITE_TOOLS else (
                'navigation' if task['tool'] in {'plan_navigation_route', 'navigate_to_destination'} else None)
            if resource:
                previous = last_resource.get(resource)
                if previous and previous not in task['depends_on']:
                    task['depends_on'].append(previous)
                    order_only[task['id']].append(previous)
                last_resource[resource] = task['id']
        return validate_plan(raw, catalog), order_only

    def execute(self, plan: TaskPlan, user_text: str):
        plan, order_only = self._effective_plan(plan)
        self.last_plan = plan.model_dump()
        self.last_plan['resource_order_edges'] = order_only
        records = {task.id: TaskRecord(task.id, task.tool, list(task.depends_on), deepcopy(task.arguments),
                    task.condition.model_dump() if task.condition else None) for task in plan.tasks}
        self.last_records = list(records.values())
        for task_id, record in records.items():
            record.order_only = order_only[task_id]
        self.service.trace('executor.plan', self.last_plan)
        grants = write_grants(user_text)
        start = monotonic()
        running = {}

        def finish(record, status, error=None):
            record.status, record.error = status, error
            if record.started_ms is not None:
                record.duration_ms = round((monotonic() - start) * 1000 - record.started_ms, 2)
            self.service.trace('task.status', record.to_dict())

        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix='cockpit-task') as pool:
            while any(record.status not in TERMINAL for record in records.values()):
                for task in plan.tasks:
                    record = records[task.id]
                    if record.status != 'pending':
                        continue
                    predecessors = [records[dep] for dep in task.depends_on]
                    if not all(dep.status in TERMINAL for dep in predecessors):
                        continue
                    if any(dep.status != 'succeeded' and dep.id not in record.order_only for dep in predecessors):
                        finish(record, 'skipped', {'code': 'DEPENDENCY_FAILED', 'message': '前置任务未成功，未调用本任务工具。'})
                        continue
                    if monotonic() - start >= self.budget_seconds:
                        finish(record, 'skipped', {'code': 'EXECUTION_BUDGET', 'message': '调度时间预算已到，未开始本任务。'})
                        continue
                    if len(running) >= self.max_workers:
                        break
                    try:
                        if task.condition and not condition_matches(task.condition, records):
                            finish(record, 'skipped', {'code': 'CONDITION_NOT_MET', 'message': '前置结果不满足执行条件，未调用工具。'})
                            continue
                        record.arguments = resolve_arguments(task.arguments, records)
                    except (ValueError, TypeError, OverflowError):
                        finish(record, 'failed', {'code': 'RESULT_REFERENCE_ERROR', 'message': '无法从真实前置结果解析参数或条件，未调用工具。'})
                        continue
                    authorized = False
                    source = ''
                    if task.tool in WRITE_TOOLS:
                        for grant in grants:
                            if not grant['used'] and grant['tool'] == task.tool and same_value(grant['arguments'], record.arguments):
                                grant['used'], authorized, source = True, True, grant['source']
                                break
                    record.authorization = {'required': task.tool in WRITE_TOOLS, 'allowed': authorized if task.tool in WRITE_TOOLS else True, 'source': source}
                    record.status, record.started_ms = 'running', round((monotonic() - start) * 1000, 2)
                    self.service.trace('task.status', record.to_dict())
                    future = pool.submit(self._run_task, record.id, task.tool, record.arguments, authorized)
                    running[future] = record
                if not running:
                    continue
                completed, _ = wait(running, return_when=FIRST_COMPLETED, timeout=0.1)
                for future in completed:
                    record = running.pop(future)
                    try:
                        record.result, record.attempts = future.result()
                        if record.result['ok']:
                            finish(record, 'succeeded')
                        else:
                            error = record.result['error']
                            finish(record, 'blocked' if error['code'] in BLOCKED_CODES else 'failed', error)
                    except Exception:
                        record.attempts = max(1, record.attempts)
                        finish(record, 'failed', {'code': 'TOOL_EXECUTION_ERROR', 'message': '工具调用异常，执行结果未知；未自动重试，请先查询状态。'})
        counts = {status: sum(r.status == status for r in records.values()) for status in TERMINAL}
        self.service.trace('executor.completed', {'counts': counts, 'duration_ms': round((monotonic() - start) * 1000, 2)})
        return [record.to_dict() for record in records.values()]

    def _run_task(self, task_id, tool, arguments, authorized):
        with self.service.task_context(task_id):
            if tool in WRITE_TOOLS:
                self.service.trace('task.guard', {'allowed': authorized, 'action': tool,
                    'reason': '原句中存在参数一致的明确完整命令。' if authorized else '原句未授予该动作与参数的明确写权限。'})
            for attempt in [1, 2]:
                result = self.service.execute(tool, arguments, allow_write=authorized)
                if result.get('ok') and tool in WRITE_TOOLS and result.get('data', {}).get('verified') is not True:
                    result = {**result, 'ok': False, 'error': {'code': 'VERIFICATION_FAILED',
                              'message': '工具未通过状态读回核验，可能已执行；未重试，请先查询状态。'}}
                # 只对无副作用的地点搜索连接失败重试一次；车控和导航计划不重试。
                if attempt == 1 and tool == 'search_navigation_places' and result.get('error', {}).get('code') == 'AMAP_NETWORK':
                    self.service.trace('task.retry', {'attempt': 2, 'reason': '地点搜索网络失败，最多重试一次。'})
                    continue
                return result, attempt


def result_text(tool, data):
    if tool == 'estimate_driving_range':
        return range_text(data)
    if tool == 'get_driving_history':
        return history_text(data)
    if tool == 'set_temperature':
        return f"模拟空调已设为{data['temperature_c']:g}℃，读回核验通过。"
    if tool == 'set_headlights':
        return f"模拟前照灯已{'打开' if data['headlights_on'] else '关闭'}，读回核验通过。"
    if tool == 'adjust_seat':
        return f"模拟座椅已设为{data['seat_position_percent']:g}%，读回核验通过。"
    if tool == 'set_wipers':
        return f"模拟雨刮器已设为{ {'off':'关闭','low':'低速','high':'高速'}[data['wiper_mode']]}，读回核验通过。"
    if tool == 'get_vehicle_state':
        return f"查询时模拟车辆电量{data['battery_percent']:g}%，空调{data['temperature_c']:g}℃，{data['motion']['label']}。"
    if tool == 'get_vehicle_location':
        return f"模拟车辆位置：{data['name']}（{data['location']}）。"
    if tool == 'get_route_options':
        return '；'.join(f"{r['name']}：{r['duration_minutes']}分钟，到达电量{r['arrival_battery_percent']:g}%，{'满足' if r['meets_reserve'] else '不满足'}{data['reserve_percent']:g}%预留" for r in data['routes']) + '。以上为模拟估算。'
    if tool == 'find_chargers':
        return '；'.join(f"{c['name']}：可用端口{c['available_ports']}，含充电行程{c['total_trip_minutes']}分钟" for c in data['chargers']) + '。模拟快照，未预约或充电。'
    if tool == 'search_navigation_places':
        return f"找到{len(data['places'])}个高德候选：" + '、'.join(p['name'] for p in data['places'][:5]) + '。'
    if data.get('status') == 'near_destination':
        return data['note']
    if 'routes' in data:
        routes = '；'.join(f"{r['distance_km']}公里／{r['duration_minutes']}分钟" for r in data['routes'])
        selection = data.get('selection')
        matched = f"自动选择最近的名称匹配候选“{data['destination']['name']}”（近似直线距离{selection['selected']['distance_m']/1000:.2f}公里）；" if selection else ''
        return matched + f"已规划{data['origin']['name']}→{data['destination']['name']}：{routes}。可查看导航面板；未启动导航，无真实耗电估算。"
    return '工具返回成功，结果已记录。'


def summarize(records):
    lines = []
    labels = {'succeeded': '成功', 'blocked': '已拦截', 'failed': '失败', 'skipped': '已跳过'}
    for record in records:
        status = record['status']
        text = result_text(record['tool'], record['result']['data']) if status == 'succeeded' else record['error']['message']
        lines.append(f"{record['id']} · {TASK_LABELS[record['tool']]} [{labels[status]}]：{text}")
    counts = {status: sum(r['status'] == status for r in records) for status in labels}
    return f"本轮{len(records)}项任务：成功{counts['succeeded']}、拦截{counts['blocked']}、失败{counts['failed']}、跳过{counts['skipped']}。\n" + '\n'.join(lines)
