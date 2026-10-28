"""业务目标DAG调度：ReAct单任务执行、全局修订与不可重放的事实账本。"""
import math
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from time import monotonic

from cockpit.task_types import TaskPlan, validate_plan, primary_tools, resource, PlanningError
from cockpit.run_state import RunState, BudgetExceeded, same_value
from cockpit.task_worker import TaskWorker, error
from cockpit.tools import build_langchain_tools
from cockpit.driving_memory import range_text, history_text

TERMINAL={'succeeded','failed','blocked','skipped','superseded'}
TASK_LABELS={'get_vehicle_state':'查询车辆状态','get_vehicle_location':'读取模拟车辆位置',
 'set_temperature':'设置空调','set_headlights':'开关前照灯','adjust_seat':'调整座椅','set_wipers':'设置雨刮',
 'get_route_options':'查询模拟机场路线','find_chargers':'查询模拟充电站','search_navigation_places':'搜索高德地点',
 'plan_navigation_route':'规划高德路线','navigate_to_destination':'就近导航规划',
 'estimate_driving_range':'估算剩余续航','get_driving_history':'读取长期行驶记忆'}

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
    kind: str
    goal: str
    inputs: dict
    depends_on: list
    tool: str
    arguments: dict
    condition: dict | None = None
    order_only: list = field(default_factory=list)
    status: str = 'pending'
    attempts: int = 0
    result: dict | None = None
    error: dict | None = None
    authorization: dict = field(default_factory=dict)
    steps: list = field(default_factory=list)
    primary_step_id: str | None = None
    evidence_ids: list = field(default_factory=list)
    needs_replan: bool = False
    version: int = 1
    superseded_by: str | None = None
    recovered_from: list = field(default_factory=list)
    auto_finalized: bool = False
    started_ms: float | None = None
    duration_ms: float = 0

    def to_dict(self): return asdict(self)


def rewrite_refs(value, mapping):
    if isinstance(value,dict):
        if set(value)=={'$ref'}:
            task_id, rest=value['$ref'].split('.',1)
            return {'$ref':mapping.get(task_id,task_id)+'.'+rest}
        return {key:rewrite_refs(child,mapping) for key,child in value.items()}
    if isinstance(value,list): return [rewrite_refs(child,mapping) for child in value]
    return value


class Executor:
    def __init__(self, service, model, max_workers=4, budget_seconds=120, max_steps=6):
        self.service, self.model = service, model
        self.max_workers=max(1,min(4,max_workers))
        self.budget_seconds=budget_seconds
        self.worker=TaskWorker(service,model,max_steps)
        self.catalog={tool.name:tool for tool in build_langchain_tools(service)}
        self.last_plan=None
        self.last_records=[]
        self.run_state=None

    def _effective_plan(self,plan):
        validate_plan(plan.model_dump(),self.catalog)
        ordered,remaining,placed=[],list(plan.tasks),set()
        while remaining:
            ready=next(task for task in remaining if set(task.depends_on)<=placed)
            ordered.append(ready); remaining.remove(ready); placed.add(ready.id)
        previous,orders={}, {task.id:[] for task in plan.tasks}
        for task in ordered:
            key=resource(task)
            if key:
                last=previous.get(key)
                if last and last not in task.depends_on: orders[task.id].append(last)
                previous[key]=task.id
        return orders

    def execute(self,plan,user_text,*,run=None,replanner=None):
        plan=validate_plan(plan.model_dump(),self.catalog)
        run=run or RunState(user_text,budget_seconds=self.budget_seconds)
        self.run_state=run
        specs={task.id:task for task in plan.tasks}
        records={}
        active=list(specs)
        version=0

        def install(new_plan,reason):
            nonlocal plan,version,active
            orders=self._effective_plan(new_plan)
            new_ids=set(task.id for task in new_plan.tasks)-set(records)
            with run.lock:
                if len(new_ids)+run.task_instances>run.max_instances: raise PlanningError('累计任务实例达到上限。')
                run.task_instances+=len(new_ids)
            version+=1; plan=new_plan; active=[task.id for task in plan.tasks]
            for task in plan.tasks:
                specs[task.id]=task
                if task.id not in records:
                    records[task.id]=TaskRecord(task.id,task.kind,task.goal,deepcopy(task.inputs),
                        list(task.depends_on)+orders[task.id],next(iter(sorted(primary_tools(task)))),deepcopy(task.inputs),
                        task.condition.model_dump() if task.condition else None,order_only=orders[task.id],version=version)
                elif records[task.id].status=='pending':
                    records[task.id].depends_on=list(task.depends_on)+orders[task.id]
                    records[task.id].order_only=orders[task.id]
            self.last_plan={**plan.model_dump(),'resource_order_edges':orders,'version':version}
            run.versions.append({'version':version,'reason':reason,'plan':deepcopy(self.last_plan)})
            self.service.trace('executor.plan',self.last_plan)
            self.last_records=[records[key] for key in active]

        def finish(record,status,err=None):
            record.status,record.error=status,err
            if record.started_ms is not None: record.duration_ms=round(run.elapsed_ms()-record.started_ms,2)
            self.service.trace('task.status',record.to_dict())

        def apply_repair(decision):
            if decision.action!='replace':
                if decision.tasks or decision.replacements: raise PlanningError('结束重规划时不能夹带任务。')
                run.clarification=decision.clarification
                return False
            if decision.clarification: raise PlanningError('任务修订与澄清不能混合。')
            remaining={key:specs[key] for key in active if records[key].status=='pending' or records[key].needs_replan}
            pinned={key:records[key].to_dict() for key in active if key not in remaining}
            mapping=decision.replacements
            if not set(mapping)<=set(remaining) or len(set(mapping.values()))!=len(mapping): raise PlanningError('替换映射无效。')
            if any(value in records for value in mapping.values()): raise PlanningError('替代任务必须使用全新执行ID。')
            validated=validate_plan({'tasks':[task.model_dump() for task in decision.tasks],'clarification':''},self.catalog,completed=pinned)
            new_by_id={task.id:task for task in validated.tasks}
            expected={mapping.get(key,key) for key in remaining}
            if not expected<=set(new_by_id): raise PlanningError('重规划丢弃了原剩余目标。')
            for key,old in remaining.items():
                new=new_by_id[mapping.get(key,key)]
                if records[key].needs_replan and key not in mapping: raise PlanningError('失败恢复必须使用新任务ID。')
                if new.kind!=old.kind or new.goal!=old.goal or not same_value(new.inputs,rewrite_refs(old.inputs,mapping)):
                    raise PlanningError('重规划不能修改原目标或输入约束。')
                if not {mapping.get(dep,dep) for dep in old.depends_on} <= set(new.depends_on):
                    raise PlanningError('重规划不能删除用户原任务的前置约束。')
                if (new.condition.model_dump() if new.condition else None)!=rewrite_condition(old.condition,mapping):
                    raise PlanningError('重规划不能放宽原任务执行条件。')
            for key in set(new_by_id)-expected:
                if new_by_id[key].kind not in {'vehicle_state','driving_range','driving_history'} or key in records:
                    raise PlanningError('附加前置只能是新的只读状态/续航/记忆查询。')
            all_tasks=[specs[key] for key in active if key not in remaining]+list(validated.tasks)
            proposed=validate_plan({'tasks':[task.model_dump() for task in all_tasks],'clarification':''},self.catalog)
            if run.task_instances+len(set(new_by_id)-set(records))>run.max_instances: raise PlanningError('累计任务实例达到上限。')
            # 先验证全部新图，再改变记录。成功事实、授权账本、失败回执均不能被覆盖。
            for old_id,new_id in mapping.items():
                old=records[old_id]
                old.superseded_by=new_id
                if old.status=='pending': old.status='superseded'
                run.audit.append(old.to_dict())
            install(proposed,decision.reason)
            for old_id,new_id in mapping.items():
                records[new_id].recovered_from=[*records[old_id].recovered_from,old_id]
            self.service.trace('replanner.applied',{'version':version,'replacements':mapping,'reason':decision.reason})
            return True

        install(plan,'初始全局规划')
        running={}
        pause=False
        with ThreadPoolExecutor(max_workers=self.max_workers,thread_name_prefix='cockpit-react') as pool:
            while pause or running or any(records[key].status=='pending' for key in active):
                if pause and not running:
                    candidates=[{'task':specs[key].model_dump(),'execution':records[key].to_dict()} for key in active
                                if records[key].status=='pending' or records[key].needs_replan]
                    try:
                        if replanner is None or run.replans>=run.max_replans: raise PlanningError('本轮不再继续重规划。')
                        decision=replanner.repair(run,candidates,[records[key].to_dict() for key in active
                                                  if records[key].status=='succeeded'])
                        adopted=apply_repair(decision)
                        if not adopted: self.service.trace('replanner.stopped',{'reason':decision.reason,'clarification':decision.clarification})
                    except Exception:
                        adopted=False
                        self.service.trace('replanner.stopped',{'reason':'修订失败、预算耗尽或违反目标约束；保留原真实结果，不重试写操作。'})
                    if not adopted:
                        for key in active: records[key].needs_replan=False
                    pause=False
                if not pause:
                    for key in list(active):
                        record,task=records[key],specs[key]
                        if record.status!='pending': continue
                        predecessors=[records[dep] for dep in record.depends_on]
                        if not all(parent.status in TERMINAL for parent in predecessors): continue
                        causal=[parent for parent in predecessors if parent.id not in record.order_only]
                        if any(parent.status!='succeeded' for parent in causal):
                            finish(record,'skipped',{'code':'DEPENDENCY_FAILED','message':'前置目标未成功，未启动本任务。'}); continue
                        if len(running)>=self.max_workers: break
                        try:
                            run.check()
                            if task.condition and not condition_matches(task.condition,records):
                                finish(record,'skipped',{'code':'CONDITION_NOT_MET','message':'真实前置结果不满足条件，未启动任务。'}); continue
                            inputs=resolve_arguments(task.inputs,records)
                            resolved=task.model_copy(update={'inputs':inputs})
                        except BudgetExceeded as exc:
                            finish(record,'skipped',{'code':'EXECUTION_BUDGET','message':str(exc)}); continue
                        except (ValueError,TypeError,OverflowError):
                            record.needs_replan=task.kind!='vehicle_control'
                            finish(record,'failed',{'code':'RESULT_REFERENCE_ERROR','message':'前置真实结果没有需要的参数，未猜值。'})
                            pause=record.needs_replan
                            if pause: break
                            continue
                        record.arguments=deepcopy(inputs)
                        record.status,record.started_ms='running',run.elapsed_ms()
                        self.service.trace('task.status',record.to_dict())
                        context={dep:records[dep].result for dep in task.depends_on if records[dep].status=='succeeded'}
                        running[pool.submit(self.worker.run,resolved,run,context)]=record
                if running:
                    completed,_=wait(running,timeout=.1,return_when=FIRST_COMPLETED)
                    for future in completed:
                        record=running.pop(future)
                        try:
                            result=future.result()
                            for name in ['tool','result','error','steps','primary_step_id','evidence_ids','authorization','needs_replan','auto_finalized']:
                                setattr(record,name,result[name])
                            record.attempts=sum(bool(step['called']) for step in record.steps)
                            finish(record,result['status'],result['error'])
                            pause=pause or record.needs_replan
                        except Exception:
                            finish(record,'failed',{'code':'TOOL_EXECUTION_ERROR','message':'任务执行异常，结果可能未知；不自动重试写入。'})
                elif not pause and any(records[key].status=='pending' for key in active):
                    for key in active:
                        if records[key].status=='pending': finish(records[key],'skipped',{'code':'DEPENDENCY_FAILED','message':'没有可继续调度的依赖链。'})
        self.last_records=[records[key] for key in active]
        result=[record.to_dict() for record in self.last_records]
        counts={status:sum(row['status']==status for row in result) for status in ['succeeded','blocked','failed','skipped']}
        self.service.trace('executor.completed',{'counts':counts,'duration_ms':run.elapsed_ms(),'versions':version})
        return result


def rewrite_condition(condition,mapping):
    if condition is None: return None
    value=condition.model_dump()
    task_id,rest=value['ref'].split('.',1)
    value['ref']=mapping.get(task_id,task_id)+'.'+rest
    return value


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
