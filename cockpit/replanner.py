"""全局剩余计划修订；只提交结构化建议，执行权限和采纳条件仍由调度器检查。"""

import json
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from cockpit.planner import INPUT_GUIDE
from cockpit.task_types import Task, TaskPlan, PlanningError, submitted


class Revision(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['new','revise']
    reason: str = Field(max_length=500)
    plan: TaskPlan | None = None


class Repair(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    action: Literal['replace','stop']
    reason: str = Field(min_length=1, max_length=500)
    tasks: list[Task] = Field(default_factory=list, max_length=12)
    replacements: dict[str,str] = Field(default_factory=dict)
    clarification: str = Field(default='', max_length=500)


REPLANNER_PROMPT = '''你是全局Replanner，只修订剩余业务目标，不调用业务工具。
成功任务不可修改或重放；硬失败、安全拦截、无授权、缺配置、写入结果未知不能通过重规划重试。
可恢复任务用新ID替代，并提供replacements={旧ID:新ID}；其他未完成任务保留原ID。
必须提交所有剩余任务，修订其depends_on及$ref，使其引用新的替代任务或已成功任务。
原任务的kind、goal及inputs约束必须原样保留（仅把输入中的旧结果引用改为新ID）。不得放宽目的地、路线偏好或车控参数。
可增加vehicle_state/driving_range/driving_history只读前置任务；其余新增任务必须替换原同类目标。
工具循环由各任务ReAct负责；你的作用是补足前置、重排依赖或在原目标内重新组织剩余计划。
无法在原约束内修复时action=stop，tasks=[]、replacements={}，reason说明实际限制，clarification写需要用户补充的问题。
只能通过submit_replan提交修订。''' + INPUT_GUIDE

NEXT_PROMPT = '''你是下一轮需求识别的Replanner。只通过submit_revision输出。
判断本轮是独立新请求(action=new,plan=null)，还是对上轮目标的明确修改/追问(action=revise,plan为本轮新的业务目标图)。
修改时只计划本轮真正需要执行的新动作/查询，不能重放上轮已完成的空调/灯等控制，不能声称撤销已发生的动作。
本轮写授权只来自本轮原句，历史命令不授权。新的目的地可依据本轮明确要求改变，其他原约束在未被明确修改时保留。
缺少关键参数时plan.tasks=[]并提出澄清；有任务时plan.clarification为空。状态需重新查询，不能把上轮旧状态当最新遥测。
不能调用业务工具。''' + INPUT_GUIDE


def brief(row):
    return {key:row.get(key) for key in ['id','kind','goal','inputs','status','tool','error','result']}


class Replanner:
    def __init__(self, service, model):
        self.service = service
        self.next_model = model.bind_tools([{'type':'function','function':{'name':'submit_revision','description':'识别下一轮新请求或目标修改。','parameters':Revision.model_json_schema()}}],tool_choice='auto')
        self.repair_model = model.bind_tools([{'type':'function','function':{'name':'submit_replan','description':'提交经代码约束检查的剩余任务修订。','parameters':Repair.model_json_schema()}}],tool_choice='auto')

    def next_request(self, text, history, previous, run):
        self.service.trace('replanner.next_started', {'previous_run_id':previous.get('run',{}).get('run_id')})
        context = {'user_request':text, 'previous_plan':previous.get('plan'),
                   'previous_results':[brief(row) for row in previous.get('tasks',[])]}
        payload = json.dumps(context,ensure_ascii=False,allow_nan=False)
        if len(payload)>48000: payload = payload[:48000]+'\n[旧结果过长已截断，以本轮重新查询为准]'
        response = run.invoke(self.next_model,[SystemMessage(content=NEXT_PROMPT),*history[-4:],HumanMessage(content=payload)])
        try:
            decision = Revision.model_validate(submitted(response,'submit_revision'))
            if decision.action == 'new':
                # 分类器偶尔附带草案；新请求始终交给Planner，绝不执行这份多余草案。
                decision = decision.model_copy(update={'plan':None})
            elif decision.plan is None:
                raise ValueError()
        except (ValueError,TypeError):
            # 识别层格式错误不应使本轮原始请求丢失；只允许回到一次完整规划，不执行坏图。
            decision = Revision(action='new',reason='需求修订结构未通过校验，按本轮原句与已有上下文重新规划。',plan=None)
            self.service.trace('replanner.next_fallback', {'reason':decision.reason})
        self.service.trace('replanner.next_decision', {'action':decision.action,'reason':decision.reason})
        return decision

    def repair(self, run, remaining, completed):
        run.check()
        with run.lock:
            if run.replans >= run.max_replans: raise PlanningError('本轮重规划次数已达上限。')
            run.replans += 1
        self.service.trace('replanner.started', {'attempt':run.replans,'remaining_ids':[row['task']['id'] for row in remaining]})
        payload = {'user_request':run.user_text,'remaining':remaining,'completed':[brief(row) for row in completed],
                   'write_ledger':run.write_ledger}
        text = json.dumps(payload,ensure_ascii=False,allow_nan=False)
        if len(text)>60000: raise PlanningError('重规划上下文过长，未继续增加任务。')
        response = run.invoke(self.repair_model,[SystemMessage(content=REPLANNER_PROMPT),HumanMessage(content=text)])
        try:
            return Repair.model_validate(submitted(response,'submit_replan'))
        except (ValueError,TypeError):
            raise PlanningError('剩余任务修订格式不正确，原执行记录已保留。') from None
