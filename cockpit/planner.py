"""Planner只提交业务目标图；工具循环由单任务ReAct负责。"""
import json
from langchain_core.messages import HumanMessage, SystemMessage
from cockpit.tools import build_langchain_tools
from cockpit.task_types import (MAX_TASKS, Task, TaskPlan, Condition, PlanningError, validate_plan, references, submitted)

INPUT_GUIDE = """
任务inputs格式（只填允许字段，不填allow_write或allowed_tools）：
vehicle_control: {"tool":"set_temperature","arguments":{"temperature":24}}，其他车控同真实工具参数。
vehicle_state: {}或{"view":"location"}。
driving_range: {}。driving_history: {"limit":10}，limit可省略。
navigation: 目的地导航{"mode":"destination","destination":"北京南站","strategy":"recommended"}；
明确起终点{"mode":"between_places","origin_name":"北京南站","destination":"首都机场T3","city":"北京"}；
仅搜索地点{"mode":"search","keywords":"北京南站","city":"北京"}。支持origin/destination为前置查询得到的{name,location,id}引用，不猜坐标。
airport_advice: {"view":"advice","destination":"机场"}；仅查路线view=routes；仅查充电站view=chargers、route_id可选。
复杂机场建议作为一个业务目标交给ReAct动态查询，不必把每次工具调用拆成全局任务。
一个目标的证据来自该目标内的真实工具步骤；任务data是主要工具真实data，可供后续$ref引用。
"""
PLANNER_PROMPT = """你是中文智能座舱Planner，只通过submit_plan提交结构化任务图，不执行工具，不报告成功。
单个不明确意图和多个意图都可规划。保留所有明确任务，不擅自增加副作用。
任务字段：id、kind、goal、inputs、depends_on、condition。kind只有vehicle_control/vehicle_state/driving_range/driving_history/navigation/airport_advice。任务最多12项；缺少目的地/目标温度等关键参数时返回tasks=[]及clarification，不猜测。
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
        self.service, self.max_tasks = service, max_tasks
        self.catalog = {tool.name:tool for tool in build_langchain_tools(service)}
        schema={'type':'function','function':{'name':'submit_plan','description':'提交全局业务目标DAG，不是逐工具调用列表。','parameters':TaskPlan.model_json_schema()}}
        self.model=model.bind_tools([schema],tool_choice='auto')

    def plan(self, text, history, run=None):
        self.service.trace('planner.started',{'max_tasks':self.max_tasks})
        catalog=[{'name':t.name,'description':t.description,'parameters':t.args_schema.model_json_schema()} for t in self.catalog.values()]
        messages=[SystemMessage(content=PLANNER_PROMPT+INPUT_GUIDE+'\n业务工具说明：'+json.dumps(catalog,ensure_ascii=False)),*history,HumanMessage(content=text)]
        response=run.invoke(self.model,messages) if run else self.model.invoke(messages)
        plan=validate_plan(submitted(response,'submit_plan'),self.catalog,self.max_tasks)
        self.service.trace('planner.plan',plan.model_dump())
        return plan
