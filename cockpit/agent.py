"""Planner + Executor入口；保留LiveAgent名称供原有CLI/Web调用。"""

import json
import os
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from cockpit.executor import Executor, result_text
from cockpit.planner import Planner, PlanningError, MAX_TASKS
from cockpit.responder import Responder
from cockpit.tools import ToolService


class ConfigurationError(ValueError):
    pass


class LiveAgent:
    def __init__(self, service: ToolService, *, model: Any, max_tasks=MAX_TASKS, max_workers=4, response_model=None):
        self.service = service
        self.planner = Planner(service, model, max_tasks=max_tasks)
        self.executor = Executor(service, max_workers=max_workers)
        self.responder = Responder(response_model if response_model is not None else model, service.trace)
        self.last_execution = None

    @classmethod
    def from_env(cls, service: ToolService):
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI

        load_dotenv(Path(__file__).resolve().parents[1] / '.env', override=False)
        key, model_name = os.getenv('OPENAI_API_KEY', '').strip(), os.getenv('OPENAI_MODEL', '').strip()
        if not key or not model_name:
            raise ConfigurationError('请在 .env 中填写 OPENAI_API_KEY 和 OPENAI_MODEL，再重试；当前未调用模型。')
        model = ChatOpenAI(model=model_name, api_key=key,
                           base_url=os.getenv('OPENAI_BASE_URL', '').strip() or 'https://api.openai.com/v1',
                           timeout=20, max_retries=0, use_responses_api=False)
        return cls(service, model=model)

    def run(self, text: str, history: list):
        self.last_execution = None
        try:
            plan = self.planner.plan(text, history)
            if plan.clarification:
                reply = '本轮未执行任何任务。需要补充信息：' + plan.clarification
                self.last_execution = {'plan': plan.model_dump(), 'tasks': [], 'counts': {}, 'clarification': plan.clarification}
            else:
                records = self.executor.execute(plan, text)
                counts = {status: sum(r['status'] == status for r in records) for status in ['succeeded', 'blocked', 'failed', 'skipped']}
                self.last_execution = {'plan': self.executor.last_plan, 'tasks': records, 'counts': counts}
                reply, answer_info = self.responder.respond(text, records, history)
                self.last_execution['answer'] = answer_info
        except PlanningError:
            self.service.trace('planner.error', {'message': '任务计划校验失败，本轮未执行业务工具。'})
            raise
        # 闭合的普通消息对保留上下文；不把未执行的模型tool_call伪装成工具观察结果。
        receipt = [{'id': r['id'], 'tool': r['tool'], 'arguments': r['arguments'], 'status': r['status'],
                    'observation': result_text(r['tool'], r['result']['data']) if r['status'] == 'succeeded' else r['error']}
                   for r in self.last_execution['tasks']]
        history_reply = reply + ('\n[真实执行记录]\n' + json.dumps(receipt, ensure_ascii=False) if receipt else '')
        return reply, [HumanMessage(content=text), AIMessage(content=history_reply)]


def friendly_error(exc):
    if isinstance(exc, (ConfigurationError, PlanningError)):
        return str(exc)
    name = type(exc).__name__.lower()
    if isinstance(exc, ImportError):
        return '缺少模型依赖，请先安装 requirements.txt；也可以显式选择 --mode offline。'
    if 'timeout' in name:
        return '模型规划请求超时（网络超时配置20秒），本轮未开始任务；请重试。'
    if 'authentication' in name or 'permission' in name:
        return '模型服务认证或权限失败，请检查密钥和模型访问权限。'
    if 'connection' in name:
        return '无法连接模型服务，请检查网络及 OPENAI_BASE_URL。'
    if 'ratelimit' in name:
        return '模型服务限流或额度不足，请检查账户后重试。'
    if 'badrequest' in name or 'invalidrequest' in name or 'notfound' in name:
        return '模型或接口不兼容，请核对模型名、地址和标准工具调用支持。'
    return '本轮Planner或执行汇总失败，请查询当前状态后重试；未生成离线替代结果。'
