"""真实 Agent：LangChain 负责模型—工具循环，这里提供上下文和运行边界。"""

import os
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage

from cockpit.tools import ToolService, build_langchain_tools

SYSTEM_PROMPT = """你是中文座舱教学助手。所有车辆、路线和充电站均为模拟数据。
涉及车辆状态或路线时先调用工具，依据工具结果回答，不能编造查询、修改或预约成功。
机场出行需查询当前车辆状态和路线；判断到达电量是否至少为工具给出的预留电量。
如某路线电量不足，可查询其充电站；不要推荐 available_ports 为0的站点。
尽量比较耗时、到达电量与充电备选；不擅自假定用户的到达时间约束。
只支持机场路线。不支持的目的地明确说明限制。耗电字段单位是电量百分点。
只输出简洁结论和必要依据，不输出内部思考过程。
车控只有完整明确命令才会由本地NLU授权；你没有温度写权限。
否定、假设、引述、咨询或不完整指令不能触发修改；需要车控时提示单独输入完整命令。
工具返回 ok=false 表示操作失败，不能声称执行成功。修改后的实际值以工具结果为准。
聊天历史可能包含失败记录，失败不代表成功；当前工具结果优先于历史状态。
"""


class ConfigurationError(ValueError):
    pass


class LiveAgent:
    def __init__(self, service: ToolService, *, model: Any, step_limit: int = 12):
        self.service = service
        self.step_limit = step_limit
        self.graph = create_agent(
            model=model, tools=build_langchain_tools(service), system_prompt=SYSTEM_PROMPT,
        )

    @classmethod
    def from_env(cls, service: ToolService) -> "LiveAgent":
        from dotenv import load_dotenv
        from langchain_openai import ChatOpenAI

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        key = os.getenv("OPENAI_API_KEY", "").strip()
        model_name = os.getenv("OPENAI_MODEL", "").strip()
        if not key or not model_name:
            raise ConfigurationError("请在 .env 中填写 OPENAI_API_KEY 和 OPENAI_MODEL，再重试；当前未调用模型。")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or "https://api.openai.com/v1"
        model = ChatOpenAI(
            model=model_name, api_key=key, base_url=base_url,
            timeout=20, max_retries=0, use_responses_api=False,
        )
        return cls(service, model=model)

    def run(self, text: str, history: list) -> tuple[str, list]:
        messages = [*history, HumanMessage(content=text)]
        final_messages = messages
        observed = set()
        # values 返回累计状态；只展示 tool_calls，不打印 AIMessage 的推理内容。
        for state in self.graph.stream(
            {"messages": messages}, config={"recursion_limit": self.step_limit}, stream_mode="values",
        ):
            final_messages = state["messages"]
            message = final_messages[-1]
            if isinstance(message, AIMessage) and message.tool_calls and message.id not in observed:
                observed.add(message.id)
                self.service.trace("agent.tools_selected", {
                    "calls": [{"name": c["name"], "arguments": c["args"]} for c in message.tool_calls]
                })
        last = final_messages[-1]
        if not isinstance(last, AIMessage) or last.tool_calls:
            raise RuntimeError("Agent未产生最终回答。")
        # 兼容纯文本和内容块；只提取普通 text，忽略 reasoning 等扩展块。
        reply = last.content if isinstance(last.content, str) else "\n".join(
            block["text"] for block in last.content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        )
        if not reply.strip():
            raise RuntimeError("模型没有返回可显示的文本。")
        return reply, final_messages[len(history):]


def friendly_error(exc: Exception) -> str:
    """不直接打印供应商异常正文，避免其中的请求头、地址或密钥进入终端。"""
    if isinstance(exc, ConfigurationError):
        return str(exc)
    name = type(exc).__name__.lower()
    if isinstance(exc, ImportError):
        return "缺少模型依赖，请先安装 requirements.txt；也可以显式选择 --mode offline。"
    if "timeout" in name:
        return "模型请求超时（网络超时配置20秒），请重试；没有切换为离线结果。"
    if "recursion" in name:
        return "达到Agent执行步数上限，本轮已停止。可缩小问题后重试。"
    if "authentication" in name or "permission" in name:
        return "模型服务认证或权限失败，请检查密钥和模型访问权限。"
    if "connection" in name:
        return "无法连接模型服务，请检查网络及 OPENAI_BASE_URL。"
    if "ratelimit" in name:
        return "模型服务限流或额度不足，请检查账户后重试。"
    if "badrequest" in name or "notfound" in name:
        return "模型或接口不兼容，请核对模型名、地址和标准工具调用支持。"
    return "本轮Agent调用失败，请检查模型服务配置及工具调用兼容性；没有生成离线替代结果。"
