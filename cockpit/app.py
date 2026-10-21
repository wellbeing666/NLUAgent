"""应用路由与会话：先读这里，再追踪 NLU、Agent 和工具层。"""

from typing import Callable

from cockpit.semantic_nlu import NLURouter
from cockpit.tools import ToolService, Trace, quiet_trace

DEMO_QUERIES = {
    "temperature": "空调调到24度",
    "trip": "电量只有25%，去机场要不要充电？帮我找个方案",
    "invalid": "空调调到35度",
}


class CabinApp:
    def __init__(self, mode: str, trace: Trace = quiet_trace, agent_factory: Callable | None = None,
                 nlu: NLURouter | None = None):
        if mode not in {"live", "offline"}:
            raise ValueError("mode 必须为 live 或 offline")
        self.mode = mode
        self.service = ToolService(trace)
        self.trace = trace
        self.agent_factory = agent_factory
        self.agent = None
        self.turns: list[list] = []
        self.nlu = nlu if nlu is not None else NLURouter()

    def reset(self) -> str:
        self.service.reset()
        self.turns.clear()
        return "已重置模拟车辆（电量25%、空调26℃）并清空本地对话上下文。"

    def _remember(self, text: str, reply: str) -> None:
        self.turns.append([{"role": "user", "content": text}, {"role": "assistant", "content": reply}])
        self.turns = self.turns[-10:]

    def handle(self, text: str) -> str:
        text = text.strip()
        if not text:
            return "请输入指令，或输入 /help 查看示例。"
        decision = self.nlu.recognize(text)
        self.trace("nlu.result", decision.trace_data())
        if decision.action == "clarify":
            reply = "请给出明确的绝对目标温度，例如：把车内温度设置成二十四度；本地暂不支持相对调节。"
            self._remember(text, reply)
            return reply
        if decision.action == "execute":
            intent = decision.intent
            self.trace("route", {"path": decision.source, "intent": intent.name,
                                 "mode": self.mode, "note": "本地NLU处理，未调用云端大模型"})
            result = self.service.execute(intent.tool, intent.arguments, allow_write=intent.name == "set_temperature")
            if not result["ok"]:
                reply = "操作未执行：" + result["error"]["message"]
            elif intent.name == "set_temperature":
                reply = f"模拟空调已设为{result['data']['temperature_c']:g}℃，已读取状态核验。"
            else:
                state = result["data"]
                reply = f"模拟车辆：电量{state['battery_percent']:g}%，空调{state['temperature_c']:g}℃。"
            self._remember(text, reply)
            return reply
        if self.mode == "offline":
            self.trace("route", {"path": "offline_script", "note": "离线脚本演示，无模型决策"})
            if text in {DEMO_QUERIES["trip"], "演示出行"}:
                return self._offline_trip()
            return "当前请求未被本地NLU接受，未执行操作。离线还支持固定案例“演示出行”；自由追问请使用live模式。"
        self.trace("route", {"path": "langchain_agent", "note": "真实模型决策，不使用离线脚本"})
        history = [message for turn in self.turns for message in turn]
        try:
            if self.agent is None:
                if self.agent_factory:
                    self.agent = self.agent_factory(self.service)
                else:
                    from cockpit.agent import LiveAgent
                    self.agent = LiveAgent.from_env(self.service)
            reply, new_messages = self.agent.run(text, history)
            self.turns.append(new_messages)
            self.turns = self.turns[-10:]
            return reply
        except Exception as exc:
            # 一次失败不结束CLI，也不把不完整的工具消息写入下一轮历史。
            if isinstance(exc, ImportError):
                reply = "缺少模型依赖，请先安装 requirements.txt；也可以显式选择 --mode offline。"
            else:
                from cockpit.agent import friendly_error
                reply = friendly_error(exc)
            self.trace("agent.error", {"message": reply})
            self._remember(text, "本轮未完成：" + reply)
            return reply

    def _offline_trip(self) -> str:
        # 固定调用顺序写在代码里，便于与真实模式的模型选工具形成对照。
        state = self.service.execute("get_vehicle_state", {})["data"]
        routes = self.service.execute("get_route_options", {"destination": "机场"})["data"]["routes"]
        chargers = self.service.execute("find_chargers", {"route_id": "airport_fast"})["data"]["chargers"]
        eco = next(r for r in routes if r["route_id"] == "airport_eco")
        fast = next(r for r in routes if r["route_id"] == "airport_fast")
        charger = next(c for c in chargers if c["available_ports"] > 0)
        return (
            "[离线脚本演示｜固定案例，不解析额外约束] "
            f"当前电量{state['battery_percent']:g}%。"
            f"节能路线需{eco['duration_minutes']}分钟，到达剩余{eco['arrival_battery_percent']:g}%，"
            "满足10%预留，可不充电；"
            f"高速需{fast['duration_minutes']}分钟，但到达只剩{fast['arrival_battery_percent']:g}%，低于预留。"
            f"高速备选：在{charger['name']}充电{charger['charging_minutes']}分钟，"
            f"含绕路总计{charger['total_trip_minutes']}分钟，到达约{charger['arrival_battery_percent']:g}%。"
            "未给出到达时限，固定案例建议节能路线。以上为模拟估算，未启动导航或预约充电。"
        )
