"""用受控模型测试真实 LangChain 循环；不代表已验证外部模型效果。"""

import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import Field

from cockpit.agent import ConfigurationError, LiveAgent, friendly_error
from cockpit.app import CabinApp
from cockpit.tools import ToolService


class ScriptedModel(FakeMessagesListChatModel):
    seen_messages: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


class AgentTests(unittest.TestCase):
    def test_multistep_loop_and_followup_context(self):
        events = []
        model = ScriptedModel(responses=[
            tool_call("get_vehicle_state", {}, "state1"),
            tool_call("get_route_options", {"destination": "机场"}, "routes1"),
            tool_call("find_chargers", {"route_id": "airport_fast"}, "chargers1"),
            AIMessage(content="节能路线65分钟，到达剩13%，满足10%预留。"),
            tool_call("get_route_options", {"destination": "机场"}, "routes2"),
            AIMessage(content="继续选择节能路线，耗电12个百分点。"),
        ])
        app = CabinApp("live", trace=lambda e, d: events.append((e, d)),
                       agent_factory=lambda s: LiveAgent(s, model=model))
        app.handle("空调调到24度")
        answer = app.handle("电量只有25%，去机场要不要充电？")
        self.assertIn("65分钟", answer)
        self.assertIn("12个百分点", app.handle("换一条耗电更少的路线"))
        followup_input = model.seen_messages[-2]
        self.assertTrue(any(m.content == "换一条耗电更少的路线" for m in followup_input))
        self.assertTrue(any("空调调到24度" == m.content for m in followup_input))
        self.assertTrue(any(isinstance(m, ToolMessage) and m.name == "find_chargers" for m in followup_input))
        calls = [d["name"] for e, d in events if e == "tool.call"]
        self.assertEqual(calls, ["set_temperature", "get_vehicle_state", "get_route_options",
                                 "find_chargers", "get_route_options"])
        app.reset()
        self.assertEqual(app.turns, [])

    def test_model_cannot_modify_temperature_for_negative_input(self):
        model = ScriptedModel(responses=[
            tool_call("set_temperature", {"temperature": 24}, "unauthorized1"),
            AIMessage(content="温度保持不变。"),
        ])
        app = CabinApp("live", agent_factory=lambda s: LiveAgent(s, model=model))
        self.assertIn("保持不变", app.handle("不要把空调调到24度"))
        self.assertEqual(app.service.state.temperature_c, 26)
        tool_result = next(m for m in app.turns[-1] if isinstance(m, ToolMessage))
        self.assertIn("WRITE_NOT_AUTHORIZED", tool_result.content)

    def test_step_limit_and_next_request_recovers(self):
        model = ScriptedModel(responses=[
            tool_call("get_vehicle_state", {}, f"repeat{i}") for i in range(10)
        ])
        app = CabinApp("live", agent_factory=lambda s: LiveAgent(s, model=model, step_limit=3))
        reply = app.handle("一直查询状态")
        self.assertIn("步数上限", reply)
        self.assertFalse(any(isinstance(m, ToolMessage) for m in app.turns[-1]))
        self.assertIn("已设为24", app.handle("空调调到24度"))

    def test_timeout_redaction_and_next_request_recovers(self):
        def timeout_factory(service):
            raise TimeoutError("Bearer do-not-print-this-secret")
        events = []
        app = CabinApp("live", agent_factory=timeout_factory, trace=lambda e, d: events.append((e, d)))
        self.assertIn("超时", app.handle("去机场"))
        self.assertNotIn("do-not-print", str(events) + str(app.turns))
        self.assertIn("电量25", app.handle("查看车辆状态"))

    def test_missing_config_is_explicit(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "", "OPENAI_MODEL": ""}):
            with self.assertRaises(ConfigurationError):
                LiveAgent.from_env(ToolService())
            app = CabinApp("live")
            self.assertIn("未调用模型", app.handle("去机场"))
            self.assertIsNone(app.agent)

    def test_reasoning_blocks_not_displayed(self):
        model = ScriptedModel(responses=[AIMessage(content=[
            {"type": "reasoning", "reasoning": "hidden-test-reasoning"},
            {"type": "text", "text": "这是最终回答。"},
        ])])
        agent = LiveAgent(ToolService(), model=model)
        reply, _ = agent.run("你好", [])
        self.assertEqual(reply, "这是最终回答。")

    def test_error_messages_do_not_include_raw_provider_details(self):
        self.assertNotIn("secret", friendly_error(RuntimeError("api_key=secret")))


if __name__ == "__main__":
    unittest.main()
