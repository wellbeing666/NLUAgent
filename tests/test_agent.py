"""使用受控规划模型验证Planner+Executor，不调用外部服务。"""

import json
import unittest
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from pydantic import Field

from cockpit.agent import ConfigurationError, LiveAgent, friendly_error
from cockpit.app import CabinApp
from cockpit.tools import ToolService, build_langchain_tools
from cockpit.navigation import AmapClient
from test_navigation import ORIGIN, DESTINATION, FixtureTransport


class ScriptedModel(FakeMessagesListChatModel):
    seen_messages: list = Field(default_factory=list)

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        # 固定计划接线测试没有预置语言回答：显式触发兜底，不消费下一轮的计划响应。
        from cockpit.responder import RESPONSE_PROMPT
        if messages and messages[0].content == RESPONSE_PROMPT:
            raise TimeoutError('固定计划测试未配置回答模型')
        self.seen_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


def tool_call(name, args, call_id):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def task(tool, arguments, task_id="t1", depends_on=None, condition=None):
    return {"id": task_id, "tool": tool, "arguments": arguments, "depends_on": depends_on or [], "condition": condition}


def plan_message(tasks, clarification=""):
    return tool_call("submit_plan", {"tasks": tasks, "clarification": clarification}, "plan")


class AgentTests(unittest.TestCase):
    def test_destination_only_chat_uses_current_position_without_manual_origin(self):
        model = ScriptedModel(responses=[plan_message([task("navigate_to_destination", {"destination": "北京南站"})])])
        app = CabinApp("live", agent_factory=lambda service: LiveAgent(service, model=model))
        app.service._navigation = AmapClient(key="test-only", transport=FixtureTransport())
        app.service.set_vehicle_location({"name": "北京天安门（模拟）", "location": "116.397428,39.909230"})
        reply = app.handle("导航到北京南站")
        self.assertIn("最近", reply)
        self.assertEqual(app.service.navigation_plan["destination"]["id"], ORIGIN["id"])
        self.assertTrue(app.service.navigation_plan["automatic"])
        self.assertEqual(app.last_execution["tasks"][0]["status"], "succeeded")
        self.assertNotIn("set_vehicle_location", [tool.name for tool in build_langchain_tools(app.service)])

    def test_navigation_without_location_does_not_guess_origin(self):
        model = ScriptedModel(responses=[plan_message([task("navigate_to_destination", {"destination": "北京南站"})])])
        service = ToolService(navigation_factory=lambda: self.fail("不应创建高德客户端"))
        agent = LiveAgent(service, model=model)
        reply, messages = agent.run("导航到北京南站", [])
        self.assertIn("设置模拟车辆位置", reply)
        self.assertEqual(agent.last_execution["tasks"][0]["error"]["code"], "LOCATION_NOT_SET")
        self.assertEqual(len(messages), 2)

    def test_search_result_references_feed_route_tool(self):
        search1 = task("search_navigation_places", {"keywords": "南站", "city": "北京"}, "a")
        search2 = task("search_navigation_places", {"keywords": "机场", "city": "北京"}, "b")
        def poi(ref):
            return {k: {"$ref": ref + "." + k} for k in ["name", "location", "id"]}
        route = task("plan_navigation_route", {"origin": poi("a.data.places.0"), "destination": poi("b.data.places.0")}, "c", ["a", "b"])
        service = ToolService(navigation_factory=lambda: AmapClient(key="test-only", transport=FixtureTransport()))
        agent = LiveAgent(service, model=ScriptedModel(responses=[plan_message([search1, search2, route])]))
        agent.run("查询从北京南站去首都机场的路线", [])
        self.assertEqual([r["status"] for r in agent.last_execution["tasks"]], ["succeeded"] * 3)
        self.assertEqual(service.navigation_plan["destination"]["id"], DESTINATION["id"])
        self.assertNotIn("arrival_battery_percent", str(service.navigation_plan))

    def test_plan_history_preserves_real_observations_and_followup(self):
        model = ScriptedModel(responses=[
            plan_message([task("get_vehicle_state", {}, "a"), task("get_route_options", {"destination": "机场"}, "b"),
                          task("find_chargers", {"route_id": {"$ref": "b.data.routes.0.route_id"}}, "c", ["a", "b"])]),
            plan_message([task("get_route_options", {"destination": "机场"})]),
        ])
        app = CabinApp("live", agent_factory=lambda service: LiveAgent(service, model=model))
        app.handle("空调调到24度")
        answer = app.handle("电量只有25%，去机场要不要充电？")
        self.assertIn("65分钟", answer)
        self.assertIn("13%", answer)
        self.assertIn("65分钟", app.handle("换一条耗电更少的路线"))
        history = model.seen_messages[-1]
        self.assertTrue(any(m.content == "换一条耗电更少的路线" for m in history))
        self.assertTrue(any("find_chargers" in str(m.content) and "真实执行记录" in str(m.content) for m in history))
        app.reset()
        self.assertEqual(app.turns, [])
        self.assertIsNone(app.last_execution)

    def test_negative_command_does_not_receive_application_write_permission(self):
        model = ScriptedModel(responses=[plan_message([task("set_temperature", {"temperature": 24})])])
        app = CabinApp("live", agent_factory=lambda service: LiveAgent(service, model=model))
        self.assertIn("拦截", app.handle("不要把空调调到24度"))
        self.assertEqual(app.service.state.temperature_c, 26)
        self.assertEqual(app.last_execution["tasks"][0]["error"]["code"], "WRITE_NOT_AUTHORIZED")

    def test_task_limit_rejects_before_execution_and_local_request_recovers(self):
        model = ScriptedModel(responses=[plan_message([task("get_vehicle_state", {}, f"t{i}") for i in range(4)])])
        app = CabinApp("live", agent_factory=lambda service: LiveAgent(service, model=model, max_tasks=3))
        self.assertIn("任务数量上限", app.handle("一直查询车辆状态"))
        self.assertIsNone(app.last_execution)
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

    def test_reasoning_blocks_not_displayed_or_logged(self):
        payload = json.dumps({"tasks": [], "clarification": "请补充具体需求。"}, ensure_ascii=False)
        model = ScriptedModel(responses=[AIMessage(content=[
            {"type": "reasoning", "reasoning": "hidden-test-reasoning"}, {"type": "text", "text": payload}
        ])])
        events = []
        agent = LiveAgent(ToolService(lambda e, d: events.append((e, d))), model=model)
        reply, _ = agent.run("你好", [])
        self.assertIn("请补充具体需求", reply)
        self.assertNotIn("hidden-test-reasoning", reply + str(events))

    def test_error_messages_do_not_include_raw_provider_details(self):
        self.assertNotIn("secret", friendly_error(RuntimeError("api_key=secret")))


if __name__ == "__main__":
    unittest.main()
