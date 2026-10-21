"""这些测试验证业务约束，不访问模型服务。"""

import unittest

from cockpit.app import DEMO_QUERIES, CabinApp
from cockpit.data import CHARGERS, RESERVE_PERCENT, ROUTES
from cockpit.nlu import parse_intent
from cockpit.tools import ToolService


class NLUTests(unittest.TestCase):
    def test_complete_commands(self):
        for text in ["空调调到24度", "请把空调温度设置为24℃。", "帮我空调设为24.5度"]:
            with self.subTest(text=text):
                self.assertEqual(parse_intent(text).name, "set_temperature")
        self.assertEqual(parse_intent("查看车辆状态").tool, "get_vehicle_state")

    def test_no_keyword_based_execution(self):
        for text in ["不要把空调调到24度", "如果空调调到24度会怎样？", "他说‘空调调到24度’",
                     "空调调到24度好吗", "空调调到24度，然后查路线", "设置一下空调", "再低两度"]:
            with self.subTest(text=text):
                self.assertIsNone(parse_intent(text))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.service = ToolService()

    def test_temperature_boundaries_and_readback(self):
        for value in [16, 24.5, 30]:
            result = self.service.execute("set_temperature", {"temperature": value}, allow_write=True)
            self.assertTrue(result["ok"])
            self.assertTrue(result["data"]["verified"])
            state = self.service.execute("get_vehicle_state", {})["data"]
            self.assertEqual(state["temperature_c"], value)

    def test_invalid_values_do_not_mutate(self):
        for value in [15.9, 30.1, True, "24", None, float("nan"), float("inf")]:
            with self.subTest(value=value):
                result = self.service.execute("set_temperature", {"temperature": value}, allow_write=True)
                self.assertFalse(result["ok"])
                self.assertEqual(self.service.state.temperature_c, 26)

    def test_write_requires_local_authorization(self):
        result = self.service.execute("set_temperature", {"temperature": 24})
        self.assertEqual(result["error"]["code"], "WRITE_NOT_AUTHORIZED")
        self.assertEqual(self.service.state.temperature_c, 26)

    def test_argument_validation(self):
        for name, args in [
            ("unknown", {}), ("get_vehicle_state", {"extra": 1}),
            ("set_temperature", {}), ("get_route_options", {"destination": "火车站"}),
            ("get_route_options", {}), ("find_chargers", {"route_id": "unknown"}),
            ("find_chargers", {"route_id": None}), ("get_vehicle_state", []),
        ]:
            with self.subTest(name=name, args=args):
                self.assertFalse(self.service.execute(name, args)["ok"])

    def test_route_and_charger_estimates_are_consistent(self):
        routes = self.service.execute("get_route_options", {"destination": "机场"})["data"]["routes"]
        self.assertEqual([r["arrival_battery_percent"] for r in routes], [7, 13])
        self.assertEqual([r["meets_reserve"] for r in routes], [False, True])
        for route in routes:
            self.assertEqual(route["arrival_battery_percent"], 25 - route["battery_consumption_percent"])
            self.assertEqual(route["meets_reserve"], route["arrival_battery_percent"] >= RESERVE_PERCENT)
        chargers = self.service.execute("find_chargers", {})["data"]["chargers"]
        self.assertEqual(chargers[0]["total_trip_minutes"], 76)
        self.assertEqual(chargers[0]["arrival_battery_percent"], 37)
        self.assertEqual(chargers[1]["available_ports"], 0)
        self.assertTrue(all(c["route_id"] in {r["route_id"] for r in ROUTES} for c in CHARGERS))
        routes[0]["name"] = "changed"
        self.assertEqual(ROUTES[0]["name"], "机场高速")
        self.assertEqual(self.service.state.battery_percent, 25)  # 查询不会真的开车或充电。


class AppTests(unittest.TestCase):
    def test_three_offline_cases(self):
        events = []
        app = CabinApp("offline", trace=lambda event, data: events.append((event, data)))
        self.assertIn("已设为24", app.handle(DEMO_QUERIES["temperature"]))
        trip = app.handle(DEMO_QUERIES["trip"])
        self.assertIn("离线脚本演示", trip)
        self.assertIn("65分钟", trip)
        self.assertIn("76分钟", trip)
        self.assertIn("操作未执行", app.handle(DEMO_QUERIES["invalid"]))
        self.assertEqual(app.service.state.temperature_c, 24)
        calls = [d["name"] for e, d in events if e == "tool.call"]
        self.assertEqual(calls, ["set_temperature", "get_vehicle_state", "get_route_options",
                                 "find_chargers", "set_temperature"])

    def test_negative_and_unknown_offline_input_do_not_execute(self):
        app = CabinApp("offline")
        app.handle("不要把空调调到24度")
        self.assertEqual(app.service.state.temperature_c, 26)
        self.assertIn("自由追问", app.handle("换一条耗电更少的路线"))

    def test_reset_clears_state_and_history(self):
        app = CabinApp("offline")
        app.handle("空调调到24度")
        self.assertTrue(app.turns)
        app.reset()
        self.assertEqual(app.service.state.temperature_c, 26)
        self.assertEqual(app.turns, [])

    def test_live_nlu_does_not_initialize_model(self):
        def forbidden_factory(service):
            raise AssertionError("不应创建模型")
        app = CabinApp("live", agent_factory=forbidden_factory)
        self.assertIn("已设为24", app.handle("空调调到24度"))

    def test_history_keeps_ten_complete_turns(self):
        app = CabinApp("offline")
        for _ in range(12):
            app.handle("查看车辆状态")
        self.assertEqual(len(app.turns), 10)
        self.assertTrue(all(len(turn) == 2 for turn in app.turns))


if __name__ == "__main__":
    unittest.main()
