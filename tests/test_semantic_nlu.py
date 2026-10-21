"""不加载权重：用可注入编码器验证匹配与执行边界。真实CPU测试另行显式开启。"""

import io
import os
import unittest
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cockpit.app import CabinApp
from cockpit.nlu import extract_temperature, mask_temperature, parse_intent, temperature_request_check
from cockpit.nlu_cli import metrics, predicted_label, validate_splits
from cockpit.semantic_nlu import LocalEncoder, NLURouter, NLUStartupError, SemanticNLU

EXAMPLES = {
    "set_temperature": ["把空调设成23度", "将车里温度设到二十四度"],
    "query_vehicle": ["电量剩多少", "空调目前几度"],
}
THRESHOLDS = {name: {"similarity": .75, "margin": .1} for name in EXAMPLES}


class ToyEncoder:
    def __init__(self, overrides=None):
        self.mapping = {mask_temperature(text): [1, 0, 0] if name == "set_temperature" else [0, 1, 0]
                        for name, examples in EXAMPLES.items() for text in examples}
        self.mapping.update({mask_temperature(k): v for k, v in (overrides or {}).items()})
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [self.mapping.get(text, [0, 0, 1]) for text in texts]


def make_router(overrides=None):
    return NLURouter(SemanticNLU(ToyEncoder(overrides), EXAMPLES, THRESHOLDS))


class SlotAndGuardTests(unittest.TestCase):
    def test_chinese_arabic_and_fullwidth_numbers(self):
        for text, expected in [("温度二十四度", 24), ("温度二十四点五度", 24.5),
                               ("温度24.5℃", 24.5), ("温度２３摄氏度", 23),
                               ("温度负三度", -3), ("温度一百零五度", 105)]:
            with self.subTest(text=text):
                self.assertEqual(extract_temperature(text), [expected])

    def test_temperature_is_masked_for_matching(self):
        self.assertEqual(mask_temperature("把空调设成23度"), mask_temperature("把空调设成二十四点五度"))

    def test_malformed_chinese_numbers_do_not_execute(self):
        for text in ["把温度设成二十十度", "把温度设成二十四点度"]:
            self.assertNotEqual(temperature_request_check(text)[0], "execute")

    def test_question_mark_cannot_authorize_rule_write(self):
        for text in ["空调调到24度？", "请把空调设为24度?", "空调调到24度好吗"]:
            self.assertIsNone(parse_intent(text))

    def test_only_explicit_absolute_command_passes(self):
        self.assertEqual(temperature_request_check("把车内温度设置成二十四度")[:2], ("execute", 24))
        for text in ["空调二十四度", "今天车里二十六度", "我喜欢空调二十三度", "当前空调设成24度",
                     "把空调设到23度或25度", "研究将温度设为24度的影响", "把空调设24度并查电量"]:
            with self.subTest(text=text):
                self.assertEqual(temperature_request_check(text)[0], "fallback")

    def test_questions_negations_hypotheticals_and_quotes_rejected(self):
        for text in ["别把空调设成24度", "空调改成24度可以吗", "假设空调设为24度",
                     "他说把空调设为24度", "把空调设置成24度然后开车窗", "明天把空调改成24度"]:
            self.assertEqual(temperature_request_check(text)[0], "fallback")

    def test_relative_and_missing_slots_need_clarification(self):
        for text in ["把温度调低一点", "再低两度", "帮我调一下空调"]:
            self.assertEqual(temperature_request_check(text)[0], "clarify")

    def test_reading_setpoint_is_not_a_missing_write_slot(self):
        for text in ["显示一下车里的空调设定值", "空调设为几度了"]:
            self.assertEqual(temperature_request_check(text)[0], "fallback")


class SemanticTests(unittest.TestCase):
    def test_normalized_vectors_and_distinct_intent_margin(self):
        encoder = ToyEncoder({"把车内温度设置成二十四度": [4, 0, 0]})
        nlu = SemanticNLU(encoder, EXAMPLES, THRESHOLDS)
        result = nlu.recognize("把车内温度设置成二十四度")
        self.assertEqual(result.action, "execute")
        self.assertEqual(result.intent.arguments, {"temperature": 24})
        self.assertAlmostEqual(result.similarity, 1)
        # 两个温度示例并列第一不该导致margin=0；比较的是第二个不同意图。
        self.assertAlmostEqual(result.margin, 1)
        self.assertEqual(len(encoder.calls), 2)

    def test_example_vectors_encoded_once(self):
        encoder = ToyEncoder()
        nlu = SemanticNLU(encoder, EXAMPLES, THRESHOLDS)
        nlu.recognize("未知1")
        nlu.recognize("未知2")
        self.assertEqual(len(encoder.calls[0]), 4)
        self.assertEqual([len(call) for call in encoder.calls[1:]], [1, 1])

    def test_low_similarity_rejected(self):
        result = make_router().recognize("讲个笑话")
        self.assertEqual(result.action, "fallback")
        self.assertEqual(result.reason, "below_similarity_or_margin_threshold")

    def test_small_margin_rejected(self):
        result = make_router({"看看车辆情况": [1, .99, 0]}).recognize("看看车辆情况")
        self.assertEqual(result.action, "fallback")
        self.assertLess(result.margin, .1)

    def test_read_query_accepts_question_mark(self):
        result = make_router({"现在还有多少电？": [0, 1, 0]}).recognize("现在还有多少电？")
        self.assertEqual(result.action, "execute")
        self.assertEqual(result.intent.tool, "get_vehicle_state")

    def test_high_similarity_does_not_bypass_write_guard(self):
        for text in ["空调二十四度", "空调设成24度好吗", "如果空调改成24度"]:
            self.assertNotEqual(make_router({text: [1, 0, 0]}).recognize(text).action, "execute")

    def test_trip_and_unsupported_device_are_not_local_queries(self):
        for text in ["查电量再规划去机场的路线", "手机还有多少电", "查一下发动机温度"]:
            self.assertEqual(make_router({text: [0, 1, 0]}).recognize(text).action, "fallback")

    def test_rules_take_priority_without_query_embedding(self):
        router = make_router()
        self.assertEqual(router.recognize("空调调到24度").source, "rule_nlu")
        self.assertEqual(len(router.semantic.encoder.calls), 1)

    def test_app_semantic_write_and_state_query_without_agent(self):
        def forbidden_agent(service):
            raise AssertionError("本地命令不应创建云端Agent")
        router = make_router({"把车内温度设置成二十四度": [1, 0, 0], "现在还有多少电": [0, 1, 0]})
        app = CabinApp("live", agent_factory=forbidden_agent, nlu=router)
        self.assertIn("已设为24", app.handle("把车内温度设置成二十四度"))
        self.assertIn("电量25", app.handle("现在还有多少电"))
        self.assertIsNone(app.agent)
        self.assertEqual(len(app.turns), 2)

    def test_semantic_out_of_range_is_rejected_by_shared_tool(self):
        app = CabinApp("offline", nlu=make_router({"把温度改成三十五度": [1, 0, 0]}))
        self.assertIn("操作未执行", app.handle("把温度改成三十五度"))
        self.assertEqual(app.service.state.temperature_c, 26)

    def test_missing_slot_and_relative_do_not_execute(self):
        app = CabinApp("offline", nlu=make_router())
        for text in ["有点热", "帮我调一下空调", "再低两度"]:
            self.assertIn("绝对目标温度", app.handle(text))
        self.assertEqual(app.service.state.temperature_c, 26)

    def test_fallback_still_reaches_agent_in_live_mode(self):
        class Agent:
            def run(self, text, history):
                return "走原有Agent路径", [{"role": "user", "content": text}, {"role": "assistant", "content": "结果"}]
        app = CabinApp("live", nlu=make_router(), agent_factory=lambda s: Agent())
        self.assertIn("原有Agent", app.handle("去机场要不要充电"))

    def test_missing_local_model_never_downloads(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parents[1]) as path:
            with patch("cockpit.semantic_nlu._load_model") as loader:
                with self.assertRaises(NLUStartupError):
                    LocalEncoder(Path(path))
                loader.assert_not_called()

    def test_hybrid_startup_error_does_not_fallback_to_agent(self):
        from main import main
        output = io.StringIO()
        with patch("main.NLURouter.hybrid", side_effect=NLUStartupError("模型尚未准备")):
            with redirect_stdout(output), redirect_stderr(output):
                self.assertEqual(main(["--nlu", "hybrid", "--mode", "live", "--query", "去机场"]), 2)
        self.assertIn("本地NLU启动失败", output.getvalue())

    def test_invalid_encoder_output_is_rejected(self):
        class BadEncoder:
            def encode(self, texts):
                return [[float("nan"), 0] for _ in texts]
        with self.assertRaises(ValueError):
            SemanticNLU(BadEncoder(), EXAMPLES, THRESHOLDS)


class EvaluationTests(unittest.TestCase):
    def test_data_splits_and_minimum_sizes(self):
        validate_splits()

    def test_confusion_and_false_accept_metrics(self):
        records = [
            {"expected": "set_temperature", "predicted": "set_temperature", "latency_ms": 10, "decision": {"source": "semantic_nlu"}},
            {"expected": "fallback", "predicted": "query_vehicle", "latency_ms": 20, "decision": {"source": "semantic_nlu"}},
            {"expected": "clarify", "predicted": "clarify", "latency_ms": 1, "decision": {"source": "nlu_guard"}},
        ]
        result = metrics(records)
        self.assertEqual(result["false_accept_count"], 1)
        self.assertEqual(result["misexecution_count"], 0)
        self.assertEqual(result["confusion"]["fallback"]["query_vehicle"], 1)
        self.assertAlmostEqual(result["warmed_semantic_p50_ms"], 15)


@unittest.skipUnless(os.getenv("RUN_REAL_NLU_TESTS") == "1", "设置 RUN_REAL_NLU_TESTS=1 才加载真实本地权重")
class RealLocalNLUTests(unittest.TestCase):
    def test_cpu_model_inference_without_any_network_or_cloud(self):
        with patch("socket.socket.connect", side_effect=AssertionError("推理过程中禁止网络")) as connect:
            router = NLURouter.hybrid()
            self.assertEqual(next(router.semantic.encoder.model.parameters()).device.type, "cpu")
            for mode in ["offline", "live"]:
                app = CabinApp(mode, nlu=router)
                self.assertIn("已设为24", app.handle("把车内温度设置成二十四度"))
                self.assertIn("电量25", app.handle("现在还有多少电"))
                # live下否定请求本来会交给Agent，这里只验证本地决策不授予写权限。
                self.assertNotEqual(router.recognize("不要将空调改成二十度").action, "execute")
                self.assertEqual(app.service.state.temperature_c, 24)
                self.assertIsNone(app.agent)
            connect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
