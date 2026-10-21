"""本地HTTP接入测试；使用规则NLU，不请求云端或下载权重。"""

import http.cookiejar
import json
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch

from cockpit.semantic_nlu import NLURouter, NLUStartupError
from web import CabinWebServer


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def router(name):
            if name == "hybrid":
                raise NLUStartupError("测试：模型未准备完整")
            return NLURouter()
        cls.server = CabinWebServer(("127.0.0.1", 0), nlu_name="rules", router_factory=router)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=3)

    def setUp(self):
        self.jar = http.cookiejar.CookieJar()
        self.client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.jar))

    def call(self, path, data=None, headers=None, client=None):
        request_headers = {"Origin": self.base}
        if data is not None:
            request_headers["Content-Type"] = "application/json"
        request_headers.update(headers or {})
        body = None if data is None else json.dumps(data).encode("utf-8")
        request = urllib.request.Request(self.base + path, data=body, headers=request_headers)
        try:
            response = (client or self.client).open(request, timeout=5)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            raw = response.read().decode("utf-8")
            payload = json.loads(raw) if response.headers.get_content_type() == "application/json" else raw
            return response.status, payload, response.headers

    def test_static_assets_and_no_private_file_serving(self):
        for path in ["/", "/app.css", "/app.js", "/favicon.svg"]:
            status, _, headers = self.call(path)
            self.assertEqual(status, 200)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        for path in ["/.env", "/.git/config", "/../.env", "/models/config.json"]:
            self.assertEqual(self.call(path)[0], 404)

    def test_temperature_chat_and_event_trace(self):
        status, result, headers = self.call("/api/chat", {"text": "空调调到24度"})
        self.assertEqual(status, 200)
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.assertEqual(result["vehicle"]["temperature_c"], 24)
        self.assertEqual(result["last"]["nlu"]["source"], "rule_nlu")
        self.assertEqual([e["event"] for e in result["last"]["events"]],
                         ["nlu.result", "route", "tool.call", "tool.result"])
        self.assertEqual(len(self.call("/api/session")[1]["messages"]), 2)

    def test_invalid_temperature_keeps_vehicle_state(self):
        self.call("/api/chat", {"text": "空调调到24度"})
        _, result, _ = self.call("/api/chat", {"text": "空调调到35度"})
        self.assertEqual(result["vehicle"]["temperature_c"], 24)
        self.assertTrue(result["last"]["rejected"])

    def test_reset_clears_history_and_state(self):
        self.call("/api/chat", {"text": "空调调到24度"})
        _, result, _ = self.call("/api/reset", {})
        self.assertEqual(result["vehicle"]["temperature_c"], 26)
        self.assertEqual(result["messages"], [])
        self.assertIsNone(result["last"])

    def test_browser_sessions_are_isolated(self):
        self.call("/api/chat", {"text": "空调调到24度"})
        other = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.assertEqual(self.call("/api/session", client=other)[1]["vehicle"]["temperature_c"], 26)
        self.assertEqual(self.call("/api/session")[1]["vehicle"]["temperature_c"], 24)

    def test_mode_switch_preserves_history_and_local_control(self):
        self.call("/api/chat", {"text": "空调调到24度"})
        status, result, _ = self.call("/api/settings", {"mode": "live", "nlu": "rules"})
        self.assertEqual(status, 200)
        self.assertEqual(result["mode"], "live")
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["vehicle"]["temperature_c"], 24)
        self.assertEqual(self.call("/api/chat", {"text": "查看车辆状态"})[1]["last"]["route"], "rule_nlu")

    def test_failed_model_switch_does_not_change_settings(self):
        self.assertEqual(self.call("/api/settings", {"mode": "live", "nlu": "hybrid"})[0], 400)
        _, state, _ = self.call("/api/session")
        self.assertEqual((state["mode"], state["nlu"]), ("offline", "rules"))

    def test_rejects_cross_origin_and_invalid_payload(self):
        self.assertEqual(self.call("/api/reset", {}, headers={"Origin": "https://example.com"})[0], 403)
        self.assertEqual(self.call("/api/chat", {"text": "hi"}, headers={"Host": "example.com"})[0], 403)
        self.assertEqual(self.call("/api/chat", {"text": "hi"}, headers={"Content-Type": "text/plain"})[0], 400)
        for data in [[], {"text": ""}, {"text": 42}, {"text": "x" * 2001}]:
            self.assertEqual(self.call("/api/chat", data)[0], 400)
        self.assertEqual(self.call("/api/settings", {"mode": "invalid", "nlu": "rules"})[0], 400)

    def test_concurrent_request_is_rejected_without_reexecuting(self):
        self.call("/api/session")
        token = next(cookie.value for cookie in self.jar if cookie.name == "cabin_session")
        session = self.server.sessions[token]
        with session.lock:
            self.assertEqual(self.call("/api/chat", {"text": "空调调到24度"})[0], 409)
        self.assertEqual(session.app.service.state.temperature_c, 26)

    def test_internal_errors_are_not_exposed(self):
        self.call("/api/session")
        token = next(cookie.value for cookie in self.jar if cookie.name == "cabin_session")
        with patch.object(self.server.sessions[token].app, "handle", side_effect=RuntimeError("secret-should-not-leak")):
            status, payload, _ = self.call("/api/chat", {"text": "测试异常"})
        self.assertEqual(status, 500)
        self.assertNotIn("secret", json.dumps(payload))


if __name__ == "__main__":
    unittest.main()
