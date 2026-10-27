"""仅用于浏览器界面测试：python tests/preview_navigation.py；不连接高德。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import web
from cockpit.navigation import AmapClient
from cockpit.agent import LiveAgent
from langchain_core.messages import AIMessage
from test_navigation import FixtureTransport
from test_agent import ScriptedModel, plan_message, task


class FixtureSession(web.WebSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_configured = True
        self.app.service._navigation = AmapClient(key="test-only", transport=FixtureTransport())
        self.app.agent_factory = lambda service: LiveAgent(service, model=ScriptedModel(responses=[
            plan_message([task('navigate_to_destination', {'destination': '北京南站'})]),
        ]))


class FixtureHandler(web.Handler):
    def do_GET(self):
        if self.path == "/" and self.local_request():
            html = (web.FRONTEND / "index.html").read_text(encoding="utf-8")
            html = html.replace("从一句指令，了解 NLU 与 Agent 如何协作。", "界面测试：固定响应，未连接高德，不是实际路线结果。")
            body = html.encode("utf-8")
            self.headers_out(200, "text/html; charset=utf-8", len(body))
            self.wfile.write(body)
        else:
            super().do_GET()


if __name__ == "__main__":
    web.WebSession = FixtureSession
    web.Handler = FixtureHandler
    server = web.CabinWebServer(("127.0.0.1", 8766), nlu_name="rules")
    print("界面测试（固定响应）：http://127.0.0.1:8766/；未连接高德。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
