"""本地网页入口：python web.py；只使用Python标准库，复用CabinApp。"""

import argparse
import json
import math
import secrets
import threading
from datetime import datetime
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter, monotonic
from urllib.parse import urlsplit

from cockpit.app import CabinApp
from cockpit.semantic_nlu import NLURouter, NLUStartupError

FRONTEND = Path(__file__).resolve().parent / "frontend"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/favicon.svg": ("favicon.svg", "image/svg+xml"),
}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


class WebSession:
    """每个浏览器使用独立车辆与历史；模型本身由NLURouter缓存共享。"""

    def __init__(self, mode, nlu_name, router):
        self.lock = threading.Lock()
        self.event_lock = threading.Lock()
        self.events = []
        self.messages = []
        self.last = None
        self.nlu_name = nlu_name
        self.updated = monotonic()
        self.app = CabinApp(mode, trace=self.trace, nlu=router)

    def trace(self, event, payload):
        with self.event_lock:
            if len(self.events) < 200:
                self.events.append({"event": event, "data": json_safe(payload)})

    def snapshot(self):
        state = self.app.service.state
        return {"mode": self.app.mode, "nlu": self.nlu_name,
                "vehicle": {"battery_percent": state.battery_percent, "temperature_c": state.temperature_c},
                "messages": self.messages, "last": self.last}

    def chat(self, text):
        self.events = []
        start = perf_counter()
        reply = self.app.handle("查看车辆状态" if text == "/state" else text)
        decision = next((e["data"] for e in self.events if e["event"] == "nlu.result"), None)
        route = next((e["data"]["path"] for e in self.events if e["event"] == "route"), "nlu_guard")
        error = any(e["event"] == "agent.error" for e in self.events)
        rejected = any(e["event"] == "tool.result" and not e["data"]["result"]["ok"] for e in self.events)
        now = datetime.now().astimezone().strftime("%H:%M")
        self.last = {"reply": reply, "nlu": decision, "route": route, "events": list(self.events),
                     "duration_ms": round((perf_counter() - start) * 1000, 2),
                     "error": error, "rejected": rejected}
        self.messages.extend([
            {"role": "user", "content": text, "at": now},
            {"role": "assistant", "content": reply, "at": now, "route": route,
             "error": error, "rejected": rejected},
        ])
        self.messages = self.messages[-20:]
        return self.snapshot()

    def reset(self):
        self.app.reset()
        self.messages.clear()
        self.events.clear()
        self.last = None
        return self.snapshot()


class CabinWebServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, mode="offline", nlu_name="hybrid", router_factory=None):
        self.mode = mode
        self.nlu_name = nlu_name
        self.sessions = {}
        self.sessions_lock = threading.Lock()
        self.router_lock = threading.Lock()
        self.router_factory = router_factory or (lambda name: NLURouter.hybrid() if name == "hybrid" else NLURouter())
        self.get_router(nlu_name)  # 显式hybrid启动失败时停止，不静默改用云端或规则。
        super().__init__(address, Handler)

    def get_router(self, name):
        with self.router_lock:
            return self.router_factory(name)

    def session(self, cookie_header):
        cookies = SimpleCookie()
        try:
            cookies.load(cookie_header or "")
        except CookieError:
            pass
        token = cookies["cabin_session"].value if "cabin_session" in cookies else ""
        with self.sessions_lock:
            if token in self.sessions:
                session = self.sessions[token]
                session.updated = monotonic()
                return session, None
            for key, session in list(self.sessions.items()):
                if monotonic() - session.updated > 3600 and not session.lock.locked():
                    del self.sessions[key]
            if len(self.sessions) >= 32:
                raise ValueError("本地会话数量已达上限，请稍后再试或重启服务。")
            token = secrets.token_urlsafe(32)
            session = WebSession(self.mode, self.nlu_name, self.get_router(self.nlu_name))
            self.sessions[token] = session
            return session, token


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def headers_out(self, status, content_type, size, cookie=None):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(size))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self'; script-src 'self'; img-src 'self' data:; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        if cookie:
            self.send_header("Set-Cookie", f"cabin_session={cookie}; HttpOnly; SameSite=Strict; Path=/")
        self.end_headers()

    def respond(self, data, status=200, cookie=None):
        body = json.dumps(json_safe(data), ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.headers_out(status, "application/json; charset=utf-8", len(body), cookie)
        self.wfile.write(body)

    def local_request(self):
        port = self.server.server_address[1]
        host = self.headers.get("Host", "")
        if host not in {f"127.0.0.1:{port}", f"localhost:{port}"}:
            return False
        origin = self.headers.get("Origin")
        return origin is None or origin == f"http://{host}"

    def do_GET(self):
        if not self.local_request():
            self.respond({"error": "只接受当前本地页面的请求。"}, 403)
            return
        path = urlsplit(self.path).path
        if path in ASSETS:
            name, content_type = ASSETS[path]
            body = (FRONTEND / name).read_bytes()
            self.headers_out(200, content_type, len(body))
            self.wfile.write(body)
        elif path == "/api/session":
            self.run_session("state", {})
        else:
            self.respond({"error": "页面或接口不存在。"}, 404)

    def do_POST(self):
        # 小请求先读完再返回错误，避免Windows因未消费请求体而重置连接。
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 16384:
                raise ValueError("请求内容为空或过长。")
            raw = self.rfile.read(length)
        except ValueError as exc:
            self.respond({"error": str(exc)}, 400)
            return
        if not self.local_request():
            self.respond({"error": "只接受当前本地页面的请求。"}, 403)
            return
        path = urlsplit(self.path).path
        actions = {"/api/chat": "chat", "/api/settings": "settings", "/api/reset": "reset"}
        if path not in actions:
            self.respond({"error": "接口不存在。"}, 404)
            return
        try:
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("请使用JSON请求。")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("请求必须是JSON对象。")
            self.run_session(actions[path], data)
        except (ValueError, UnicodeDecodeError) as exc:
            self.respond({"error": str(exc)}, 400)

    def run_session(self, action, data):
        try:
            session, cookie = self.server.session(self.headers.get("Cookie"))
        except ValueError as exc:
            self.respond({"error": str(exc)}, 503)
            return
        if not session.lock.acquire(blocking=False):
            self.respond({"error": "当前会话正在处理请求，请稍后刷新状态。"}, 409, cookie)
            return
        try:
            if action == "chat":
                text = data.get("text")
                if not isinstance(text, str) or not text.strip() or len(text) > 2000:
                    raise ValueError("请输入1～2000个字符的指令。")
                result = session.chat(text.strip())
            elif action == "settings":
                mode, nlu = data.get("mode"), data.get("nlu")
                if mode not in {"offline", "live"} or nlu not in {"rules", "hybrid"}:
                    raise ValueError("不支持的运行模式。")
                router = self.server.get_router(nlu)
                session.app.mode, session.app.nlu, session.nlu_name = mode, router, nlu
                result = session.snapshot()
            elif action == "reset":
                result = session.reset()
            else:
                result = session.snapshot()
            session.updated = monotonic()
            self.respond(result, cookie=cookie)
        except (ValueError, NLUStartupError) as exc:
            self.respond({"error": str(exc)}, 400, cookie)
        except Exception:
            # 不向浏览器传递供应商异常正文、配置或本地堆栈。
            self.respond({"error": "本轮处理失败，请刷新状态后再试；详细配置可在本地检查。"}, 500, cookie)
        finally:
            session.lock.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description="智能座舱本地网页")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--nlu", choices=["rules", "hybrid"], default="hybrid")
    args = parser.parse_args(argv)
    try:
        server = CabinWebServer(("127.0.0.1", args.port), args.mode, args.nlu)
    except (NLUStartupError, OSError, ValueError) as exc:
        print(f"网页启动失败：{exc}", flush=True)
        return 2
    print(f"座舱网页已启动：http://127.0.0.1:{server.server_address[1]}", flush=True)
    print("仅监听本机；Ctrl+C停止。车辆和路线均为模拟数据。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
