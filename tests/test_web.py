"""本地HTTP接入测试；使用规则NLU，不请求云端或下载权重。"""

import http.cookiejar
import json
import threading
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest.mock import patch
from pathlib import Path
from uuid import uuid4
from cockpit.driving_memory import DrivingMemory

from cockpit.semantic_nlu import NLURouter, NLUStartupError
from web import CabinWebServer
from cockpit.navigation import AmapClient
from test_navigation import ORIGIN, DESTINATION, FixtureTransport


class WebTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.memory_directory = tempfile.TemporaryDirectory()
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
        cls.memory_directory.cleanup()

    def setUp(self):
        with self.server.sessions_lock:
            self.server.sessions.clear()  # 测试间隔离，避免新增用例触发真实服务的32会话上限。
        self.server.memory = DrivingMemory(Path(self.memory_directory.name) / f'{uuid4()}.sqlite3')
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
        for path in ["/", "/app.css", "/app.js", "/favicon.svg", "/navigation.js", "/navigation.css"]:
            status, _, headers = self.call(path)
            self.assertEqual(status, 200)
            self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        for path in ["/.env", "/.git/config", "/../.env", "/models/config.json"]:
            self.assertEqual(self.call(path)[0], 404)

    def test_occupied_port_cannot_silently_serve_old_process(self):
        with self.assertRaises(OSError):
            CabinWebServer(self.server.server_address, nlu_name='rules')

    def test_temperature_chat_and_event_trace(self):
        status, result, headers = self.call("/api/chat", {"text": "空调调到24度"})
        self.assertEqual(status, 200)
        self.assertIn("HttpOnly", headers["Set-Cookie"])
        self.assertIn("SameSite=Strict", headers["Set-Cookie"])
        self.assertEqual(result["vehicle"]["temperature_c"], 24)
        self.assertEqual(result["last"]["nlu"]["source"], "rule_nlu")
        self.assertEqual([e["event"] for e in result["last"]["events"]],
                         ["nlu.result", "route", "tool.call", "safety.check", "tool.result"])
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

    def test_driving_memory_persists_and_updates_range_across_browser_sessions(self):
        _, initial, _ = self.call('/api/session')
        self.assertEqual(initial['driving_memory']['estimate']['estimated_range_km'], 50)
        record = {'distance_km':50, 'energy_kwh':8.5, 'record_id':str(uuid4())}
        status, saved, _ = self.call('/api/driving-memory/trips', record)
        self.assertEqual(status, 200)
        self.assertEqual(saved['session']['driving_memory']['estimate']['estimated_range_km'], 52.9)
        self.assertTrue(self.call('/api/driving-memory/trips', record)[1]['result']['duplicate'])
        other = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.assertEqual(self.call('/api/session', client=other)[1]['driving_memory']['total_records'], 1)
        self.call('/api/vehicle/battery', {'battery_percent':40,'capacity_kwh':80})
        self.assertEqual(self.call('/api/session', client=other)[1]['vehicle']['battery_percent'], 25)
        self.assertEqual(self.call('/api/session', client=other)[1]['driving_memory']['capacity_kwh'], 80)
        self.call('/api/reset', {})
        _, reset, _ = self.call('/api/session')
        self.assertEqual(reset['driving_memory']['total_records'], 1)
        self.assertEqual(reset['driving_memory']['capacity_kwh'], 80)
        self.assertEqual(reset['vehicle']['battery_percent'], 25)

    def test_driving_memory_endpoints_validate_and_reject_cross_origin(self):
        record = {'distance_km':50, 'energy_kwh':8.5, 'record_id':str(uuid4())}
        self.assertEqual(self.call('/api/driving-memory/trips', record, headers={'Origin':'https://example.com'})[0], 403)
        for values in [{**record,'vehicle_id':'other'}, {**record,'energy_kwh':-1}, {**record,'record_id':'bad'}]:
            self.assertEqual(self.call('/api/driving-memory/trips', values)[0], 400)
        for values in [{'battery_percent':True,'capacity_kwh':60}, {'battery_percent':50,'capacity_kwh':0}]:
            self.assertEqual(self.call('/api/vehicle/battery', values)[0], 400)
        self.assertEqual(self.call('/api/session')[1]['driving_memory']['total_records'], 0)

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

    def navigation_session(self):
        self.call("/api/session")
        token = next(cookie.value for cookie in self.jar if cookie.name == "cabin_session")
        session = self.server.sessions[token]
        session.app.service._navigation = AmapClient(key="private-test-key", transport=FixtureTransport())
        return session

    def test_navigation_search_and_route_without_model(self):
        session = self.navigation_session()
        status, result, _ = self.call('/api/navigation/search', {"keywords": "南站", "city": "北京"})
        self.assertEqual(status, 200)
        self.assertEqual(len(result['result']['data']['places']), 1)
        _, result, _ = self.call('/api/navigation/route', {"origin": ORIGIN, "destination": DESTINATION})
        self.assertTrue(result['result']['ok'])
        self.assertFalse(result['result']['simulated'])
        self.assertIsNone(session.app.agent)
        self.assertEqual(result['session']['last']['events'][0]['event'], 'tool.call')
        self.assertEqual(result['session']['navigation']['plan']['origin']['name'], ORIGIN['name'])
        self.assertNotIn('private-test-key', json.dumps(result))
        self.assertNotIn('geometry', json.dumps(result))

    def test_navigation_map_is_bound_to_session_and_current_route(self):
        self.navigation_session()
        _, result, _ = self.call('/api/navigation/route', {"origin": ORIGIN, "destination": DESTINATION})
        route_id = result['result']['data']['routes'][0]['route_id']
        self.assertEqual(self.call('/api/navigation/map', {"route_id": route_id})[0], 200)
        other = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.assertEqual(self.call('/api/navigation/map', {"route_id": route_id}, client=other)[0], 400)
        self.call('/api/reset', {})
        self.assertEqual(self.call('/api/navigation/map', {"route_id": route_id})[0], 400)

    def test_navigation_missing_key_is_explicit_without_mock_success(self):
        session = self.navigation_session()
        session.app.service._navigation = AmapClient(key="", transport=lambda url: self.fail('不应调用高德'))
        _, result, _ = self.call('/api/navigation/search', {"keywords": "南站", "city": "北京"})
        self.assertFalse(result['result']['ok'])
        self.assertEqual(result['result']['error']['code'], 'AMAP_NOT_CONFIGURED')
        self.assertFalse(result['result']['simulated'])

    def test_map_failure_does_not_erase_route(self):
        session = self.navigation_session()
        _, result, _ = self.call('/api/navigation/route', {"origin": ORIGIN, "destination": DESTINATION})
        route_id = result['result']['data']['routes'][0]['route_id']
        session.app.service._navigation._transport = lambda url: (b'{"status":"0","infocode":"10009"}', 'application/json')
        self.assertEqual(self.call('/api/navigation/map', {"route_id": route_id})[0], 400)
        self.assertIsNotNone(self.call('/api/session')[1]['navigation']['plan'])

    def test_navigation_rejects_cross_origin_and_untrusted_url(self):
        self.assertEqual(self.call('/api/navigation/search', {}, headers={"Origin": "https://example.com"})[0], 403)
        self.navigation_session()
        _, result, _ = self.call('/api/navigation/route', {"origin": ORIGIN, "destination": DESTINATION, "url": "https://example.com"})
        self.assertEqual(result['result']['error']['code'], 'INVALID_ARGUMENT')
        self.assertEqual(self.call('/api/navigation/map', {"route_id": "x", "key": "x"})[0], 400)

    def test_manual_vehicle_position_is_validated_isolated_and_reset(self):
        status, result, _ = self.call('/api/vehicle/location', {'name': '北京天安门（模拟）', 'location': '116.397428,39.909230'})
        self.assertEqual(status, 200)
        self.assertTrue(result['vehicle']['location']['simulated'])
        self.assertEqual(result['vehicle']['location']['source'], 'manual_simulation')
        self.assertEqual(self.call('/api/vehicle/location', {'name': 'x', 'location': 'NaN,39'})[0], 400)
        self.assertEqual(self.call('/api/vehicle/location', {'name': 'x', 'location': '1,2', 'source': 'gps'})[0], 400)
        other = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.assertIsNone(self.call('/api/session', client=other)[1]['vehicle']['location'])
        self.assertEqual(self.call('/api/vehicle/location', {}, headers={'Origin': 'https://example.com'})[0], 403)
        self.call('/api/reset', {})
        self.assertIsNone(self.call('/api/session')[1]['vehicle']['location'])

    def test_chat_auto_navigation_updates_frontend_snapshot(self):
        from cockpit.agent import LiveAgent
        from langchain_core.messages import AIMessage
        from test_agent import ScriptedModel, plan_message, task
        session = self.navigation_session()
        self.call('/api/vehicle/location', {'name': '北京天安门（模拟）', 'location': '116.397428,39.909230'})
        self.call('/api/settings', {'mode': 'live', 'nlu': 'rules'})
        model = ScriptedModel(responses=[
            plan_message([task('navigate_to_destination', {'destination': '北京南站'})]),
        ])
        session.app.agent = LiveAgent(session.app.service, model=model)
        status, result, _ = self.call('/api/chat', {'text': '导航到北京南站'})
        self.assertEqual(status, 200)
        self.assertTrue(result['navigation']['plan']['automatic'])
        self.assertEqual(result['navigation']['plan']['destination']['id'], ORIGIN['id'])
        self.assertEqual(result['navigation']['plan']['selection']['candidate_count'], 2)
        self.assertIn('navigation.match', [e['event'] for e in result['last']['events']])
        self.assertEqual(len(result['messages']), 2)

    def test_offline_navigation_chat_never_calls_provider(self):
        session = self.navigation_session()
        session.app.service._navigation = AmapClient(key='', transport=lambda url: self.fail('离线聊天不应联网'))
        _, result, _ = self.call('/api/chat', {'text': '导航到北京南站'})
        self.assertIn('live', result['last']['reply'])
        self.assertIsNone(result['navigation']['plan'])

    def test_multi_intent_chat_exposes_actual_task_results_and_clears_on_local_turn(self):
        from cockpit.agent import LiveAgent
        from test_agent import ScriptedModel, plan_message, task
        session = self.navigation_session()
        self.call('/api/settings', {'mode': 'live', 'nlu': 'rules'})
        model = ScriptedModel(responses=[plan_message([
            task('set_temperature', {'temperature':24}, 'a'),
            task('set_headlights', {'enabled':True}, 'b'),
            task('get_vehicle_state', {}, 'c', ['a','b']),
        ])])
        session.app.agent = LiveAgent(session.app.service, model=model)
        status, result, _ = self.call('/api/chat', {'text':'空调调到24度，并打开前照灯，然后查询车辆状态'})
        self.assertEqual(status, 200)
        self.assertEqual(result['execution']['counts']['succeeded'], 3)
        self.assertEqual([r['status'] for r in result['execution']['tasks']], ['succeeded'] * 3)
        self.assertEqual(result['execution']['tasks'][2]['result']['data']['temperature_c'], 24)
        self.assertTrue(result['vehicle']['headlights_on'])
        self.assertIn('executor.completed', [e['event'] for e in result['last']['events']])
        _, result, _ = self.call('/api/chat', {'text':'查看车辆状态'})
        self.assertIsNone(result['execution'])
        self.assertEqual(len(model.seen_messages), 1)

    def test_multi_intent_safety_block_skips_dependent_but_continues_independent(self):
        from cockpit.agent import LiveAgent
        from test_agent import ScriptedModel, plan_message, task
        session = self.navigation_session()
        self.call('/api/settings', {'mode':'live', 'nlu':'rules'})
        self.call('/api/vehicle/driving-state', {'speed_kph':40, 'gear':'D', 'parking_brake':False})
        session.app.agent = LiveAgent(session.app.service, model=ScriptedModel(responses=[plan_message([
            task('adjust_seat', {'position_percent':60}, 'a'),
            task('get_vehicle_state', {}, 'b', ['a']),
            task('set_temperature', {'temperature':24}, 'c'),
        ])]))
        _, result, _ = self.call('/api/chat', {'text':'座椅调到60%，然后查询车辆状态，同时空调调到24度'})
        self.assertEqual([r['status'] for r in result['execution']['tasks']], ['blocked','skipped','succeeded'])
        self.assertEqual(result['execution']['tasks'][1]['attempts'], 0)
        self.assertEqual(result['vehicle']['seat_position_percent'], 50)
        self.assertEqual(result['vehicle']['temperature_c'], 24)
        self.assertTrue(result['last']['safety_blocked'])

    def test_react_recovery_snapshot_keeps_versions_steps_and_successful_overall_status(self):
        from cockpit.agent import LiveAgent
        from cockpit.task_worker import error
        from test_react_replanner import RouterModel,business,plan
        from test_agent import task
        from copy import deepcopy
        session=self.navigation_session()
        self.call('/api/settings',{'mode':'live','nlu':'rules'})
        self.call('/api/vehicle/location',{'name':'北京天安门（模拟）','location':'116.397428,39.909230'})
        nav=business('nav','navigation',{'destination':'北京南站'})
        newer=deepcopy(nav); newer['id']='nav2'
        repair={'action':'replace','reason':'刷新后重算','tasks':[newer],'replacements':{'nav':'nav2'},'clarification':''}
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'temp'),nav)],repairs=[repair])
        session.app.agent=LiveAgent(session.app.service,model=model,max_workers=1)
        original=session.app.service.execute
        once=[True]
        def stale(name,args,**kwargs):
            if name=='navigate_to_destination' and once[0]:
                once[0]=False
                return error('NAVIGATION_STALE','模拟位置版本已变化')
            return original(name,args,**kwargs)
        with patch.object(session.app.service,'execute',side_effect=stale):
            status,result,_=self.call('/api/chat',{'text':'空调调到24度，并导航到北京南站'})
        self.assertEqual(status,200)
        self.assertEqual(result['execution']['counts']['succeeded'],2)
        self.assertEqual(len(result['execution']['run']['versions']),2)
        self.assertEqual(result['execution']['run']['audit'][0]['superseded_by'],'nav2')
        self.assertFalse(result['last']['rejected'])
        self.assertFalse(result['messages'][-1]['rejected'])
        self.assertTrue(all(row['steps'] for row in result['execution']['tasks']))

    def test_next_turn_revision_keeps_old_controls_out_of_new_execution(self):
        from cockpit.agent import LiveAgent
        from test_react_replanner import RouterModel,business,plan,WestFixture
        from test_agent import task
        session=self.navigation_session()
        session.app.service._navigation=AmapClient(key='fixture',transport=WestFixture())
        self.call('/api/settings',{'mode':'live','nlu':'rules'})
        self.call('/api/vehicle/location',{'name':'北京天安门（模拟）','location':'116.397428,39.909230'})
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'temp'),business('nav','navigation',{'destination':'北京南站'}))],
            revisions=[{'action':'revise','reason':'用户换目的地','plan':plan(business('navnew','navigation',{'destination':'北京西站'}))}])
        session.app.agent=LiveAgent(session.app.service,model=model)
        self.call('/api/chat',{'text':'空调调到24度，并导航到北京南站'})
        _,result,_=self.call('/api/chat',{'text':'改去北京西站'})
        self.assertEqual(result['execution']['counts']['succeeded'],1)
        self.assertEqual(result['execution']['run']['write_ledger'],[])
        self.assertEqual(result['vehicle']['temperature_c'],24)
        self.assertEqual(result['navigation']['plan']['destination']['name'],'北京西站')

    def test_driving_state_blocks_local_write_and_preserves_actuator(self):
        self.call('/api/chat', {'text': '打开前照灯'})
        status, snapshot, _ = self.call('/api/vehicle/driving-state', {'speed_kph': 40, 'gear': 'D', 'parking_brake': False})
        self.assertEqual(status, 200)
        self.assertEqual(snapshot['vehicle']['motion']['status'], 'moving')
        _, result, _ = self.call('/api/chat', {'text': '关闭前照灯'})
        self.assertTrue(result['last']['safety_blocked'])
        self.assertTrue(result['messages'][-1]['safety_blocked'])
        self.assertTrue(result['vehicle']['headlights_on'])
        self.assertEqual(result['last']['events'][-1]['data']['result']['error']['code'], 'SAFETY_BLOCKED')
        self.assertIn('safety.check', [e['event'] for e in result['last']['events']])
        _, result, _ = self.call('/api/vehicle/driving-state', {'speed_kph': 0, 'gear': 'P', 'parking_brake': True})
        self.assertTrue(result['vehicle']['headlights_on'])  # 不自动重试之前的关闭操作。
        _, result, _ = self.call('/api/chat', {'text': '关闭前照灯'})
        self.assertFalse(result['vehicle']['headlights_on'])
        self.assertFalse(result['last']['safety_blocked'])

    def test_driving_telemetry_endpoint_rejects_invalid_payload_and_origin(self):
        base = {'speed_kph': 0, 'gear': 'P', 'parking_brake': True}
        values = [{}, {**base, 'speed_kph': True}, {**base, 'speed_kph': float('nan')},
                  {**base, 'gear': []}, {**base, 'parking_brake': 'true'}, {**base, 'allow_write': True}]
        for data in values:
            self.assertEqual(self.call('/api/vehicle/driving-state', data)[0], 400)
        self.assertEqual(self.call('/api/vehicle/driving-state', base, headers={'Origin': 'https://example.com'})[0], 403)
        self.assertEqual(self.call('/api/session')[1]['vehicle']['motion']['status'], 'parked')

    def test_zero_speed_in_d_and_unknown_are_not_safe_parking(self):
        for data in [{'speed_kph': 0, 'gear': 'D', 'parking_brake': False},
                     {'speed_kph': None, 'gear': None, 'parking_brake': None}]:
            self.call('/api/vehicle/driving-state', data)
            _, result, _ = self.call('/api/chat', {'text': '座椅调到60%'})
            self.assertTrue(result['last']['safety_blocked'])
            self.assertEqual(result['vehicle']['seat_position_percent'], 50)

    def test_driving_state_is_per_browser_session_and_resets(self):
        self.call('/api/vehicle/driving-state', {'speed_kph': 40, 'gear': 'D', 'parking_brake': False})
        other = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
        self.assertEqual(self.call('/api/session', client=other)[1]['vehicle']['motion']['status'], 'parked')
        _, snapshot, _ = self.call('/api/reset', {})
        self.assertEqual(snapshot['vehicle']['motion']['status'], 'parked')
        self.assertFalse(snapshot['vehicle']['headlights_on'])
        self.assertEqual(snapshot['vehicle']['seat_position_percent'], 50)
        self.assertEqual(snapshot['vehicle']['wiper_mode'], 'off')

    def test_driving_state_update_cannot_race_with_active_chat(self):
        self.call('/api/session')
        token = next(cookie.value for cookie in self.jar if cookie.name == 'cabin_session')
        session = self.server.sessions[token]
        with session.lock:
            self.assertEqual(self.call('/api/vehicle/driving-state', {'speed_kph': 40, 'gear': 'D', 'parking_brake': False})[0], 409)
        self.assertEqual(session.app.service.state.speed_kph, 0)


if __name__ == "__main__":
    unittest.main()
