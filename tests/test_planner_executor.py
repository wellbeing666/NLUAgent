"""结构化计划、并行执行、依赖/条件和授权回归；不访问外部模型或地图。"""

import json
import threading
import unittest
from copy import deepcopy
from unittest.mock import patch
from urllib.parse import urlsplit, parse_qs

from langchain_core.messages import AIMessage

from cockpit.app import CabinApp
from cockpit.agent import LiveAgent
from cockpit.executor import Executor, TaskRecord, condition_matches
from cockpit.planner import Planner, PlanningError, TaskPlan, Condition, validate_plan
from cockpit.safety import MOTION_PRESETS
from cockpit.semantic_nlu import NLUDecision
from cockpit.nlu import Intent
from cockpit.navigation import AmapClient
from cockpit.tools import ToolService, build_langchain_tools
from test_agent import ScriptedModel, plan_message, task
from test_navigation import FixtureTransport, ORIGIN, DESTINATION


def plan(tasks):
    return TaskPlan.model_validate({'tasks': tasks, 'clarification': ''})


class PlanValidationTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.service = ToolService(lambda e, d: self.events.append((e, d)))
        self.catalog = {tool.name: tool for tool in build_langchain_tools(self.service)}

    def reject(self, tasks):
        with self.assertRaises(PlanningError):
            validate_plan({'tasks': tasks, 'clarification': ''}, self.catalog)
        self.assertFalse(any(e == 'tool.call' for e, _ in self.events))

    def test_rejects_unknown_tools_and_model_supplied_permission(self):
        self.reject([task('set_driving_state', {'speed_kph': 0})])
        self.reject([task('set_temperature', {'temperature': 24, 'allow_write': True})])
        value = task('set_temperature', {'temperature': 24})
        value['authorization'] = True
        self.reject([value])

    def test_rejects_duplicate_missing_self_and_cyclic_dependencies(self):
        self.reject([task('get_vehicle_state', {}, 'a'), task('get_vehicle_state', {}, 'a')])
        self.reject([task('get_vehicle_state', {}, 'a', ['missing'])])
        self.reject([task('get_vehicle_state', {}, 'a', ['a'])])
        self.reject([task('get_vehicle_state', {}, 'a', ['b']), task('get_vehicle_state', {}, 'b', ['a'])])

    def test_references_require_declared_dependency_and_safe_path(self):
        for ref in ['a.data', 'a.data.routes.0.route_id']:
            self.reject([task('get_vehicle_state', {}, 'a'), task('find_chargers', {'route_id': {'$ref': ref}}, 'b')])
        self.reject([task('get_vehicle_state', {}, 'a'), task('find_chargers', {'route_id': {'$ref': 'a.__class__'}}, 'b', ['a'])])
        self.reject([task('get_vehicle_state', {}, 'a'), task('find_chargers', {'route_id': {'$ref': 'a.data', 'extra': 1}}, 'b', ['a'])])

    def test_condition_requires_dependency_and_bounded_operator(self):
        self.reject([task('get_vehicle_state', {}, 'a'), task('find_chargers', {}, 'b', condition={'ref':'a.data.battery_percent','operator':'lt','value':30})])
        self.reject([task('get_vehicle_state', {}, 'a'), task('find_chargers', {}, 'b', ['a'], {'ref':'a.data.battery_percent','operator':'eval','value':30})])

    def test_whole_plan_is_validated_before_any_side_effect(self):
        model = ScriptedModel(responses=[plan_message([task('set_temperature', {'temperature': 24}, 'a'), task('unknown', {}, 'b')])])
        with self.assertRaises(PlanningError):
            Planner(self.service, model).plan('空调调到24度，并执行未知工具', [])
        self.assertEqual(self.service.state.temperature_c, 26)

    def test_planner_direct_business_tool_call_is_not_executed(self):
        model = ScriptedModel(responses=[AIMessage(content='', tool_calls=[{'name':'set_temperature','args':{'temperature':24},'id':'x'}])])
        with self.assertRaises(PlanningError):
            Planner(self.service, model).plan('空调调到24度', [])
        self.assertEqual(self.service.state.temperature_c, 26)

    def test_invalid_json_and_nan_are_rejected_without_raw_model_content(self):
        for content in ['secret-provider-response', '{"tasks":[{"id":"a","tool":"set_temperature","arguments":{"temperature":NaN}}]}']:
            model = ScriptedModel(responses=[AIMessage(content=content)])
            with self.assertRaises(PlanningError) as error:
                Planner(self.service, model).plan('测试', [])
            self.assertNotIn('secret-provider', str(error.exception))

    def test_clarification_contains_no_executable_tasks(self):
        model = ScriptedModel(responses=[plan_message([], '你希望设置多少度？')])
        app = CabinApp('live', agent_factory=lambda service: LiveAgent(service, model=model))
        reply = app.handle('有点热，帮我调一下空调')
        self.assertIn('未执行任何任务', reply)
        self.assertEqual(app.last_execution['tasks'], [])


class SchedulerTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.service = ToolService(lambda e, d: self.events.append((e, d)))
        self.executor = Executor(self.service, ScriptedModel(responses=[]))

    def test_multi_controls_authorized_from_original_clauses_and_query_depends(self):
        tasks = [task('set_temperature', {'temperature':24}, 'a'), task('set_headlights', {'enabled':True}, 'b'),
                 task('get_vehicle_state', {}, 'c', ['a','b'])]
        records = self.executor.execute(plan(tasks), '先把空调调到24度，并打开前照灯，然后查询车辆状态')
        self.assertEqual([r['status'] for r in records], ['succeeded'] * 3)
        self.assertEqual(records[2]['result']['data']['temperature_c'], 24)
        self.assertTrue(records[2]['result']['data']['headlights_on'])
        self.assertTrue(all(r['result']['safety']['allowed'] for r in records[:2]))
        calls = [d for e, d in self.events if e == 'tool.call']
        self.assertTrue(all('task_id' in call for call in calls))

    def test_unmatched_or_fabricated_parameter_is_blocked(self):
        records = self.executor.execute(plan([task('set_temperature', {'temperature': 30})]), '空调调到24度并查询车辆状态')
        self.assertEqual(records[0]['status'], 'blocked')
        self.assertEqual(records[0]['error']['code'], 'WRITE_NOT_AUTHORIZED')
        self.assertEqual(self.service.state.temperature_c, 26)

    def test_denial_quote_question_and_conditional_scope_do_not_grant_writes(self):
        samples = ['不要打开前照灯和关闭雨刮器', '他说“打开前照灯”，然后查询状态',
                   '打开前照灯，然后查询状态好吗？', '如果车辆已经停车，就打开前照灯',
                   '当电量低于30%时，空调调到24度', '停车后打开前照灯']
        for text in samples:
            record = self.executor.execute(plan([task('set_headlights', {'enabled':True})]), text)[0]
            self.assertEqual(record['status'], 'blocked')
        self.assertFalse(self.service.state.headlights_on)

    def test_one_original_command_cannot_authorize_duplicate_writes(self):
        records = self.executor.execute(plan([task('set_temperature', {'temperature':24}, 'a'), task('set_temperature', {'temperature':24}, 'b')]), '空调调到24度')
        self.assertEqual([r['status'] for r in records], ['succeeded', 'blocked'])

    def test_blocked_dependency_is_skipped_but_independent_task_continues(self):
        self.service.set_driving_state(**MOTION_PRESETS['driving'])
        tasks = [task('set_headlights', {'enabled':False}, 'a'), task('get_vehicle_state', {}, 'b', ['a']),
                 task('set_temperature', {'temperature':24}, 'c')]
        records = self.executor.execute(plan(tasks), '关闭前照灯，然后查询车辆状态，同时空调调到24度')
        self.assertEqual([r['status'] for r in records], ['blocked','skipped','succeeded'])
        self.assertEqual(records[1]['attempts'], 0)
        self.assertEqual(self.service.state.temperature_c, 24)

    def test_condition_uses_true_previous_result_and_false_branch_never_calls_tool(self):
        for threshold, expected in [(30, 'succeeded'), (20, 'skipped')]:
            records = self.executor.execute(plan([task('get_vehicle_state', {}, 'a'),
                task('find_chargers', {'route_id':'airport_fast'}, 'b', ['a'], {'ref':'a.data.battery_percent','operator':'lt','value':threshold})]), '查询电量并按结果查询充电站')
            self.assertEqual(records[1]['status'], expected)
            self.assertEqual(records[1]['attempts'], int(expected == 'succeeded'))

    def test_result_refs_use_actual_values_and_missing_data_fails_without_guess(self):
        tasks = [task('get_route_options', {'destination':'机场'}, 'a'),
                 task('find_chargers', {'route_id':{'$ref':'a.data.routes.0.route_id'}}, 'b', ['a'])]
        records = self.executor.execute(plan(tasks), '查路线然后查充电站')
        self.assertEqual(records[1]['arguments']['route_id'], 'airport_fast')
        tasks[1]['inputs']['route_id'] = {'$ref':'a.data.routes.99.route_id'}
        records = self.executor.execute(plan(tasks), '查路线然后查充电站')
        self.assertEqual(records[1]['status'], 'failed')
        self.assertEqual(records[1]['attempts'], 0)

    def test_wrong_type_condition_does_not_coerce_or_execute(self):
        records = self.executor.execute(plan([task('get_vehicle_state', {}, 'a'), task('find_chargers', {}, 'b', ['a'],
            {'ref':'a.data.motion.status','operator':'ne','value':0})]), '查询状态和充电站')
        self.assertEqual(records[1]['status'], 'failed')
        self.assertEqual(records[1]['attempts'], 0)

    def test_same_actuator_has_order_constraint_but_failure_not_causal(self):
        tasks = [task('set_temperature', {'temperature':35}, 'a'), task('set_temperature', {'temperature':24}, 'b')]
        records = self.executor.execute(plan(tasks), '空调调到35度，并空调调到24度')
        self.assertEqual([r['status'] for r in records], ['failed', 'succeeded'])
        self.assertEqual(records[1]['order_only'], ['a'])
        self.assertEqual(self.service.state.temperature_c, 24)

    def test_array_order_need_not_be_topological(self):
        tasks = [task('set_temperature', {'temperature':24}, 'b', ['a']),
                 task('set_temperature', {'temperature':23}, 'a')]
        records = self.executor.execute(plan(tasks), '空调调到23度，然后空调调到24度')
        self.assertEqual([r['status'] for r in records], ['succeeded', 'succeeded'])
        self.assertEqual(self.service.state.temperature_c, 24)

    def test_single_battery_query_is_not_misread_as_conjunction(self):
        from cockpit.task_auth import split_clauses
        self.assertEqual(split_clauses('现在还有多少电'), ['现在还有多少电'])

    def test_independent_real_tool_searches_actually_overlap(self):
        barrier = threading.Barrier(2)
        transport = FixtureTransport()
        def simultaneous(url):
            if urlsplit(url).path == '/v3/place/text':
                barrier.wait(timeout=2)
            return transport(url)
        self.service._navigation = AmapClient(key='test', transport=simultaneous)
        tasks = [task('search_navigation_places', {'keywords':'南站','city':'北京'}, 'a'),
                 task('search_navigation_places', {'keywords':'机场','city':'北京'}, 'b')]
        records = self.executor.execute(plan(tasks), '独立查询南站和机场')
        self.assertEqual([r['status'] for r in records], ['succeeded'] * 2)

    def test_failed_search_retries_once_without_retrying_other_failures(self):
        transport = FixtureTransport()
        calls = []
        def flaky(url):
            calls.append(urlsplit(url).path)
            if len(calls) == 1:
                from cockpit.navigation import NavigationError
                raise NavigationError('AMAP_NETWORK', '测试连接失败')
            return transport(url)
        self.service._navigation = AmapClient(key='test', transport=flaky)
        records = self.executor.execute(plan([task('search_navigation_places', {'keywords':'南站','city':'北京'})]), '查询南站')
        self.assertEqual(records[0]['status'], 'succeeded')
        self.assertEqual(records[0]['attempts'], 2)
        records = self.executor.execute(plan([task('set_temperature', {'temperature':35})]), '空调调到35度')
        self.assertEqual(records[0]['attempts'], 1)

    def test_readback_failure_is_not_reported_as_success_and_not_retried(self):
        original = self.service.execute
        def unverified(name, args, **kwargs):
            result = original(name, args, **kwargs)
            result['data']['verified'] = False
            return result
        with patch.object(self.service, 'execute', side_effect=unverified):
            records = self.executor.execute(plan([task('set_temperature', {'temperature':24})]), '空调调到24度')
        self.assertEqual(records[0]['status'], 'failed')
        self.assertEqual(records[0]['error']['code'], 'VERIFICATION_FAILED')
        self.assertEqual(records[0]['attempts'], 1)

    def test_unexpected_tool_error_is_redacted_and_other_task_continues(self):
        original = self.service.execute
        def broken(name, args, **kwargs):
            if name == 'get_vehicle_location':
                raise RuntimeError('private-secret-provider-details')
            return original(name, args, **kwargs)
        with patch.object(self.service, 'execute', side_effect=broken):
            records = self.executor.execute(plan([task('get_vehicle_location', {}, 'a'), task('get_vehicle_state', {}, 'b')]), '查状态')
        self.assertEqual([r['status'] for r in records], ['failed','succeeded'])
        self.assertEqual(records[0]['attempts'], 1)
        self.assertNotIn('private-secret', str(records))

    def test_zero_dispatch_budget_does_not_start_or_retry_tasks(self):
        executor = Executor(self.service, ScriptedModel(responses=[]), budget_seconds=0)
        records = executor.execute(plan([task('set_temperature', {'temperature':24})]), '空调调到24度')
        self.assertEqual(records[0]['status'], 'skipped')
        self.assertEqual(self.service.state.temperature_c, 26)

    def test_state_change_during_network_call_does_not_commit_stale_navigation(self):
        started, release = threading.Event(), threading.Event()
        transport = FixtureTransport()
        def slow(url):
            if urlsplit(url).path == '/v5/direction/driving':
                started.set()
                release.wait(timeout=2)
            return transport(url)
        self.service._navigation = AmapClient(key='test', transport=slow)
        result = []
        worker = threading.Thread(target=lambda: result.append(self.service.execute('plan_navigation_route', {'origin': ORIGIN, 'destination':DESTINATION})))
        worker.start()
        try:
            self.assertTrue(started.wait(2))
            self.service.set_vehicle_location({'name':'新模拟起点','location':'116.5,39.9'})
        finally:
            release.set()
            worker.join(3)
        self.assertEqual(result[0]['error']['code'], 'NAVIGATION_STALE')
        self.assertIsNone(self.service.navigation_plan)

    def test_nlu_partial_semantic_acceptance_cannot_drop_another_intent(self):
        class PartialNLU:
            def recognize(self, text):
                return NLUDecision(action='execute', source='semantic_nlu', intent=Intent('query_vehicle','get_vehicle_state',{}))
        model = ScriptedModel(responses=[plan_message([task('get_vehicle_state', {}, 'a'), task('set_temperature', {'temperature':24}, 'b')])])
        from cockpit.agent import LiveAgent
        app = CabinApp('live', nlu=PartialNLU(), agent_factory=lambda service: LiveAgent(service, model=model))
        app.handle('查看车辆状态，并空调调到24度')
        self.assertEqual(len(app.last_execution['tasks']), 2)
        self.assertEqual(app.service.state.temperature_c, 24)


if __name__ == '__main__':
    unittest.main()
