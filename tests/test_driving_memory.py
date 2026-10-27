"""真实SQLite文件、确定性续航公式、重启/去重与Agent接线；不调用模型或高德。"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

from cockpit.app import CabinApp
from cockpit.driving_memory import DrivingMemory, MemoryUnavailable
from cockpit.tools import ToolService, build_langchain_tools
from cockpit.semantic_nlu import NLURouter
from cockpit.agent import LiveAgent
from test_agent import ScriptedModel, plan_message, task


class DrivingMemoryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'memory.sqlite3'
        self.memory = DrivingMemory(self.path)
        self.service = ToolService(memory=self.memory)

    def record(self, distance=50, energy=8.5, token=None, vehicle='demo_vehicle'):
        return self.memory.record_trip(vehicle, distance, energy, token or str(uuid4()))

    def estimate(self):
        result = self.service.execute('estimate_driving_range', {})
        self.assertTrue(result['ok'], result)
        return result['data']

    def test_empty_history_is_explicit_assumption_with_reserve(self):
        data = self.estimate()
        self.assertEqual(data['basis'], 'default_assumption')
        self.assertEqual(data['sample_count'], 0)
        self.assertEqual(data['estimated_range_km'], 50)
        self.assertEqual(data['range_to_empty_km'], 83.3)
        self.assertIn('非实测', ''.join(data['warnings']))

    def test_uses_distance_weighted_consumption_instead_of_mean_of_trip_rates(self):
        self.record(10, 1)   # 10kWh/100km
        self.record(90, 18)  # 20kWh/100km
        data = self.estimate()
        self.assertEqual(data['consumption_kwh_per_100km'], 19)
        self.assertEqual(data['sample_count'], 2)
        self.assertEqual(data['historical_distance_km'], 100)
        self.assertEqual(data['estimated_range_km'], 47.4)

    def test_reloads_memory_and_capacity_from_new_store_and_service(self):
        self.record()
        self.service.set_battery(battery_percent=40, capacity_kwh=80)
        restarted = ToolService(memory=DrivingMemory(self.path))
        data = restarted.execute('estimate_driving_range', {})['data']
        self.assertEqual(data['sample_count'], 1)
        self.assertEqual(data['capacity_kwh'], 80)
        self.assertEqual(data['capacity_source'], 'manual_entry')
        self.assertEqual(data['battery_percent'], 25)  # SOC不冒充持久化的最新遥测。
        self.assertEqual(data['estimated_range_km'], 70.6)

    def test_reset_keeps_long_term_history_and_capacity(self):
        app = CabinApp('offline', memory=self.memory)
        app.service.record_driving_trip(distance_km=50, energy_kwh=8.5, record_id=str(uuid4()))
        app.service.set_battery(battery_percent=50, capacity_kwh=75)
        app.handle('当前电量还能跑多远')
        app.reset()
        result = app.service.driving_memory_snapshot()
        self.assertEqual(result['total_records'], 1)
        self.assertEqual(result['capacity_kwh'], 75)
        self.assertEqual(result['estimate']['battery_percent'], 25)
        self.assertEqual(app.turns, [])

    def test_different_vehicles_do_not_share_history_or_capacity(self):
        self.record(vehicle='car_a')
        self.memory.set_capacity('car_a', 80)
        self.assertEqual(self.memory.snapshot('car_a')['sample_count'], 1)
        self.assertEqual(self.memory.snapshot('car_b')['sample_count'], 0)
        self.assertEqual(self.memory.snapshot('car_b')['capacity_kwh'], 60)

    def test_idempotent_submission_survives_restart_and_rejects_changed_payload(self):
        token = str(uuid4())
        self.assertFalse(self.record(token=token)['duplicate'])
        duplicate = DrivingMemory(self.path).record_trip('demo_vehicle', 50, 8.5, token)
        self.assertTrue(duplicate['duplicate'])
        self.assertEqual(self.memory.snapshot()['total_records'], 1)
        with self.assertRaises(ValueError):
            self.record(50, 9, token)
        self.assertEqual(self.memory.snapshot()['total_records'], 1)

    def test_concurrent_duplicate_from_two_stores_is_only_saved_once(self):
        token, results = str(uuid4()), []
        def save():
            results.append(DrivingMemory(self.path).record_trip('demo_vehicle', 50, 8.5, token))
        workers = [threading.Thread(target=save) for _ in range(2)]
        for worker in workers: worker.start()
        for worker in workers: worker.join(6)
        self.assertEqual(len(results), 2)
        self.assertEqual(sum(result['duplicate'] for result in results), 1)
        self.assertEqual(self.memory.snapshot()['total_records'], 1)

    def test_invalid_samples_never_enter_memory(self):
        for distance, energy in [(0, 1), (-1, 1), (True, 8), ('50', 8), (50, False),
                                 (50, 0), (50, -1), (50, float('nan')), (float('inf'), 1),
                                 (10, 20), (10, 0.1), (10**1000, 1)]:
            with self.subTest(distance=str(distance)[:20], energy=energy):
                with self.assertRaises(ValueError):
                    self.record(distance, energy)
        self.assertEqual(self.memory.snapshot()['total_records'], 0)
        with self.assertRaises(ValueError): self.record(token='not-an-id')

    def test_latest_100_records_define_window_without_deleting_older_memory(self):
        self.record(100, 50)
        for _ in range(100): self.record(10, 1)
        data = self.estimate()
        self.assertEqual(data['total_records'], 101)
        self.assertEqual(data['sample_count'], 100)
        self.assertEqual(data['consumption_kwh_per_100km'], 10)
        self.assertEqual(len(self.memory.snapshot()['records']), 10)

    def test_battery_limits_reserve_and_zero_do_not_produce_negative_range(self):
        for percent in [0, 5, 10]:
            self.service.set_battery(battery_percent=percent, capacity_kwh=60)
            self.assertEqual(self.estimate()['estimated_range_km'], 0)
        self.service.set_battery(battery_percent=100, capacity_kwh=60)
        self.assertEqual(self.estimate()['estimated_range_km'], 300)

    def test_invalid_battery_does_not_mutate_soc_or_saved_capacity(self):
        for percent, capacity in [(101, 60), (-1, 60), (True, 60), (25, 0), (25, float('inf')), ('25', 60)]:
            with self.assertRaises(ValueError):
                self.service.set_battery(battery_percent=percent, capacity_kwh=capacity)
        self.assertEqual(self.service.state.battery_percent, 25)
        self.assertEqual(self.memory.snapshot()['capacity_kwh'], 60)

    def test_tool_rejects_model_soc_overrides_and_history_limit(self):
        for name, args in [('estimate_driving_range', {'battery_percent':99}),
                           ('estimate_driving_range', {'vehicle_id':'other'}),
                           ('get_driving_history', {'limit':True}), ('get_driving_history', {'limit':21})]:
            self.assertEqual(self.service.execute(name, args)['error']['code'], 'INVALID_ARGUMENT')

    def test_record_and_queries_never_decrement_current_battery(self):
        self.service.record_driving_trip(distance_km=50, energy_kwh=8.5, record_id=str(uuid4()))
        for _ in range(2): self.estimate()
        self.service.execute('get_route_options', {'destination':'机场'})
        self.assertEqual(self.service.state.battery_percent, 25)
        self.assertEqual(self.memory.snapshot()['total_records'], 1)

    def test_nlu_range_and_history_are_local_in_both_modes(self):
        for mode in ['offline', 'live']:
            app = CabinApp(mode, memory=self.memory, agent_factory=lambda _: self.fail('不应调用模型'))
            self.assertIn('50公里', app.handle('当前电量还能跑多远？'))
            self.assertIn('还没有', app.handle('查看历史行驶记录'))

    def test_unknown_range_expression_cannot_be_absorbed_by_two_class_vector_nlu(self):
        class ForbiddenSemantic:
            def recognize(self, text):
                raise AssertionError('不应由旧两类向量NLU处理续航请求')
        decision = NLURouter(ForbiddenSemantic()).recognize('根据历史能耗看看还可以跑多少公里')
        self.assertEqual(decision.action, 'fallback')
        self.assertEqual(decision.reason, 'range_or_memory_request')

    def test_planner_executor_reads_true_memory_and_never_has_record_write_tool(self):
        self.record()
        model = ScriptedModel(responses=[plan_message([task('estimate_driving_range', {}, 'a'),
            task('get_driving_history', {'limit':3}, 'b')])])
        app = CabinApp('live', memory=self.memory, agent_factory=lambda service:LiveAgent(service, model=model))
        reply = app.handle('根据历史估算剩余续航，并查看历史行驶记录')
        self.assertIn('52.9公里', reply)
        self.assertEqual([r['status'] for r in app.last_execution['tasks']], ['succeeded','succeeded'])
        self.assertNotIn('record_driving_trip', {tool.name for tool in build_langchain_tools(app.service)})

    def test_storage_error_is_explicit_and_does_not_fall_back_to_default(self):
        blocked_path = Path(self.directory.name) / 'a_file'
        blocked_path.write_text('not a directory', encoding='utf-8')
        service = ToolService(memory=DrivingMemory(blocked_path / 'memory.sqlite3'))
        result = service.execute('estimate_driving_range', {})
        self.assertFalse(result['ok'])
        self.assertEqual(result['error']['code'], 'MEMORY_UNAVAILABLE')
        with self.assertRaises(MemoryUnavailable):
            service.set_battery(battery_percent=80, capacity_kwh=60)
        self.assertEqual(service.state.battery_percent, 25)


if __name__ == '__main__':
    unittest.main()
