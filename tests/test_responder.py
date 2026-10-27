"""只读语言整理与故障隔离；使用独立的回答模型，不访问供应商。"""

import json
import unittest
from unittest.mock import Mock

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage

from cockpit.agent import LiveAgent
from cockpit.app import CabinApp
from cockpit.responder import Responder, observations
from cockpit.safety import MOTION_PRESETS
from cockpit.tools import ToolService
from test_agent import ScriptedModel, plan_message, task


class ResponderTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.service = ToolService(lambda event, data:self.events.append((event, data)))

    def agent(self, tasks, reply):
        model = FakeMessagesListChatModel(responses=[reply]) if isinstance(reply, AIMessage) else reply
        return LiveAgent(self.service, model=ScriptedModel(responses=[plan_message(tasks)]), response_model=model)

    def test_natural_answer_replaces_task_log_but_keeps_receipts_and_history(self):
        text = '空调调到24度，并打开前照灯'
        answer = '模拟空调已调到24℃，前照灯也已打开。'
        agent = self.agent([task('set_temperature', {'temperature':24}, 'a'), task('set_headlights', {'enabled':True}, 'b')], AIMessage(content=answer))
        reply, history = agent.run(text, [])
        self.assertEqual(reply, answer)
        self.assertNotIn('本轮2项任务', reply)
        self.assertEqual(agent.last_execution['answer']['source'], 'model')
        self.assertEqual(agent.last_execution['counts']['succeeded'], 2)
        self.assertIn('真实执行记录', history[-1].content)
        self.assertEqual(self.service.state.temperature_c, 24)
        self.assertEqual(sum(event == 'tool.call' for event, _ in self.events), 2)

    def test_answer_model_gets_user_question_current_observations_and_short_history(self):
        response = Mock()
        response.invoke.return_value = AIMessage(content='模拟车辆当前电量25%。')
        agent = self.agent([task('get_vehicle_state', {})], response)
        agent.run('看看电量', [HumanMessage(content='我关注的是电量')])
        messages = response.invoke.call_args.args[0]
        self.assertEqual(messages[1].content, '我关注的是电量')
        payload = json.loads(messages[-1].content)
        self.assertEqual(payload['user_request'], '看看电量')
        self.assertEqual(payload['current_results'][0]['result']['data']['battery_percent'], 25)
        self.assertFalse(response.bind_tools.called)

    def test_partial_failure_can_be_answered_naturally_without_changing_status(self):
        self.service.set_driving_state(**MOTION_PRESETS['driving'])
        answer = '模拟空调已调到24℃。车辆正在行驶，前照灯没有关闭；请安全停车后重新下达关灯指令。'
        agent = self.agent([task('set_temperature', {'temperature':24}, 'a'), task('set_headlights', {'enabled':False}, 'b')], AIMessage(content=answer))
        reply, _ = agent.run('空调调到24度，并关闭前照灯', [])
        self.assertEqual(reply, answer)
        self.assertEqual([row['status'] for row in agent.last_execution['tasks']], ['succeeded','blocked'])
        self.assertEqual(agent.last_execution['answer']['source'], 'model')

    def test_answer_timeout_keeps_completed_side_effects_without_retry_or_false_planning_error(self):
        response = Mock()
        response.invoke.side_effect = TimeoutError('private-secret-response')
        agent = self.agent([task('set_temperature', {'temperature':24})], response)
        app = CabinApp('live')
        app.service, app.agent = self.service, agent
        reply = app.handle('空调调到24度，并查看状态')
        self.assertIn('已设为24', reply)
        self.assertNotIn('本轮未开始', reply)
        self.assertNotIn('private-secret', reply + str(self.events))
        self.assertEqual(sum(event == 'tool.call' for event, _ in self.events), 1)
        self.assertFalse(any(event == 'agent.error' for event, _ in self.events))
        self.assertEqual(app.last_execution['tasks'][0]['status'], 'succeeded')
        self.assertEqual(app.last_execution['answer']['source'], 'fallback')

    def test_answer_tool_call_is_rejected_and_never_executed(self):
        unsafe = AIMessage(content='', tool_calls=[{'name':'set_temperature','args':{'temperature':30},'id':'write'}])
        agent = self.agent([task('get_vehicle_state', {})], unsafe)
        reply, _ = agent.run('查看模拟车辆状态', [])
        self.assertIn('回答整理暂不可用', reply)
        self.assertEqual(self.service.state.temperature_c, 26)
        self.assertEqual(sum(event == 'tool.call' for event, _ in self.events), 1)

    def test_invented_number_is_rejected(self):
        agent = self.agent([task('get_vehicle_state', {})], AIMessage(content='模拟车辆当前电量99%。'))
        reply, _ = agent.run('查电量', [])
        self.assertIn('电量25', reply)
        self.assertNotIn('99', reply)
        self.assertEqual(agent.last_execution['answer']['source'], 'fallback')

    def test_real_route_time_differences_are_available_for_comparison(self):
        answer = '按模拟方案，节能路线65分钟，高速48分钟，多用17分钟；节能路线到达还剩13%，满足10%预留。'
        agent = self.agent([task('get_route_options', {'destination':'机场'})], AIMessage(content=answer))
        reply, _ = agent.run('去机场比较路线', [])
        self.assertEqual(reply, answer)
        self.assertEqual(agent.last_execution['answer']['source'], 'model')

    def test_false_all_success_and_omitted_failure_are_rejected(self):
        self.service.set_driving_state(**MOTION_PRESETS['driving'])
        for content in ['全部完成，模拟前照灯已关闭。', '模拟前照灯已关闭。']:
            agent = self.agent([task('set_headlights', {'enabled':False})], AIMessage(content=content))
            reply, _ = agent.run('关闭前照灯', [])
            self.assertIn('已拦截', reply)
            self.assertEqual(agent.last_execution['answer']['source'], 'fallback')

    def test_empty_structured_and_tool_log_responses_fall_back(self):
        for content in ['', '{"tasks":[]}', '调用get_vehicle_state成功，电量25%。']:
            agent = self.agent([task('get_vehicle_state', {})], AIMessage(content=content))
            agent.run('查电量', [])
            self.assertEqual(agent.last_execution['answer']['source'], 'fallback')

    def test_reasoning_blocks_are_not_returned_logged_or_remembered(self):
        agent = self.agent([task('get_vehicle_state', {})], AIMessage(content=[
            {'type':'reasoning','reasoning':'hidden-answer-reasoning'}, {'type':'text','text':'模拟车辆当前电量25%。'}]))
        reply, history = agent.run('查电量', [])
        self.assertEqual(reply, '模拟车辆当前电量25%。')
        self.assertNotIn('hidden-answer-reasoning', reply + str(history) + str(self.events))

    def test_clarification_does_not_start_answer_call(self):
        response = Mock()
        agent = LiveAgent(self.service, model=ScriptedModel(responses=[plan_message([], '需要具体温度。')]), response_model=response)
        agent.run('有点热', [])
        response.invoke.assert_not_called()

    def test_range_reply_must_disclose_default_assumption_and_reserve(self):
        from cockpit.responder import validate_reply
        row = {'tool':'estimate_driving_range', 'status':'succeeded', 'result':{'data':{
            'basis':'default_assumption','capacity_source':'teaching_default'}}}
        for reply in ['还能跑50公里。', '保留10%电量估算约50公里。']:
            with self.assertRaises(ValueError):
                validate_reply(reply, '50 10', [row])
        validate_reply('按默认假设，保留10%电量估算约50公里。', '50 10', [row])

    def test_navigation_reply_cannot_claim_vehicle_has_arrived(self):
        from cockpit.responder import validate_reply
        row = {'tool':'navigate_to_destination','status':'succeeded','result':{}}
        with self.assertRaises(ValueError):
            validate_reply('已到达北京南站。', '', [row])


if __name__ == '__main__':
    unittest.main()
