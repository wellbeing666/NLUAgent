"""Planner/ReAct/Replanner行为与安全边界；固定模型按真实观察选择下一步。"""

import json
import threading
import unittest
from copy import deepcopy
from time import monotonic
from urllib.parse import urlsplit,parse_qs
from unittest.mock import patch

from langchain_core.messages import AIMessage, ToolMessage

from cockpit.agent import LiveAgent
from cockpit.executor import Executor
from cockpit.navigation import AmapClient
from cockpit.run_state import RunState, BudgetExceeded
from cockpit.safety import MOTION_PRESETS
from cockpit.task_types import TaskPlan, PlanningError
from cockpit.task_worker import error
from cockpit.tools import ToolService
from test_agent import task, tool_call
from test_navigation import FixtureTransport


class WestFixture(FixtureTransport):
    def __call__(self,url):
        parsed=urlsplit(url)
        if parsed.path in {'/v3/place/around','/v3/place/text'} and '北京西站' in parse_qs(parsed.query).get('keywords',[''])[0]:
            return json.dumps({'status':'1','pois':[{'name':'北京西站','location':'116.320000,39.890000','id':'FIXTURE_WEST'}]}).encode(),'application/json'
        return super().__call__(url)


def business(id, kind, inputs=None, depends=None, goal=None):
    return {'id':id,'kind':kind,'goal':goal or kind,'inputs':inputs or {},'depends_on':depends or [],'condition':None}


def finish(prior, primary=None):
    good = [row for row in prior if row['result'].get('ok')]
    return tool_call('finish_task',{'primary_step_id':primary or good[-1]['step_id'],
                                  'evidence_ids':[row['step_id'] for row in good]},'finish')


def default_worker(spec, prior):
    inputs, kind = spec['inputs'], spec['kind']
    good = [row for row in prior if row['result'].get('ok')]
    if prior and not prior[-1]['result'].get('ok'):
        return tool_call('request_replan',{'reason':'真实工具结果需要调整前置任务'},'repair')
    if kind == 'vehicle_control':
        return finish(prior) if good else tool_call(inputs['tool'],inputs['arguments'],'write')
    if kind == 'vehicle_state':
        return finish(prior) if good else tool_call('get_vehicle_location' if inputs.get('view')=='location' else 'get_vehicle_state',{},'read')
    if kind == 'navigation':
        if inputs.get('mode')=='search':
            return finish(prior) if good else tool_call('search_navigation_places',{key:inputs[key] for key in ['keywords','city']},'search')
        if not good: return tool_call('get_vehicle_location',{},'origin')
        if len(good)==1: return tool_call('navigate_to_destination',{'destination':inputs['destination'],'strategy':inputs.get('strategy','recommended')},'nav')
        return finish(prior)
    if kind == 'airport_advice':
        if not good: return tool_call('get_vehicle_state',{},'state')
        if len(good)==1: return tool_call('get_route_options',{'destination':'机场'},'routes')
        routes = good[1]['result']['data']['routes']
        if len(good)==2 and not routes[0]['meets_reserve']:
            return tool_call('find_chargers',{'route_id':routes[0]['route_id']},'chargers')
        return finish(prior,good[1]['step_id'])
    if kind == 'driving_range':
        return finish(prior) if good else tool_call('estimate_driving_range',{},'range')
    return finish(prior) if good else tool_call('get_driving_history',inputs,'history')


class Bound:
    def __init__(self, owner, names): self.owner,self.names=owner,names
    def invoke(self,messages): return self.owner.dispatch(self.names,messages)


class RouterModel:
    def __init__(self, plans=None, workers=None, repairs=None, revisions=None):
        self.plans=list(plans or [])
        self.workers=workers or {}
        self.repairs=list(repairs or [])
        self.revisions=list(revisions or [])
        self.calls=[]
        self.lock=threading.Lock()

    def bind_tools(self,tools,**kwargs):
        names={tool['function']['name'] if isinstance(tool,dict) else tool.name for tool in tools}
        return Bound(self,names)

    def invoke(self,messages): raise TimeoutError('该固定模型未提供自然语言回答')

    def dispatch(self,names,messages):
        with self.lock:
            self.calls.append(sorted(names))
        if 'submit_plan' in names:
            return tool_call('submit_plan',self.plans.pop(0),'plan')
        if 'submit_revision' in names:
            value=self.revisions.pop(0) if self.revisions else {'action':'new','reason':'新请求','plan':None}
            return tool_call('submit_revision',value,'revision')
        if 'submit_replan' in names:
            value=self.repairs.pop(0) if self.repairs else {'action':'stop','reason':'没有合法替代方案','tasks':[],'replacements':{},'clarification':''}
            if callable(value): value=value(json.loads(messages[-1].content))
            return tool_call('submit_replan',value,'replan')
        payload=json.loads(messages[1].content)
        spec=payload['task']
        prior=[{'name':message.name,**json.loads(message.content)} for message in messages if isinstance(message,ToolMessage)]
        handler=self.workers.get(spec['id'],default_worker)
        if isinstance(handler,list):
            with self.lock: action=handler.pop(0)
            return action(spec,prior) if callable(action) else action
        return handler(spec,prior)


def plan(*tasks): return {'tasks':list(tasks),'clarification':''}


class ReactReplannerTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.service=ToolService(lambda event,data:self.events.append((event,data)))
        self.service._navigation=AmapClient(key='fixture',transport=FixtureTransport())
        self.service.set_vehicle_location({'name':'北京天安门（模拟）','location':'116.397428,39.909230'})

    def agent(self,model,**kwargs): return LiveAgent(self.service,model=model,**kwargs)

    def test_business_goal_dynamically_selects_chargers_from_route_observation(self):
        for battery, expected in [(25,True),(80,False)]:
            self.service.state.battery_percent=battery
            model=RouterModel(plans=[plan(business('trip','airport_advice',{'view':'advice','destination':'机场'}))])
            agent=self.agent(model)
            agent.run('帮我分析机场出行',[])
            row=agent.last_execution['tasks'][0]
            self.assertEqual(row['status'],'succeeded')
            called=[step['tool'] for step in row['steps'] if step['called']]
            self.assertEqual('find_chargers' in called,expected)
            self.assertEqual(row['result']['data']['routes'][0]['arrival_battery_percent'],battery-18)

    def test_navigation_goal_observes_location_before_deciding_navigation_tool(self):
        model=RouterModel(plans=[plan(business('nav','navigation',{'destination':'北京南站'}))])
        agent=self.agent(model)
        agent.run('导航到北京南站',[])
        row=agent.last_execution['tasks'][0]
        self.assertEqual([step['tool'] for step in row['steps']],['get_vehicle_location','navigate_to_destination'])
        self.assertEqual(row['status'],'succeeded')
        self.assertEqual(row['primary_step_id'],row['steps'][-1]['id'])
        self.assertIn(row['primary_step_id'],row['evidence_ids'])

    def repair_fixture(self,change=None):
        control=task('set_temperature',{'temperature':24},'temp')
        nav=business('nav','navigation',{'destination':'北京南站'})
        after=business('after','vehicle_state',depends=['nav'])
        new=deepcopy(nav); new['id']='nav2'; new['depends_on']=['pre']
        if change: change(new)
        repaired_after=deepcopy(after); repaired_after['depends_on']=['nav2']
        repair={'action':'replace','reason':'先刷新状态再按原目标重算','tasks':[business('pre','vehicle_state'),new,repaired_after],
                'replacements':{'nav':'nav2'},'clarification':''}
        return plan(control,nav,after),repair

    def test_replanner_replaces_failed_goal_preserves_success_and_rewrites_dependencies(self):
        initial,repair=self.repair_fixture()
        model=RouterModel(plans=[initial],repairs=[repair])
        original=self.service.execute
        first=[True]
        def stale_once(name,args,**kwargs):
            if name=='navigate_to_destination' and first[0]:
                first[0]=False
                return error('NAVIGATION_STALE','位置已更新，需要重新读取')
            return original(name,args,**kwargs)
        agent=self.agent(model,max_workers=1)
        with patch.object(self.service,'execute',side_effect=stale_once):
            reply,_=agent.run('空调调到24度，并导航到北京南站，然后查看状态',[])
        execution=agent.last_execution
        self.assertEqual(execution['counts']['succeeded'],4)
        self.assertEqual(len(execution['run']['versions']),2)
        self.assertEqual(execution['run']['audit'][0]['id'],'nav')
        self.assertEqual(execution['run']['audit'][0]['status'],'failed')
        self.assertEqual(execution['run']['audit'][0]['superseded_by'],'nav2')
        self.assertEqual(len(execution['run']['write_ledger']),1)
        self.assertEqual(sum(event=='tool.call' and data['name']=='set_temperature' for event,data in self.events),1)
        self.assertEqual(next(row for row in execution['tasks'] if row['id']=='nav2')['recovered_from'],['nav'])

    def test_replanner_cannot_change_destination_or_drop_unfinished_goal(self):
        for bad in ['destination','drop']:
            initial,repair=self.repair_fixture(lambda task:task['inputs'].update(destination='北京西站'))
            if bad=='drop': repair['tasks']=[]
            model=RouterModel(plans=[initial],repairs=[repair])
            original=self.service.execute
            def fail(name,args,**kwargs):
                return error('NAVIGATION_STALE','需要刷新') if name=='navigate_to_destination' else original(name,args,**kwargs)
            agent=self.agent(model,max_workers=1)
            with patch.object(self.service,'execute',side_effect=fail): agent.run('空调调到24度，并导航到北京南站',[])
            self.assertEqual(len(agent.last_execution['run']['versions']),1)
            self.assertEqual(agent.last_execution['run']['audit'],[])

    def test_replanner_cannot_replay_completed_control_or_add_new_write_goal(self):
        initial,repair=self.repair_fixture()
        repair['tasks'].append(task('set_temperature',{'temperature':24},'replay'))
        model=RouterModel(plans=[initial],repairs=[repair])
        original=self.service.execute
        def stale(name,args,**kwargs):
            return error('NAVIGATION_STALE','需要刷新') if name=='navigate_to_destination' else original(name,args,**kwargs)
        agent=self.agent(model,max_workers=1)
        with patch.object(self.service,'execute',side_effect=stale): agent.run('空调调到24度，并导航到北京南站',[])
        self.assertEqual(len(agent.last_execution['run']['versions']),1)
        self.assertEqual(len(agent.last_execution['run']['write_ledger']),1)

    def test_replanning_loop_stops_at_two_revisions(self):
        nav=business('nav','navigation',{'destination':'北京南站'})
        def revise(payload):
            old=payload['remaining'][0]['task']
            new=deepcopy(old); new['id']=old['id']+'x'
            return {'action':'replace','reason':'刷新后重查','tasks':[new],'replacements':{old['id']:new['id']},'clarification':''}
        model=RouterModel(plans=[plan(nav)],repairs=[revise,revise,revise])
        original=self.service.execute
        def stale(name,args,**kwargs):
            return error('NAVIGATION_STALE','需要刷新') if name=='navigate_to_destination' else original(name,args,**kwargs)
        agent=self.agent(model)
        with patch.object(self.service,'execute',side_effect=stale): agent.run('导航到北京南站',[])
        self.assertEqual(agent.last_execution['run']['replans'],2)
        self.assertEqual(len(agent.last_execution['run']['versions']),3)
        self.assertEqual(agent.last_execution['tasks'][0]['status'],'failed')

    def test_duplicate_write_in_same_worker_never_executes_twice(self):
        def repeat(spec,prior):
            if len(prior)<2: return tool_call('set_temperature',{'temperature':24},'write'+str(len(prior)))
            return finish(prior,prior[0]['step_id'])
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'temp'))],workers={'temp':repeat})
        agent=self.agent(model)
        agent.run('空调调到24度',[])
        row=agent.last_execution['tasks'][0]
        self.assertEqual(row['status'],'succeeded')
        self.assertEqual(row['attempts'],1)
        self.assertEqual([step['called'] for step in row['steps']],[True,False])
        self.assertEqual(len(agent.last_execution['run']['write_ledger']),1)

    def test_timeout_after_verified_write_preserves_success_without_retry(self):
        def timeout_after_write(spec,prior):
            if prior: raise TimeoutError('private-provider-token')
            return tool_call('set_temperature',{'temperature':24},'write')
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'temp'))],workers={'temp':timeout_after_write})
        agent=self.agent(model)
        reply,_=agent.run('空调调到24度',[])
        self.assertEqual(agent.last_execution['tasks'][0]['status'],'succeeded')
        self.assertTrue(agent.last_execution['tasks'][0]['auto_finalized'])
        self.assertEqual(self.service.state.temperature_c,24)
        self.assertNotIn('private-provider',reply+str(self.events))

    def test_forged_finish_and_cross_kind_tools_cannot_change_state(self):
        for kind, response in [('vehicle_control',tool_call('finish_task',{'primary_step_id':'fake','evidence_ids':['fake']},'finish')),
                               ('vehicle_state',tool_call('set_temperature',{'temperature':30},'write'))]:
            spec=task('set_temperature',{'temperature':24},'t') if kind=='vehicle_control' else business('t',kind)
            model=RouterModel(plans=[plan(spec)],workers={'t':lambda spec,prior:response})
            agent=self.agent(model,max_replans=0)
            agent.run('空调调到24度',[])
            self.assertEqual(self.service.state.temperature_c,26)
            self.assertEqual(agent.last_execution['tasks'][0]['attempts'],0)
            self.assertEqual(agent.last_execution['tasks'][0]['status'],'failed')

    def test_changed_control_parameters_are_blocked_before_tool_call(self):
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'t'))],
                          workers={'t':lambda spec,prior:tool_call('set_temperature',{'temperature':30},'write')})
        agent=self.agent(model)
        agent.run('空调调到24度',[])
        self.assertEqual(agent.last_execution['tasks'][0]['status'],'blocked')
        self.assertEqual(agent.last_execution['run']['write_ledger'],[])
        self.assertEqual(self.service.state.temperature_c,26)

    def test_duplicate_control_goals_share_one_consumable_authorization(self):
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'a'),task('set_temperature',{'temperature':24},'b'))])
        agent=self.agent(model)
        agent.run('空调调到24度',[])
        self.assertEqual([row['status'] for row in agent.last_execution['tasks']],['succeeded','blocked'])
        self.assertEqual(len(agent.last_execution['run']['write_ledger']),1)

    def test_safety_block_is_not_replanned_or_retried_after_parking(self):
        self.service.set_driving_state(**MOTION_PRESETS['driving'])
        spec=task('set_headlights',{'enabled':True},'lamp')
        model=RouterModel(plans=[plan(spec),plan(spec)])
        agent=self.agent(model)
        agent.run('打开前照灯',[])
        self.assertEqual(agent.last_execution['tasks'][0]['error']['code'],'SAFETY_BLOCKED')
        self.assertEqual(agent.last_execution['run']['replans'],0)
        self.service.set_driving_state(**MOTION_PRESETS['parked'])
        agent.run('继续刚才的操作',[])
        self.assertFalse(self.service.state.headlights_on)
        self.assertEqual(agent.last_execution['tasks'][0]['error']['code'],'WRITE_NOT_AUTHORIZED')

    def test_network_retry_allowance_is_shared_with_replanned_worker(self):
        spec=business('search','navigation',{'mode':'search','keywords':'北京南站','city':'北京'})
        newer=deepcopy(spec); newer['id']='search2'
        repairs=[{'action':'replace','reason':'原地点查询恢复','tasks':[newer],'replacements':{'search':'search2'},'clarification':''}]
        def retry_then_replan(spec,prior):
            if len(prior)<2: return tool_call('search_navigation_places',{'keywords':'北京南站','city':'北京'},'search'+str(len(prior)))
            return tool_call('request_replan',{'reason':'网络查询失败'},'repair')
        model=RouterModel(plans=[plan(spec)],repairs=repairs,workers={'search':retry_then_replan,'search2':retry_then_replan})
        original=self.service.execute
        calls=[]
        def network(name,args,**kwargs):
            if name=='search_navigation_places':
                calls.append(name); return error('AMAP_NETWORK','测试连接失败')
            return original(name,args,**kwargs)
        agent=self.agent(model)
        with patch.object(self.service,'execute',side_effect=network): agent.run('搜索北京南站',[])
        self.assertEqual(len(calls),2)
        self.assertEqual(agent.last_execution['tasks'][0]['error']['code'],'RETRY_EXHAUSTED')

    def test_model_budget_after_write_preserves_result_and_stops_new_calls(self):
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'t'))])
        agent=self.agent(model,max_model_calls=2)
        reply,_=agent.run('空调调到24度',[])
        self.assertEqual(agent.last_execution['run']['model_calls'],2)
        self.assertEqual(agent.last_execution['tasks'][0]['status'],'succeeded')
        self.assertEqual(agent.last_execution['answer']['source'],'fallback')
        self.assertEqual(self.service.state.temperature_c,24)

    def test_next_turn_destination_change_does_not_replay_control(self):
        self.service._navigation=AmapClient(key='fixture',transport=WestFixture())
        initial=plan(task('set_temperature',{'temperature':24},'temp'),business('nav','navigation',{'destination':'北京南站'}))
        revised=plan(business('newnav','navigation',{'destination':'北京西站'}))
        model=RouterModel(plans=[initial],revisions=[{'action':'revise','reason':'用户修改目的地','plan':revised}])
        agent=self.agent(model)
        agent.run('空调调到24度，并导航到北京南站',[])
        agent.run('改去北京西站',[])
        self.assertEqual([row['id'] for row in agent.last_execution['tasks']],['newnav'])
        self.assertEqual(agent.last_execution['tasks'][0]['status'],'succeeded')
        self.assertEqual(sum(event=='tool.call' and data['name']=='set_temperature' for event,data in self.events),1)
        self.assertEqual(self.service.state.temperature_c,24)
        agent.reset()
        self.assertIsNone(agent.previous_execution)

    def test_near_destination_requests_clarification_without_claiming_navigation_success(self):
        model=RouterModel(plans=[plan(business('nav','navigation',{'destination':'北京西站'}))])
        agent=self.agent(model)
        reply,_=agent.run('导航到北京西站',[])
        self.assertEqual(agent.last_execution['tasks'][0]['status'],'failed')
        self.assertEqual(agent.last_execution['tasks'][0]['error']['code'],'NAVIGATION_NEEDS_CLARIFICATION')
        self.assertEqual(agent.last_execution['run']['replans'],0)
        self.assertIn('核对具体入口',reply)

    def test_new_request_classifier_cannot_execute_its_extra_draft(self):
        old=plan(task('set_temperature',{'temperature':24},'t'))
        next_plan=plan(business('s','vehicle_state'))
        draft=plan(task('set_temperature',{'temperature':30},'injected'))
        model=RouterModel(plans=[old,next_plan],revisions=[{'action':'new','reason':'新请求附带多余草案','plan':draft}])
        agent=self.agent(model)
        agent.run('空调调到24度',[])
        agent.run('再看看车辆状态',[])
        self.assertEqual([row['id'] for row in agent.last_execution['tasks']],['s'])
        self.assertEqual(self.service.state.temperature_c,24)

    def test_invalid_revision_falls_back_to_one_fresh_plan(self):
        model=RouterModel(plans=[plan(business('a','vehicle_state')),plan(business('b','vehicle_state'))],
                          revisions=[{'action':'revise','reason':'遗漏plan','plan':None}])
        agent=self.agent(model)
        agent.run('查看状态',[])
        agent.run('再看看电量',[])
        self.assertEqual(agent.last_execution['tasks'][0]['id'],'b')
        self.assertTrue(any(event=='replanner.next_fallback' for event,_ in self.events))

    def test_instance_budget_rejects_repair_atomically(self):
        initial,repair=self.repair_fixture()
        model=RouterModel(repairs=[repair])
        from cockpit.replanner import Replanner
        run=RunState('空调调到24度，并导航到北京南站',max_instances=3)
        executor=Executor(self.service,model,max_workers=1)
        original=self.service.execute
        def stale(name,args,**kwargs):
            return error('NAVIGATION_STALE','位置已更新') if name=='navigate_to_destination' else original(name,args,**kwargs)
        with patch.object(self.service,'execute',side_effect=stale):
            rows=executor.execute(TaskPlan.model_validate(initial),run.user_text,run=run,replanner=Replanner(self.service,model))
        self.assertEqual(len(run.versions),1)
        self.assertEqual(run.audit,[])
        self.assertEqual(run.task_instances,3)

    def test_replanning_waits_for_inflight_success_before_freezing_plan(self):
        started,release,slow_done=threading.Event(),threading.Event(),threading.Event()
        nav=business('nav','navigation',{'destination':'北京南站'})
        newer=deepcopy(nav); newer['id']='nav2'
        def repair(payload):
            self.assertTrue(slow_done.is_set())
            self.assertTrue(any(row['id']=='slow' and row['status']=='succeeded' for row in payload['completed']))
            return {'action':'replace','reason':'在途目标已返回后修订','tasks':[newer],'replacements':{'nav':'nav2'},'clarification':''}
        model=RouterModel(plans=[plan(business('slow','vehicle_state'),nav)],repairs=[repair])
        original=self.service.execute
        failed=[False]
        def instrument(name,args,**kwargs):
            if name=='get_vehicle_state' and not slow_done.is_set():
                started.set()
                release.wait(3)
                result=original(name,args,**kwargs)
                slow_done.set()
                return result
            if name=='navigate_to_destination' and not failed[0]:
                self.assertTrue(started.wait(2))
                failed[0]=True
                release.set()
                return error('NAVIGATION_STALE','需要刷新')
            return original(name,args,**kwargs)
        agent=self.agent(model,max_workers=2)
        with patch.object(self.service,'execute',side_effect=instrument): agent.run('查询状态并导航到北京南站',[])
        self.assertEqual(agent.last_execution['counts']['succeeded'],2)
        self.assertEqual(len(agent.last_execution['run']['versions']),2)

    def test_elapsed_budget_keeps_verified_inflight_write_but_skips_next_goal(self):
        run=RunState('空调调到24度，然后查询状态')
        model=RouterModel()
        executor=Executor(self.service,model,max_workers=1)
        original=self.service.execute
        def expire(name,args,**kwargs):
            result=original(name,args,**kwargs)
            if name=='set_temperature': run.deadline=monotonic()-1
            return result
        with patch.object(self.service,'execute',side_effect=expire):
            rows=executor.execute(TaskPlan.model_validate(plan(task('set_temperature',{'temperature':24},'temp'),
                 business('state','vehicle_state',depends=['temp']))),run.user_text,run=run)
        self.assertEqual([row['status'] for row in rows],['succeeded','skipped'])
        self.assertEqual(rows[1]['error']['code'],'EXECUTION_BUDGET')
        before=run.model_calls
        with self.assertRaises(BudgetExceeded): run.invoke(model,[],final=True)
        self.assertEqual(run.model_calls,before)

    def test_navigation_network_failure_cannot_bypass_shared_retry_limit(self):
        model=RouterModel(plans=[plan(business('nav','navigation',{'destination':'北京南站'}))],
            workers={'nav':lambda spec,prior:tool_call('navigate_to_destination',{'destination':'北京南站'},'nav')})
        original=self.service.execute
        calls=[]
        def fail(name,args,**kwargs):
            if name=='navigate_to_destination':
                calls.append(args)
                return error('AMAP_NETWORK','固定网络故障')
            return original(name,args,**kwargs)
        agent=self.agent(model)
        with patch.object(self.service,'execute',side_effect=fail): agent.run('导航到北京南站',[])
        self.assertEqual(len(calls),2)
        self.assertEqual(agent.last_execution['tasks'][0]['error']['code'],'RETRY_EXHAUSTED')

    def test_replanner_cannot_remove_original_dependency(self):
        nav=business('nav','navigation',{'destination':'北京南站'},depends=['state'])
        new=deepcopy(nav); new['id']='nav2'; new['depends_on']=[]
        repair={'action':'replace','reason':'试图删除依赖','tasks':[new],'replacements':{'nav':'nav2'},'clarification':''}
        model=RouterModel(plans=[plan(business('state','vehicle_state'),nav)],repairs=[repair])
        original=self.service.execute
        def stale(name,args,**kwargs):
            return error('NAVIGATION_STALE','需要刷新') if name=='navigate_to_destination' else original(name,args,**kwargs)
        agent=self.agent(model)
        with patch.object(self.service,'execute',side_effect=stale): agent.run('先查询状态再导航到北京南站',[])
        self.assertEqual(len(agent.last_execution['run']['versions']),1)

    def test_worker_reasoning_never_enters_trace_or_history(self):
        def hidden(spec,prior):
            if prior: return finish(prior)
            return AIMessage(content=[{'type':'reasoning','reasoning':'hidden-react-internal-reasoning'}],
                             tool_calls=[{'name':'get_vehicle_state','args':{},'id':'read'}])
        model=RouterModel(plans=[plan(business('s','vehicle_state'))],workers={'s':hidden})
        agent=self.agent(model)
        reply,history=agent.run('看看车辆',[])
        self.assertNotIn('hidden-react-internal-reasoning',reply+str(history)+str(self.events)+str(agent.last_execution))

    def test_search_goal_cannot_write_navigation_state(self):
        response=tool_call('navigate_to_destination',{'destination':'北京南站'},'nav')
        model=RouterModel(plans=[plan(business('search','navigation',{'mode':'search','keywords':'北京南站','city':'北京'}))],
                          workers={'search':lambda spec,prior:response})
        agent=self.agent(model,max_replans=0)
        agent.run('搜索北京南站',[])
        self.assertIsNone(self.service.navigation_plan)
        self.assertFalse(any(event=='tool.call' for event,_ in self.events))

    def test_multiple_action_batch_never_executes_any_action(self):
        response=AIMessage(content='',tool_calls=[{'name':'set_temperature','args':{'temperature':24},'id':'a'},
                                                {'name':'set_headlights','args':{'enabled':True},'id':'b'}])
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'t'))],workers={'t':lambda spec,prior:response})
        agent=self.agent(model)
        agent.run('空调调到24度，并打开前照灯',[])
        self.assertEqual(self.service.state.temperature_c,26)
        self.assertFalse(self.service.state.headlights_on)
        self.assertFalse(any(event=='tool.call' for event,_ in self.events))

    def test_successful_tool_without_finish_cannot_fake_completed_read_goal(self):
        model=RouterModel(plans=[plan(business('s','vehicle_state'))],
                          workers={'s':lambda spec,prior:tool_call('finish_task',{'primary_step_id':'s_s99','evidence_ids':['s_s99']},'fake')})
        agent=self.agent(model,max_replans=0)
        agent.run('查询状态',[])
        self.assertEqual(agent.last_execution['tasks'][0]['status'],'failed')


if __name__ == '__main__':
    unittest.main()
