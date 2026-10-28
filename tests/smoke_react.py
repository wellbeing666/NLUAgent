"""显式真实模型联调：真实规划/动态调用/重规划＋固定地图和临时虚构记忆。"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from cockpit.app import CabinApp
from cockpit.driving_memory import DrivingMemory
from cockpit.navigation import AmapClient
from cockpit.tools import _TASK_ID
from cockpit.task_worker import error
from test_react_replanner import WestFixture


def main():
    with tempfile.TemporaryDirectory() as directory:
        app=CabinApp('live',memory=DrivingMemory(Path(directory)/'memory.sqlite3'))
        app.service.set_battery(battery_percent=25,capacity_kwh=60)
        app.service.record_driving_trip(distance_km=50,energy_kwh=8.5,record_id=str(uuid4()))
        app.service.set_vehicle_location({'name':'北京天安门（模拟）','location':'116.397428,39.909230'})
        app.service._navigation=AmapClient(key='fixture-no-real-amap',transport=WestFixture())
        report={'scope':'真实LLM的Planner/ReAct/Replanner/Responder；地图固定响应、人工故障注入、临时虚构记忆，不请求高德。','cases':[]}
        def ask(text):
            reply=app.handle(text)
            report['cases'].append({'input':text,'reply':reply,'execution':app.last_execution})
            print(reply,flush=True)
            (ROOT/'reports'/'react-replanner-live-smoke.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
            return app.last_execution
        first=ask('把空调调到24度，并分析当前电量去机场是否需要充电，给出建议')
        assert first and all(row['status']=='succeeded' for row in first['tasks']), '初次业务目标执行失败，详见报告。'
        assert any(len([step for step in row['steps'] if step['called']])>=2 for row in first['tasks']), '没有观察到多步ReAct。'
        original=app.service.execute
        fault_task=[None]
        def stale_once_goal(name,args,**kwargs):
            if name in {'navigate_to_destination','plan_navigation_route'}:
                if fault_task[0] is None: fault_task[0]=_TASK_ID.get()
                if _TASK_ID.get()==fault_task[0]: return error('NAVIGATION_STALE','[故障注入]查询期间位置版本已变化，请刷新前置状态后重新规划。')
            return original(name,args,**kwargs)
        with patch.object(app.service,'execute',side_effect=stale_once_goal):
            second=ask('把空调调到23度，并导航到北京南站')
        assert second and second['run']['replans']>=1 and len(second['run']['versions'])>=2, '未完成全局重规划。'
        assert all(row['status']=='succeeded' for row in second['tasks']), '重规划后仍有未完成目标。'
        assert len(second['run']['write_ledger'])==1 and app.service.state.temperature_c==23, '控制重复或未核验。'
        third=ask('改去北京西站，刚才的空调保持不动')
        assert third and all(row['kind']!='vehicle_control' for row in third['tasks']), '下一轮修改重放了车控。'
        assert any(row['kind']=='navigation' for row in third['tasks'])
        assert all(row['status']=='succeeded' for row in third['tasks']) and app.service.state.temperature_c==23
        print('真实模型动态工具调用、故障重规划和下一轮目标修改联调通过。',flush=True)


if __name__=='__main__': main()
