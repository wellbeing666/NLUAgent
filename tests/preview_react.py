"""固定模型＋地图的网页接线演示，不需要模型Key，不代表真实LLM推理。"""

import sys
from copy import deepcopy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import web
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from cockpit.agent import LiveAgent
from cockpit.navigation import AmapClient
from cockpit.tools import _TASK_ID
from cockpit.task_worker import error
from test_agent import task
from test_react_replanner import RouterModel,business,plan,WestFixture


class PreviewSession(web.WebSession):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.app.service._navigation=AmapClient(key='preview-fixture',transport=WestFixture())
        self.app.service.set_vehicle_location({'name':'北京天安门（模拟）','location':'116.397428,39.909230'})
        nav=business('nav','navigation',{'destination':'北京南站'},goal='规划到北京南站的路线')
        newer=deepcopy(nav); newer['id']='nav2'
        after=business('state','vehicle_state',depends=['nav'])
        after2=deepcopy(after); after2['depends_on']=['nav2']
        repair={'action':'replace','reason':'位置版本失效后按原目的地刷新重算','tasks':[newer,after2],
                'replacements':{'nav':'nav2'},'clarification':''}
        model=RouterModel(plans=[plan(task('set_temperature',{'temperature':24},'temp'),nav,after)],repairs=[repair])
        response=FakeMessagesListChatModel(responses=[AIMessage(content='本次为固定模型与固定地图响应演示。模拟空调已设为24℃。首次导航查询遇到位置版本变化，重新规划后已取得到北京南站的路线；这里只完成路线规划，尚未启动实时导航。')])
        self.app.agent_factory=lambda service:LiveAgent(service,model=model,response_model=response,max_workers=1)
        original=self.app.service.execute
        failed_task=[None]
        def stale(name,arguments,**options):
            if name in {'navigate_to_destination','plan_navigation_route'}:
                if failed_task[0] is None: failed_task[0]=_TASK_ID.get()
                if _TASK_ID.get()==failed_task[0]: return error('NAVIGATION_STALE','[固定故障注入]模拟位置版本已变化，请刷新后重算。')
            return original(name,arguments,**options)
        self.app.service.execute=stale


if __name__=='__main__':
    web.WebSession=PreviewSession
    raise SystemExit(web.main(['--mode','live','--nlu','rules','--port','8770','--memory-db',str(ROOT/'runtime'/'preview-react.sqlite3')]))
