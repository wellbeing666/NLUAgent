"""显式联调配置的真实模型；不是自动单元测试，也不发送真实个人位置。"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cockpit.app import CabinApp
from cockpit.safety import MOTION_PRESETS


def main():
    app = CabinApp('live')
    report = {'scope':'真实模型规划＋模拟车辆工具；未调用高德，无真实车控', 'cases':[]}
    cases = [
        ('parked', '把空调调到24度，并打开前照灯，然后查询车辆状态'),
        ('driving', '关闭前照灯，并座椅调到60%，同时空调调到23度'),
        ('parked', '查询车辆电量，如果电量低于30%，查询机场高速沿途的模拟充电站'),
        ('parked', '再查询一遍刚才那条路线的模拟充电站'),
    ]
    for motion, text in cases:
        app.service.set_driving_state(**MOTION_PRESETS[motion])
        reply = app.handle(text)
        execution = app.last_execution
        report['cases'].append({'input':text, 'motion':motion, 'reply':reply, 'execution':execution})
        print(json.dumps({'input':text, 'counts':execution['counts'] if execution else None, 'reply':reply}, ensure_ascii=False))
    (ROOT / 'reports' / 'planner-live-smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    # 验证工具真实回执，不仅检查回答文字。
    first, second, third, fourth = [case['execution'] for case in report['cases']]
    assert all([first, second, third, fourth]), '至少一轮未形成可执行计划，详见报告。'
    assert first['counts']['succeeded'] == 3, '停车多意图未全部完成。'
    assert second['counts']['blocked'] == 2 and second['counts']['succeeded'] == 1, '行驶安全分支未符合预期。'
    assert any(r['condition'] and r['status'] == 'succeeded' for r in third['tasks']), '条件读取分支未执行。'
    assert any(r['tool'] == 'find_chargers' and r['status'] == 'succeeded' for r in fourth['tasks']), '上下文追问未完成。'
    assert app.service.state.temperature_c == 23 and app.service.state.headlights_on is True
    assert app.service.state.seat_position_percent == 50
    print('真实模型四轮联调通过；报告保存至 reports/planner-live-smoke.json。')


if __name__ == '__main__':
    main()
