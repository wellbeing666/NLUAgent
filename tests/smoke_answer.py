"""显式真实模型回答联调：只查询模拟机场/模拟车控，不调用高德。"""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cockpit.app import CabinApp
from cockpit.safety import MOTION_PRESETS
from cockpit.driving_memory import DrivingMemory


def main():
    directory = tempfile.TemporaryDirectory()
    try:
        run_cases(DrivingMemory(Path(directory.name) / 'memory.sqlite3'))
    finally:
        directory.cleanup()


def run_cases(memory):
    app = CabinApp('live', memory=memory)
    report = {'scope':'真实规划和回答模型＋模拟业务工具，无高德调用', 'cases':[]}
    for motion, text in [
        ('parked','电量只有25%，去机场要不要充电？帮我比较方案，给个建议'),
        ('driving','关闭前照灯，并座椅调到60%，同时空调调到23度'),
    ]:
        app.service.set_driving_state(**MOTION_PRESETS[motion])
        reply = app.handle(text)
        report['cases'].append({'input':text,'reply':reply,'execution':app.last_execution})
        print(reply, flush=True)
    (ROOT / 'reports' / 'answer-live-smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    for case in report['cases']:
        assert case['execution'] and case['execution']['answer']['source'] == 'model', '回答未通过模型整理，详见报告。'
        assert '本轮' not in case['reply'] and 'set_temperature' not in case['reply']
    second = report['cases'][1]['execution']
    assert second['counts']['blocked'] == 2 and second['counts']['succeeded'] == 1
    assert app.service.state.temperature_c == 23 and app.service.state.seat_position_percent == 50
    print('真实模型自然回答和混合成败反馈通过，报告保存至 reports/answer-live-smoke.json。')


if __name__ == '__main__':
    main()
