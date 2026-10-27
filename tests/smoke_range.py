"""显式真实模型联调；只使用临时的虚构行程，不修改用户的长期记忆。"""

import json
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cockpit.app import CabinApp
from cockpit.driving_memory import DrivingMemory


def main():
    with tempfile.TemporaryDirectory() as directory:
        app = CabinApp('live', memory=DrivingMemory(Path(directory) / 'memory.sqlite3'))
        app.service.set_battery(battery_percent=25, capacity_kwh=60)
        app.service.record_driving_trip(distance_km=50, energy_kwh=8.5, record_id=str(uuid4()))
        text = '根据历史能耗估算剩余续航，并查看历史行驶记录'
        reply = app.handle(text)
        report = {'scope':'真实模型规划＋临时虚构行程；不修改用户数据库、不调用高德',
                  'input':text, 'reply':reply, 'execution':app.last_execution}
        (ROOT / 'reports' / 'range-live-smoke.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(reply)
        assert app.last_execution, '模型未形成执行计划，详见报告。'
        assert app.last_execution['answer']['source'] == 'model', '回答未通过模型整理，详见报告。'
        tasks = {row['tool']:row for row in app.last_execution['tasks']}
        assert all(tasks.get(tool, {}).get('status') == 'succeeded' for tool in ['estimate_driving_range','get_driving_history'])
        data = tasks['estimate_driving_range']['result']['data']
        assert data['estimated_range_km'] == 52.9 and data['sample_count'] == 1
        print('真实模型续航工具与长期记忆联调通过，报告保存至 reports/range-live-smoke.json。')


if __name__ == '__main__':
    main()
