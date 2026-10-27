"""结构化长期记忆：SQLite保存人工录入的耗电实测值，续航由确定性公式计算。"""

import math
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from uuid import UUID

DEFAULT_DB = Path(__file__).resolve().parents[1] / 'runtime' / 'driving_memory.sqlite3'
DEFAULT_VEHICLE = 'demo_vehicle'
DEFAULT_CAPACITY_KWH = 60.0
DEFAULT_CONSUMPTION = 18.0  # 教学假设，不代表厂家参数或真实车辆测量。
WINDOW_SIZE = 100


class MemoryUnavailable(RuntimeError):
    pass


def number(value, name, low, high):
    if type(value) not in {int, float} or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f'{name}必须是{low}～{high}之间的有限数字。')
    return float(value)


class DrivingMemory:
    def __init__(self, path=DEFAULT_DB):
        self.path = Path(path)
        self._lock = Lock()

    @contextmanager
    def _db(self):
        # 短连接＋事务，支持不同ToolService/进程读取同一辆车；不把连接跨线程共享。
        with self._lock:
            connection = None
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                connection = sqlite3.connect(self.path, timeout=5)
                connection.row_factory = sqlite3.Row
                connection.execute('PRAGMA foreign_keys=ON')
                connection.executescript('''
                    CREATE TABLE IF NOT EXISTS vehicles (
                        vehicle_id TEXT PRIMARY KEY,
                        capacity_kwh REAL NOT NULL CHECK(capacity_kwh BETWEEN 5 AND 200),
                        capacity_source TEXT NOT NULL DEFAULT 'teaching_default'
                    );
                    CREATE TABLE IF NOT EXISTS driving_trips (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        vehicle_id TEXT NOT NULL REFERENCES vehicles(vehicle_id),
                        record_id TEXT NOT NULL,
                        recorded_at TEXT NOT NULL,
                        distance_km REAL NOT NULL CHECK(distance_km > 0),
                        energy_kwh REAL NOT NULL CHECK(energy_kwh > 0),
                        source TEXT NOT NULL DEFAULT 'manual_entry',
                        UNIQUE(vehicle_id, record_id)
                    );
                    CREATE INDEX IF NOT EXISTS trips_vehicle_order ON driving_trips(vehicle_id, id DESC);
                ''')
                with connection:
                    yield connection
            except (sqlite3.Error, OSError):
                raise MemoryUnavailable('本地行驶记忆无法读取或保存，请检查数据库文件权限及可用空间；未使用默认值伪装成历史估算。') from None
            finally:
                if connection is not None:
                    connection.close()

    @staticmethod
    def _vehicle(db, vehicle_id):
        if not isinstance(vehicle_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,64}', vehicle_id):
            raise ValueError('车辆记忆ID只接受1～64位字母、数字、下划线或短横线。')
        db.execute('INSERT OR IGNORE INTO vehicles(vehicle_id,capacity_kwh) VALUES (?, ?)', (vehicle_id, DEFAULT_CAPACITY_KWH))

    def set_capacity(self, vehicle_id, capacity_kwh):
        value = number(capacity_kwh, '可用电池容量(kWh)', 5, 200)
        with self._db() as db:
            self._vehicle(db, vehicle_id)
            db.execute("UPDATE vehicles SET capacity_kwh=?,capacity_source='manual_entry' WHERE vehicle_id=?", (value, vehicle_id))

    def record_trip(self, vehicle_id, distance_km, energy_kwh, record_id):
        distance = number(distance_km, '行驶里程(km)', 1, 2000)
        energy = number(energy_kwh, '实际耗电量(kWh)', 0.001, 300)
        # 教学版仅收净耗电行程，明显异常、充电混入或净回收样本需人工核对。
        consumption = energy / distance * 100
        if not 3 <= consumption <= 60:
            raise ValueError('本版仅接受百公里耗电3～60kWh的行程，请核对里程/耗电单位；不要录入包含充电或净能量回收的行程。')
        try:
            token = str(UUID(record_id)) if isinstance(record_id, str) else ''
        except (ValueError, AttributeError):
            token = ''
        if not token:
            raise ValueError('record_id必须是UUID，用于防止同一次保存被重复计入。')
        with self._db() as db:
            self._vehicle(db, vehicle_id)
            # 写事务在查重前取得，跨进程并发同一record_id也只会新增一次。
            previous = db.execute('SELECT * FROM driving_trips WHERE vehicle_id=? AND record_id=?', (vehicle_id, token)).fetchone()
            if previous:
                if previous['distance_km'] != distance or previous['energy_kwh'] != energy:
                    raise ValueError('同一record_id对应的里程或耗电已变更，请使用新的记录ID。')
                return {'saved':True, 'duplicate':True, 'record':self._row(previous)}
            db.execute('INSERT INTO driving_trips(vehicle_id,record_id,recorded_at,distance_km,energy_kwh) VALUES (?,?,?,?,?)',
                       (vehicle_id, token, datetime.now(timezone.utc).isoformat(), distance, energy))
            row = db.execute('SELECT * FROM driving_trips WHERE vehicle_id=? AND record_id=?', (vehicle_id, token)).fetchone()
            return {'saved':True, 'duplicate':False, 'record':self._row(row)}

    @staticmethod
    def _row(row):
        return {**dict(row), 'consumption_kwh_per_100km': round(row['energy_kwh'] / row['distance_km'] * 100, 4)}

    def snapshot(self, vehicle_id=DEFAULT_VEHICLE, limit=10):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('历史查询limit必须是1～20的整数。')
        with self._db() as db:
            self._vehicle(db, vehicle_id)
            vehicle = db.execute('SELECT * FROM vehicles WHERE vehicle_id=?', (vehicle_id,)).fetchone()
            total = db.execute('SELECT COUNT(*) FROM driving_trips WHERE vehicle_id=?', (vehicle_id,)).fetchone()[0]
            rows = db.execute('SELECT * FROM driving_trips WHERE vehicle_id=? ORDER BY id DESC LIMIT ?', (vehicle_id, WINDOW_SIZE)).fetchall()
        distance, energy = sum(row['distance_km'] for row in rows), sum(row['energy_kwh'] for row in rows)
        return {'vehicle_id':vehicle_id, 'capacity_kwh':vehicle['capacity_kwh'], 'capacity_source':vehicle['capacity_source'], 'total_records':total,
                'window_size':WINDOW_SIZE, 'sample_count':len(rows), 'total_distance_km':round(distance, 4),
                'total_energy_kwh':round(energy, 4),
                'consumption_kwh_per_100km':energy / distance * 100 if rows else DEFAULT_CONSUMPTION,
                'basis':'history' if rows else 'default_assumption',
                'records':[self._row(row) for row in rows[:limit]],
                'note':'按最近录入的最多100条本车记录估算；人工录入不代表车载遥测。'}


def estimate_range(memory, battery_percent, reserve_percent):
    soc = number(battery_percent, '当前电量(%)', 0, 100)
    reserve = number(reserve_percent, '预留电量(%)', 0, 100)
    rate = memory['consumption_kwh_per_100km']
    capacity = memory['capacity_kwh']
    usable = capacity * max(0, soc - reserve) / 100
    warnings = ['参考估算，未建模实时气温、路况、坡度、车速、载重及电池衰减。']
    if not memory['sample_count']:
        warnings.append('没有历史行驶记录，使用18kWh/100km教学默认能耗，非实测续航。')
    if memory['capacity_source'] == 'teaching_default':
        warnings.append('电池容量60kWh也是教学初始假设，请按本车可用容量修改。')
    if 0 < memory['sample_count'] < 3:
        warnings.append('历史少于3条，样本不足，结果仅供演示。')
    if soc <= reserve:
        warnings.append('当前电量已不高于预留值，保留预留电量可行驶距离为0。')
    return {'vehicle_id':memory['vehicle_id'], 'battery_percent':soc, 'capacity_kwh':capacity, 'capacity_source':memory['capacity_source'],
            'reserve_percent':reserve, 'available_energy_kwh':round(usable, 4),
            'estimated_range_km':round(usable / rate * 100, 1),
            'range_to_empty_km':round(capacity * soc / rate, 1),
            'consumption_kwh_per_100km':round(rate, 4), 'basis':memory['basis'],
            'sample_count':memory['sample_count'], 'total_records':memory['total_records'],
            'historical_distance_km':memory['total_distance_km'],
            'method':'distance_weighted_recent_100',
            'formula':'capacity_kwh × max(battery_percent - reserve_percent, 0) ÷ consumption_kwh_per_100km',
            'warnings':warnings, 'note':'当前SOC为本会话模拟值；历史是本机持久化的人工行程记录，不是LLM对话回忆。'}


def range_text(data):
    basis = f"最近{data['sample_count']}条历史、合计{data['historical_distance_km']:g}公里" if data['basis'] == 'history' else '教学默认能耗（无历史记录）'
    return (f"当前模拟电量{data['battery_percent']:g}%，可用电池容量{data['capacity_kwh']:g}kWh。"
            f"参考{basis}，按{data['consumption_kwh_per_100km']:g}kWh/100km估算："
            f"保留{data['reserve_percent']:g}%电量可行驶约{data['estimated_range_km']:g}公里；"
            f"理论耗尽约{data['range_to_empty_km']:g}公里（不建议以耗尽为目标）。" + ''.join(data['warnings']))


def history_text(data):
    if not data['records']:
        return '本车还没有长期行驶记录，请在行驶记忆面板或使用 /trip 里程|耗电量 手动录入。'
    return (f"本车已保存{data['total_records']}条人工行程记录，最近{data['sample_count']}条合计"
            f"{data['total_distance_km']:g}公里，里程加权能耗{data['consumption_kwh_per_100km']:.2f}kWh/100km。\n" +
            '\n'.join(f"#{row['id']}：{row['distance_km']:g}公里／{row['energy_kwh']:g}kWh" for row in data['records']))
