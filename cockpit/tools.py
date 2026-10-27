"""共享工具执行层。业务校验与权限由代码执行，不依赖提示词。"""

import math
import secrets
from datetime import datetime, timezone
from copy import deepcopy
from threading import Lock
from typing import Any, Callable
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar

from cockpit.data import CHARGERS, RESERVE_PERCENT, ROUTES, VehicleState
from cockpit.navigation import AmapClient, NavigationError, STRATEGIES, point
from cockpit.safety import CONTROL_TOOLS, SafetyViolation, evaluate_action, motion_snapshot, validate_driving_state
from cockpit.driving_memory import DrivingMemory, MemoryUnavailable, DEFAULT_VEHICLE, number, estimate_range

NAVIGATION_TOOLS = {"search_navigation_places", "plan_navigation_route", "navigate_to_destination"}
_TASK_ID = ContextVar('cockpit_task_id', default=None)

Trace = Callable[[str, dict[str, Any]], None]


def quiet_trace(event: str, payload: dict[str, Any]) -> None:
    pass


class ToolService:
    def __init__(self, trace: Trace = quiet_trace, navigation_factory=None, *, memory=None, vehicle_id=DEFAULT_VEHICLE):
        self.state = VehicleState()
        self._trace = trace
        self.trace = self._emit
        self._lock = Lock()
        self._navigation_init_lock = Lock()
        self._navigation_factory = navigation_factory or AmapClient
        self._navigation = None
        self.navigation_plan = None
        self._route_geometry = {}
        self._map_cache = {}
        self._navigation_revision = 0
        self.vehicle_location = None
        self.last_safety_decision = None
        self.memory = memory if memory is not None else DrivingMemory()
        self.vehicle_id = vehicle_id

    def driving_memory_snapshot(self):
        """长期记忆按车辆ID隔离；聊天reset不删除SQLite记录。"""
        with self._lock:
            history = self.memory.snapshot(self.vehicle_id)
            return {**history, 'estimate':estimate_range(history, self.state.battery_percent, RESERVE_PERCENT)}

    def set_battery(self, *, battery_percent, capacity_kwh):
        soc = number(battery_percent, '当前模拟电量(%)', 0, 100)
        with self._lock:
            self.memory.set_capacity(self.vehicle_id, capacity_kwh)
            self.state.battery_percent = soc

    def record_driving_trip(self, *, distance_km, energy_kwh, record_id):
        # 显式人工入口，不向LLM暴露，避免模型把路线规划/虚构行程写成长期事实。
        with self._lock:
            result = self.memory.record_trip(self.vehicle_id, distance_km, energy_kwh, record_id)
        self.trace('memory.recorded', result)
        return result

    def _emit(self, event, payload):
        task_id = _TASK_ID.get()
        self._trace(event, {**payload, **({'task_id': task_id} if task_id else {})})

    @contextmanager
    def task_context(self, task_id):
        token = _TASK_ID.set(task_id)
        try:
            yield
        finally:
            _TASK_ID.reset(token)

    def reset(self) -> None:
        with self._lock:
            self.state = VehicleState()
            self._clear_navigation()
            self.vehicle_location = None
            self.last_safety_decision = None

    def set_driving_state(self, *, speed_kph, gear, parking_brake):
        """模拟遥测入口：只供演示面板/CLI，不向模型提供修改行驶状态的工具。"""
        values = validate_driving_state(speed_kph, gear, parking_brake)
        with self._lock:
            self.state.speed_kph = values["speed_kph"]
            self.state.gear = values["gear"]
            self.state.parking_brake = values["parking_brake"]
            self.last_safety_decision = None
            return motion_snapshot(self.state)

    def vehicle_snapshot(self):
        with self._lock:
            return self._vehicle_snapshot()

    def _vehicle_snapshot(self):
        return {"battery_percent": self.state.battery_percent, "temperature_c": self.state.temperature_c,
                "reserve_percent": RESERVE_PERCENT, "location": deepcopy(self.vehicle_location),
                "motion": motion_snapshot(self.state), "headlights_on": self.state.headlights_on,
                "seat_position_percent": self.state.seat_position_percent, "wiper_mode": self.state.wiper_mode,
                "last_safety_decision": deepcopy(self.last_safety_decision)}

    def _check_safety(self, name, arguments):
        decision = evaluate_action(name, arguments, self.state)
        self.last_safety_decision = decision.to_dict()
        self.trace("safety.check", decision.to_dict())
        if not decision.allowed:
            raise SafetyViolation(decision)
        return decision.to_dict()

    def set_vehicle_location(self, location):
        """只能由应用位置设置接口调用，不向LLM开放修改车辆位置的工具。"""
        value = point(location)
        with self._lock:
            self.vehicle_location = {**value, "source": "manual_simulation", "simulated": True,
                                     "coordinate_system": "GCJ-02",
                                     "updated_at": datetime.now(timezone.utc).isoformat()}
            self._clear_navigation()
            return deepcopy(self.vehicle_location)

    def _get_vehicle_location(self):
        if self.vehicle_location is None:
            raise NavigationError("LOCATION_NOT_SET", "请先在车辆状态面板设置模拟车辆位置，再说“导航到北京南站”。")
        return deepcopy(self.vehicle_location)

    def _clear_navigation(self):
        self._navigation_revision += 1
        self.navigation_plan = None
        self._route_geometry.clear()
        self._map_cache.clear()

    def _navigation_client(self):
        with self._navigation_init_lock:
            if self._navigation is None:
                self._navigation = self._navigation_factory()
            return self._navigation

    def navigation_map(self, route_id):
        with self._lock:
            if not isinstance(route_id, str) or route_id not in self._route_geometry or not self.navigation_plan:
                raise ValueError("路线已失效，请先重新规划路线。")
            if route_id in self._map_cache:
                return deepcopy(self._map_cache[route_id])
            revision = self._navigation_revision
            plan, geometry = deepcopy(self.navigation_plan), deepcopy(self._route_geometry[route_id])
        result = self._navigation_client().static_map(plan, geometry)
        with self._lock:
            if revision != self._navigation_revision:
                raise ValueError('路线已更新，旧地图结果未采用。')
            self._map_cache[route_id] = result
            return deepcopy(result)

    def execute(self, name: str, arguments: dict[str, Any], *, allow_write: bool = False) -> dict:
        # 网络查询不持有车辆状态锁；车控检查与写入仍在同一个临界区内。
        with (nullcontext() if name in NAVIGATION_TOOLS else self._lock):
            simulated = name not in NAVIGATION_TOOLS
            self.trace("tool.call", {"name": name, "arguments": arguments})
            try:
                data = self._dispatch(name, arguments, allow_write)
                result = {"ok": True, "simulated": simulated, "data": data}
                if name in CONTROL_TOOLS:
                    result["safety"] = deepcopy(self.last_safety_decision)
            except SafetyViolation as exc:
                result = self._error("SAFETY_BLOCKED", str(exc))
                result["safety"] = exc.decision.to_dict()
            except NavigationError as exc:
                result = self._error(exc.code, str(exc), simulated=simulated)
            except MemoryUnavailable as exc:
                result = self._error('MEMORY_UNAVAILABLE', str(exc))
            except (ValueError, TypeError) as exc:
                result = self._error("INVALID_ARGUMENT", str(exc), simulated=simulated)
            except PermissionError:
                result = self._error(
                    "WRITE_NOT_AUTHORIZED", "没有本地明确车控命令的写授权。请单独输入完整指令，例如：空调调到24度、打开前照灯。"
                )
            self.trace("tool.result", {"name": name, "result": result})
            return result

    @staticmethod
    def _error(code: str, message: str, simulated=True) -> dict:
        return {"ok": False, "simulated": simulated, "error": {"code": code, "message": message}}

    @staticmethod
    def _check_keys(arguments: dict, allowed: set[str], required: set[str] | None = None) -> None:
        if not isinstance(arguments, dict):
            raise TypeError("工具参数必须是对象。")
        if set(arguments) - allowed or (required or set()) - set(arguments):
            raise ValueError("工具参数缺失或包含不支持的字段。")

    def _dispatch(self, name: str, args: dict, allow_write: bool) -> dict:
        if name in NAVIGATION_TOOLS:
            return self._dispatch_navigation(name, args)
        if name == "get_vehicle_location":
            self._check_keys(args, set())
            return self._get_vehicle_location()
        if name == "get_vehicle_state":
            self._check_keys(args, set())
            return self._vehicle_snapshot()
        if name == 'estimate_driving_range':
            self._check_keys(args, set())
            history = self.memory.snapshot(self.vehicle_id)
            return estimate_range(history, self.state.battery_percent, RESERVE_PERCENT)
        if name == 'get_driving_history':
            self._check_keys(args, {'limit'})
            return self.memory.snapshot(self.vehicle_id, args.get('limit', 10))
        if name == "set_headlights":
            self._check_keys(args, {"enabled"}, {"enabled"})
            if type(args["enabled"]) is not bool:
                raise ValueError("前照灯enabled必须是布尔值。")
            self._check_safety(name, args)
            if allow_write is not True:
                raise PermissionError()
            self.state.headlights_on = args["enabled"]
            return {"headlights_on": self.state.headlights_on, "verified": self.state.headlights_on == args["enabled"]}
        if name == "adjust_seat":
            self._check_keys(args, {"position_percent"}, {"position_percent"})
            value = args["position_percent"]
            if type(value) not in {int, float} or not 0 <= value <= 100 or not math.isfinite(value):
                raise ValueError("座椅位置必须是0～100%的有限数字（教学模拟行程比例）。")
            self._check_safety(name, args)
            if allow_write is not True:
                raise PermissionError()
            self.state.seat_position_percent = float(value)
            return {"seat_position_percent": self.state.seat_position_percent,
                    "verified": self.state.seat_position_percent == value}
        if name == "set_wipers":
            self._check_keys(args, {"mode"}, {"mode"})
            if not isinstance(args["mode"], str) or args["mode"] not in {"off", "low", "high"}:
                raise ValueError("雨刮模式只支持off、low、high。")
            self._check_safety(name, args)
            if allow_write is not True:
                raise PermissionError()
            self.state.wiper_mode = args["mode"]
            return {"wiper_mode": self.state.wiper_mode, "verified": self.state.wiper_mode == args["mode"]}
        if name == "set_temperature":
            self._check_keys(args, {"temperature"}, {"temperature"})
            value = args["temperature"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("温度必须是数字。")
            if not 16 <= value <= 30 or not math.isfinite(value):
                raise ValueError("温度必须在16～30℃之间。")
            self._check_safety(name, args)
            if allow_write is not True:
                raise PermissionError()
            self.state.temperature_c = float(value)
            return {"temperature_c": self.state.temperature_c,
                    "verified": self.state.temperature_c == value}
        if name == "get_route_options":
            self._check_keys(args, {"destination"}, {"destination"})
            if args["destination"] != "机场":
                raise ValueError("模拟地图只支持目的地：机场。")
            routes = []
            for route in deepcopy(ROUTES):
                arrival = self.state.battery_percent - route["battery_consumption_percent"]
                route.update({"arrival_battery_percent": arrival,
                              "meets_reserve": arrival >= RESERVE_PERCENT})
                routes.append(route)
            return {"destination": "机场", "reserve_percent": RESERVE_PERCENT, "routes": routes}
        if name == "find_chargers":
            self._check_keys(args, {"route_id"})
            route_id = args.get("route_id", "")
            if not isinstance(route_id, str) or route_id not in {"", *(r["route_id"] for r in ROUTES)}:
                raise ValueError("route_id 必须为空、airport_fast 或 airport_eco。")
            chargers = []
            for charger in deepcopy(CHARGERS):
                if route_id and charger["route_id"] != route_id:
                    continue
                route = next(r for r in ROUTES if r["route_id"] == charger["route_id"])
                charger["total_trip_minutes"] = (
                    route["duration_minutes"] + charger["detour_minutes"] + charger["charging_minutes"]
                )
                charger["arrival_battery_percent"] = (
                    min(100, self.state.battery_percent + charger["battery_gain_percent"])
                    - route["battery_consumption_percent"]
                )
                chargers.append(charger)
            return {"chargers": chargers, "note": "固定模拟快照，不代表实时可用性；未预约、未充电。"}
        raise ValueError("未知工具。")

    def _dispatch_navigation(self, name, args):
        if name == 'search_navigation_places':
            self._check_keys(args, {'keywords', 'city'}, {'keywords', 'city'})
            return self._navigation_client().search(args['keywords'], args['city'])
        with self._lock:
            self._clear_navigation()
            revision = self._navigation_revision
            location = deepcopy(self.vehicle_location)
        if name == 'navigate_to_destination':
            self._check_keys(args, {'destination', 'strategy'}, {'destination'})
            strategy = args.get('strategy', 'recommended')
            if not isinstance(strategy, str) or strategy not in STRATEGIES:
                raise ValueError('不支持的路线偏好。')
            if location is None:
                raise NavigationError('LOCATION_NOT_SET', '请先在车辆状态面板设置模拟车辆位置，再说“导航到北京南站”。')
            self.trace('navigation.location', location)
            match = self._navigation_client().nearest(args['destination'], location['location'])
            self.trace('navigation.match', match)
            chosen = match['selected']
            if chosen['distance_m'] < 30:
                with self._lock:
                    if revision != self._navigation_revision:
                        raise NavigationError('NAVIGATION_STALE', '查询期间模拟车辆位置已变化，请重新查询。')
                return {'status': 'near_destination', 'origin_simulated': True, 'selection': match,
                        'note': '模拟起点与目的地坐标相距不足30米，请核对具体入口；未重复规划或声称车辆已到达。'}
            origin = {'name': location['name'], 'location': location['location']}
            destination = {k: chosen[k] for k in ('name', 'location', 'id') if k in chosen}
            plan = self._navigation_client().driving(origin, destination, strategy)
            plan.update({'automatic': True, 'origin_simulated': True, 'selection': match, 'origin_source': 'manual_simulation'})
        else:
            self._check_keys(args, {'origin', 'destination', 'strategy'}, {'origin', 'destination'})
            plan = self._navigation_client().driving(args['origin'], args['destination'], args.get('strategy', 'recommended'))
        return self._store_plan(plan, revision)

    def _store_plan(self, plan, revision):
        with self._lock:
            if revision != self._navigation_revision:
                raise NavigationError('NAVIGATION_STALE', '规划期间车辆位置或路线已变化，旧路线未采用，请重新查询。')
            for route in plan['routes']:
                route_id = secrets.token_urlsafe(12)
                route['route_id'] = route_id
                self._route_geometry[route_id] = {'geometry': route.pop('geometry')}
            self.navigation_plan = plan
            return deepcopy(plan)


def build_langchain_tools(service: ToolService) -> list:
    # 延迟导入：离线规则与脚本仅使用标准库，无需密钥，也不初始化模型。
    from langchain.tools import tool

    @tool
    def get_vehicle_state() -> dict:
        """读取模拟车辆的电量、温度、行驶/驻车状态、前照灯、座椅、雨刮和最后一次安全判定。"""
        return service.execute("get_vehicle_state", {})

    @tool
    def set_temperature(temperature: float) -> dict:
        """设置模拟空调温度16～30℃；写权限由应用核对本轮原句中的完整命令和参数后授予。"""
        return service.execute("set_temperature", {"temperature": temperature})

    @tool
    def set_headlights(enabled: bool) -> dict:
        """开关模拟前照灯，仅安全驻车允许；应用须核对用户原句中的完整命令后授予此任务写权限。"""
        return service.execute("set_headlights", {"enabled": enabled})

    @tool
    def adjust_seat(position_percent: float) -> dict:
        """调整模拟座椅到0～100%绝对行程比例，仅安全驻车允许；原句需有参数一致的完整命令如座椅调到60%。"""
        return service.execute("adjust_seat", {"position_percent": position_percent})

    @tool
    def set_wipers(mode: str) -> dict:
        """模拟雨刮off/low/high。关闭需安全驻车，开启/调速无行驶限制；写权限由应用核对本轮原句授予。"""
        return service.execute("set_wipers", {"mode": mode})

    @tool
    def estimate_driving_range() -> dict:
        """根据当前模拟电量、本车可用电池容量和本机长期行驶记录估算剩余公里数。无历史时明确用默认能耗。
        返回保留10%电量的estimated_range_km及理论耗尽range_to_empty_km；是能耗估算，不保证实际可达性。
        """
        return service.execute('estimate_driving_range', {})

    @tool
    def get_driving_history(limit: int = 10) -> dict:
        """读取本车本机SQLite长期行驶记忆，limit为1～20。历史来自人工里程/耗电录入，不是已规划路线或聊天臆测；不能修改历史。"""
        return service.execute('get_driving_history', {'limit':limit})

    @tool
    def get_route_options(destination: str) -> dict:
        """查询模拟路线、耗时和预计到达电量；destination 只支持“机场”。"""
        return service.execute("get_route_options", {"destination": destination})

    @tool
    def find_chargers(route_id: str = "") -> dict:
        """查询模拟充电站及充电后的行程估算。route_id 为 airport_fast、airport_eco 或空（全部）。"""
        return service.execute("find_chargers", {"route_id": route_id})

    @tool
    def search_navigation_places(keywords: str, city: str) -> dict:
        """联网搜索真实高德地点，需明确城市。返回GCJ-02坐标、名称、ID和候选列表；有歧义应让用户选择。"""
        return service.execute("search_navigation_places", {"keywords": keywords, "city": city})

    @tool
    def plan_navigation_route(origin: dict, destination: dict, strategy: str = "recommended") -> dict:
        """查询真实高德驾车路线，不启动导航。origin/destination为搜索得到的{name,location,id}，location是经度,纬度。
        strategy支持recommended/avoid_congestion/less_fee/no_highway。返回距离、时间、路书及高德链接，不含电动车耗电估算。
        """
        return service.execute("plan_navigation_route", {"origin": origin, "destination": destination, "strategy": strategy})

    @tool
    def get_vehicle_location() -> dict:
        """读取应用中用户设置的模拟车辆当前位置（GCJ-02），未设置时明确返回错误；不能猜测或修改位置。"""
        return service.execute("get_vehicle_location", {})

    @tool
    def navigate_to_destination(destination: str, strategy: str = "recommended") -> dict:
        """从当前已设置的模拟车辆位置导航到名称匹配的最近候选，无需再让用户填写起点或选择候选。
        内部读取车辆位置、搜索高德、按近似直线距离选择匹配候选并规划路线。destination只传目的地关键词，如北京南站、附近的麦当劳。
        strategy支持recommended/avoid_congestion/less_fee/no_highway。不启动外部App或车载实时导航，不能称已经开车或到达。
        """
        return service.execute("navigate_to_destination", {"destination": destination, "strategy": strategy})

    return [get_vehicle_state, set_temperature, set_headlights, adjust_seat, set_wipers, get_route_options, find_chargers,
            search_navigation_places, plan_navigation_route, get_vehicle_location, navigate_to_destination,
            estimate_driving_range, get_driving_history]
