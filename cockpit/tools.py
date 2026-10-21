"""共享工具执行层。业务校验与权限由代码执行，不依赖提示词。"""

import math
from copy import deepcopy
from threading import Lock
from typing import Any, Callable

from cockpit.data import CHARGERS, RESERVE_PERCENT, ROUTES, VehicleState

Trace = Callable[[str, dict[str, Any]], None]


def quiet_trace(event: str, payload: dict[str, Any]) -> None:
    pass


class ToolService:
    def __init__(self, trace: Trace = quiet_trace):
        self.state = VehicleState()
        self.trace = trace
        self._lock = Lock()

    def reset(self) -> None:
        with self._lock:
            self.state = VehicleState()

    def execute(self, name: str, arguments: dict[str, Any], *, allow_write: bool = False) -> dict:
        # LangChain 可能并行执行工具；串行访问本地状态，日志也保持成对。
        with self._lock:
            self.trace("tool.call", {"name": name, "arguments": arguments})
            try:
                data = self._dispatch(name, arguments, allow_write)
                result = {"ok": True, "simulated": True, "data": data}
            except (ValueError, TypeError) as exc:
                result = self._error("INVALID_ARGUMENT", str(exc))
            except PermissionError:
                result = self._error(
                    "WRITE_NOT_AUTHORIZED", "本版本只允许完整车控命令修改温度，请单独输入：空调调到24度。"
                )
            self.trace("tool.result", {"name": name, "result": result})
            return result

    @staticmethod
    def _error(code: str, message: str) -> dict:
        return {"ok": False, "simulated": True, "error": {"code": code, "message": message}}

    @staticmethod
    def _check_keys(arguments: dict, allowed: set[str], required: set[str] | None = None) -> None:
        if not isinstance(arguments, dict):
            raise TypeError("工具参数必须是对象。")
        if set(arguments) - allowed or (required or set()) - set(arguments):
            raise ValueError("工具参数缺失或包含不支持的字段。")

    def _dispatch(self, name: str, args: dict, allow_write: bool) -> dict:
        if name == "get_vehicle_state":
            self._check_keys(args, set())
            return {"battery_percent": self.state.battery_percent,
                    "temperature_c": self.state.temperature_c,
                    "reserve_percent": RESERVE_PERCENT}
        if name == "set_temperature":
            self._check_keys(args, {"temperature"}, {"temperature"})
            value = args["temperature"]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("温度必须是数字。")
            if not math.isfinite(value) or not 16 <= value <= 30:
                raise ValueError("温度必须在16～30℃之间。")
            if not allow_write:
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


def build_langchain_tools(service: ToolService) -> list:
    # 延迟导入：离线规则与脚本仅使用标准库，无需密钥，也不初始化模型。
    from langchain.tools import tool

    @tool
    def get_vehicle_state() -> dict:
        """读取模拟车辆的当前电量、空调温度和到达时最低预留电量。"""
        return service.execute("get_vehicle_state", {})

    @tool
    def set_temperature(temperature: float) -> dict:
        """设置模拟空调温度，范围16～30℃。Agent路径没有写权限，需用户单独下达完整车控命令。"""
        return service.execute("set_temperature", {"temperature": temperature})

    @tool
    def get_route_options(destination: str) -> dict:
        """查询模拟路线、耗时和预计到达电量；destination 只支持“机场”。"""
        return service.execute("get_route_options", {"destination": destination})

    @tool
    def find_chargers(route_id: str = "") -> dict:
        """查询模拟充电站及充电后的行程估算。route_id 为 airport_fast、airport_eco 或空（全部）。"""
        return service.execute("find_chargers", {"route_id": route_id})

    return [get_vehicle_state, set_temperature, get_route_options, find_chargers]
