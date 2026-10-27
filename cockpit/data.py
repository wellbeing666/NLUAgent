"""固定世界：耗电是电池电量的百分点，不是物理能耗模型。"""

from dataclasses import dataclass


@dataclass
class VehicleState:
    battery_percent: float = 25.0
    temperature_c: float = 26.0
    speed_kph: float | None = 0.0
    gear: str | None = "P"
    parking_brake: bool | None = True
    headlights_on: bool = False
    seat_position_percent: float = 50.0
    wiper_mode: str = "off"


RESERVE_PERCENT = 10.0

ROUTES = (
    {"route_id": "airport_fast", "name": "机场高速", "duration_minutes": 48,
     "distance_km": 52, "battery_consumption_percent": 18.0},
    {"route_id": "airport_eco", "name": "机场节能路线", "duration_minutes": 65,
     "distance_km": 43, "battery_consumption_percent": 12.0},
)

CHARGERS = (
    {"charger_id": "C01", "name": "高速服务区充电站", "route_id": "airport_fast",
     "available_ports": 2, "power_kw": 120, "detour_minutes": 8,
     "charging_minutes": 20, "battery_gain_percent": 30.0},
    {"charger_id": "C02", "name": "城东充电站", "route_id": "airport_eco",
     "available_ports": 0, "power_kw": 90, "detour_minutes": 12,
     "charging_minutes": 25, "battery_gain_percent": 30.0},
)
