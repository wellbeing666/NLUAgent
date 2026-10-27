"""确定性的教学车辆策略。这里只使用模拟状态，不代表真实车辆的功能安全认证。"""

import math
from dataclasses import dataclass, asdict

from cockpit.data import VehicleState

POLICY_VERSION = "cockpit-demo-v1"
MOTION_LABELS = {"parked": "已停车", "moving": "行驶中", "not_parked": "未完成驻车", "unknown": "状态未知"}
CONTROL_TOOLS = {"set_temperature", "set_headlights", "adjust_seat", "set_wipers"}
WRITE_TOOLS = set(CONTROL_TOOLS)
MOTION_PRESETS = {
    "parked": {"speed_kph": 0.0, "gear": "P", "parking_brake": True},
    "driving": {"speed_kph": 40.0, "gear": "D", "parking_brake": False},
    "unknown": {"speed_kph": None, "gear": None, "parking_brake": None},
}


def motion_snapshot(state: VehicleState) -> dict:
    speed, gear, brake = state.speed_kph, state.gear, state.parking_brake
    valid_speed = type(speed) in {int, float} and 0 <= speed <= 250 and math.isfinite(speed)
    valid_gear = isinstance(gear, str) and gear in {"P", "R", "N", "D"}
    if not valid_speed or not valid_gear or type(brake) is not bool:
        status = "unknown"
    elif speed > 0:
        status = "moving"
    elif gear == "P" and brake:
        status = "parked"
    else:
        status = "not_parked"
    return {"status": status, "label": MOTION_LABELS[status], "speed_kph": speed,
            "gear": gear, "parking_brake": brake, "source": "manual_simulation", "simulated": True}


def validate_driving_state(speed_kph, gear, parking_brake) -> dict:
    # 可以显式设置未知值演示保守拒绝，但非法输入不能伪装成停车。
    if speed_kph is not None and (type(speed_kph) not in {int, float}
                                 or not 0 <= speed_kph <= 250 or not math.isfinite(speed_kph)):
        raise ValueError("模拟车速必须是0～250km/h的有限数字，或null表示未知。")
    if gear is not None and (not isinstance(gear, str) or gear not in {"P", "R", "N", "D"}):
        raise ValueError("挡位必须为P、R、N、D，或null表示未知。")
    if parking_brake is not None and type(parking_brake) is not bool:
        raise ValueError("驻车制动必须为布尔值，或null表示未知。")
    return {"speed_kph": speed_kph, "gear": gear, "parking_brake": parking_brake}


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    action: str
    rule_id: str
    reason: str
    vehicle: dict
    policy_version: str = POLICY_VERSION

    def to_dict(self):
        return asdict(self)


class SafetyViolation(Exception):
    def __init__(self, decision: SafetyDecision):
        self.decision = decision
        super().__init__(decision.reason)


def evaluate_action(action: str, arguments: dict, state: VehicleState) -> SafetyDecision:
    """调用方先校验参数，并与执行动作共用同一把状态锁。"""
    vehicle = motion_snapshot(state)
    if action == 'set_temperature':
        return SafetyDecision(True, action, 'TEMPERATURE_ALLOWED', '空调设置不受行驶状态限制，仍需参数与写授权检查。', vehicle)
    if action == "set_wipers" and arguments["mode"] != "off":
        return SafetyDecision(True, action, "WIPERS_ON_ALLOWED", "开启或调整雨刮速度无需停车。", vehicle)
    rules = {"set_headlights": ("HEADLIGHTS_PARK_ONLY", "前照灯开关"),
             "adjust_seat": ("SEAT_PARK_ONLY", "调整座椅"),
             "set_wipers": ("WIPERS_OFF_PARK_ONLY", "关闭雨刮器")}
    if action not in rules:
        raise ValueError("安全策略不支持该车控动作。")
    rule, label = rules[action]
    if vehicle["status"] == "parked":
        return SafetyDecision(True, action, rule, "车速为0、P挡且已驻车，符合本动作安全条件。", vehicle)
    reason = f"{vehicle['label']}，禁止{label}。请先停车、切换P挡并启用驻车制动，再重新下达指令。"
    return SafetyDecision(False, action, rule, reason, vehicle)
