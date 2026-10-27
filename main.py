"""运行入口：python main.py --mode offline --demo all。"""

import argparse
import json
import math
import sys
from pathlib import Path
from uuid import uuid4

from cockpit.app import DEMO_QUERIES, CabinApp
from cockpit.semantic_nlu import NLURouter, NLUStartupError
from cockpit.safety import MOTION_PRESETS
from cockpit.driving_memory import DrivingMemory, MemoryUnavailable

HELP = """命令：/help 帮助；/state 状态；/reset 重置状态和上下文；/exit 退出。
/location 名称|经度,纬度：手动设置模拟车辆位置（GCJ-02）。
/motion parked|driving|unknown：切换模拟停车/行驶/未知状态（仅演示入口，模型不能调用）。
/battery 电量百分比|可用容量kWh：设置会话模拟电量并持久化本车容量，例如 /battery 25|60。
/trip 行驶公里数|实际耗电kWh：手动保存本车长期行驶记忆，例如 /trip 50|8.5。
/range 估算剩余续航；/trips 查看历史。重置聊天不会删除长期记录。
安全演示：打开前照灯；座椅调到60%；开启雨刮器；关闭雨刮器。
示例1：空调调到24度
示例2：电量只有25%，去机场要不要充电？帮我找个方案
示例3：空调调到35度
可输入1、2、3运行对应案例。离线模式也支持“演示出行”。
真实模式追问示例：换一条耗电更少的路线。
多意图（live）：把空调调到24度，并打开前照灯，然后查询车辆状态。
hybrid模式还支持“把车内温度设置成二十四度”“现在还有多少电”。
车控仅接受明确完整命令；offline保留机场模拟地图。
配置AMAP_API_KEY后，live的Agent可搜索高德地点与规划真实路线；也可运行web.py打开导航面板。"""


def console_trace(event: str, payload: dict) -> None:
    def json_safe(value):
        # 非法数值也要能被记录，再交给工具拒绝，不能让日志先把程序弄崩。
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, list):
            return [json_safe(item) for item in value]
        return value

    labels = {"route": "路由", "nlu.result": "NLU识别", "agent.tools_selected": "模型选择工具",
              "tool.call": "工具调用", "tool.result": "工具结果", "agent.error": "错误", "safety.check": "车辆安全判定",
              'planner.started':'开始规划', 'planner.plan':'任务计划', 'executor.plan':'调度计划', 'task.status':'任务状态',
              'task.guard':'任务写授权', 'task.retry':'任务重试', 'executor.completed':'调度完成', 'planner.error':'计划拒绝'}
    labels['memory.recorded'] = '行驶记忆保存'
    labels.update({'response.started':'整理回答', 'response.completed':'回答已生成', 'response.fallback':'回答兜底'})
    print(f"[{labels.get(event, event)}] {json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="智能座舱Agent教学原型（模拟车辆＋可选高德导航）")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--nlu", choices=["rules", "hybrid"], default="rules", help="rules规则基线；hybrid规则＋本地BGE语义识别")
    parser.add_argument('--memory-db', type=Path, help='本机行驶记忆数据库路径')
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", choices=["all", *DEMO_QUERIES], help="运行固定离线案例后退出")
    group.add_argument("--query", help="执行一条指令后退出")
    args = parser.parse_args(argv)
    if args.demo and args.mode != "offline":
        parser.error("--demo 只用于offline；真实调用请用 --mode live --query 或交互输入。")
    print(f"智能座舱学习版 | {'离线脚本演示，无模型决策' if args.mode == 'offline' else '真实模型模式，简单指令仍走本地NLU'}")
    print("车辆、机场案例、耗电和充电站为模拟数据；高德导航工具返回真实路线，不含耗电估算。")
    try:
        nlu = NLURouter.hybrid() if args.nlu == "hybrid" else NLURouter()
    except (NLUStartupError, OSError, ValueError) as exc:
        print(f"本地NLU启动失败：{exc}", file=sys.stderr)
        return 2
    print(f"NLU：{args.nlu}；本地车控需通过写授权及适用的车辆安全检查。")
    memory = DrivingMemory(args.memory_db) if args.memory_db else DrivingMemory()
    app = CabinApp(args.mode, trace=console_trace, nlu=nlu, memory=memory)
    if args.demo:
        cases = list(DEMO_QUERIES) if args.demo == "all" else [args.demo]
        for case in cases:
            print(f"\n你：{DEMO_QUERIES[case]}")
            print("助手：" + app.handle(DEMO_QUERIES[case]))
        return 0
    if args.query:
        print("助手：" + app.handle(args.query))
        return 0
    print(HELP)
    aliases = dict(zip(["1", "2", "3"], DEMO_QUERIES.values()))
    while True:
        try:
            text = input("\n你：").strip()
            if text.lower() in {"/exit", "exit", "quit"}:
                break
            if text == "/help":
                print(HELP)
                continue
            if text == "/reset":
                print("助手：" + app.reset())
                continue
            if text == "/state":
                text = "查看车辆状态"
            if text == '/range':
                text = '估算剩余续航'
            if text == '/trips':
                text = '查看历史行驶记录'
            if text.startswith('/battery ') or text.startswith('/trip '):
                try:
                    left, right = (float(value) for value in text.split(' ', 1)[1].split('|'))
                    if text.startswith('/battery '):
                        app.service.set_battery(battery_percent=left, capacity_kwh=right)
                        print('助手：已设置模拟电量并保存本车可用电池容量。')
                    else:
                        app.service.record_driving_trip(distance_km=left, energy_kwh=right, record_id=str(uuid4()))
                        print('助手：已保存人工行程记录，续航查询将读取本车长期记忆；当前电量未自动扣减。')
                except (ValueError, MemoryUnavailable) as exc:
                    print(f'助手：未完成：{exc}。格式：/battery 25|60 或 /trip 50|8.5')
                continue
            if text.startswith("/motion "):
                motion = text.removeprefix("/motion ").strip()
                if motion not in MOTION_PRESETS:
                    print("助手：仅支持 /motion parked、/motion driving、/motion unknown。")
                else:
                    snapshot = app.service.set_driving_state(**MOTION_PRESETS[motion])
                    print(f"助手：模拟车辆状态已切换为{snapshot['label']}。已拒绝的动作不会自动重试，请重新输入指令。")
                continue
            if text.startswith("/location "):
                try:
                    name, location = text.removeprefix("/location ").split("|", 1)
                    app.service.set_vehicle_location({"name": name, "location": location})
                    print("助手：已设置模拟车辆位置；live模式可输入“导航到北京南站”。")
                except ValueError as exc:
                    print(f"助手：位置未设置：{exc}。格式：/location 名称|经度,纬度")
                continue
            print("助手：" + app.handle(aliases.get(text, text)))
        except (EOFError, KeyboardInterrupt):
            break
    print("\n已退出；本次模拟状态与对话上下文不写入磁盘；手动保存的行驶记忆和电池容量持久保留。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
