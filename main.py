"""运行入口：python main.py --mode offline --demo all。"""

import argparse
import json
import math
import sys

from cockpit.app import DEMO_QUERIES, CabinApp
from cockpit.semantic_nlu import NLURouter, NLUStartupError

HELP = """命令：/help 帮助；/state 状态；/reset 重置状态和上下文；/exit 退出。
示例1：空调调到24度
示例2：电量只有25%，去机场要不要充电？帮我找个方案
示例3：空调调到35度
可输入1、2、3运行对应案例。离线模式也支持“演示出行”。
真实模式追问示例：换一条耗电更少的路线。
hybrid模式还支持“把车内温度设置成二十四度”“现在还有多少电”。
车控仅接受明确完整命令；当前仅支持机场模拟地图。"""


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
              "tool.call": "工具调用", "tool.result": "工具结果", "agent.error": "错误"}
    print(f"[{labels.get(event, event)}] {json.dumps(json_safe(payload), ensure_ascii=False, allow_nan=False)}", flush=True)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="智能座舱Agent教学原型（全部车辆与路线为模拟数据）")
    parser.add_argument("--mode", choices=["offline", "live"], default="offline")
    parser.add_argument("--nlu", choices=["rules", "hybrid"], default="rules", help="rules规则基线；hybrid规则＋本地BGE语义识别")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--demo", choices=["all", *DEMO_QUERIES], help="运行固定离线案例后退出")
    group.add_argument("--query", help="执行一条指令后退出")
    args = parser.parse_args(argv)
    if args.demo and args.mode != "offline":
        parser.error("--demo 只用于offline；真实调用请用 --mode live --query 或交互输入。")
    print(f"智能座舱学习版 | {'离线脚本演示，无模型决策' if args.mode == 'offline' else '真实模型模式，简单指令仍走本地NLU'}")
    print("车辆、地图、耗电和充电站均为模拟数据。")
    try:
        nlu = NLURouter.hybrid() if args.nlu == "hybrid" else NLURouter()
    except (NLUStartupError, OSError, ValueError) as exc:
        print(f"本地NLU启动失败：{exc}", file=sys.stderr)
        return 2
    print(f"NLU：{args.nlu}；本地温度指令需通过单独的执行检查。")
    app = CabinApp(args.mode, trace=console_trace, nlu=nlu)
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
            print("助手：" + app.handle(aliases.get(text, text)))
        except (EOFError, KeyboardInterrupt):
            break
    print("\n已退出；本次模拟状态与对话上下文不写入磁盘。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
