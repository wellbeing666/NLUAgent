"""通过实际子进程验证命令行入口与交互继续能力。"""

import os
import subprocess
import sys
import unittest
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def run_cli(self, *args, user_input=None):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", OPENAI_API_KEY="", OPENAI_MODEL="")
        return subprocess.run(
            [sys.executable, "main.py", *args], input=user_input, capture_output=True,
            encoding="utf-8", cwd=ROOT, env=env, timeout=30,
        )

    def test_demo_all_without_key(self):
        result = self.run_cli("--mode", "offline", "--demo", "all")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("模型决策", result.stdout)
        self.assertIn("离线脚本演示", result.stdout)
        self.assertIn("操作未执行", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_driving_memory_survives_cli_process_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'memory.sqlite3')
            first = self.run_cli('--mode','offline','--memory-db',path,
                                 user_input='/battery 25|60\n/trip 50|8.5\n/range\n/reset\n/trips\n/exit\n')
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn('52.9公里', first.stdout)
            self.assertIn('已保存1条人工行程记录', first.stdout)
            second = self.run_cli('--mode','offline','--memory-db',path,'--query','当前电量还能跑多远')
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn('最近1条历史', second.stdout)
            self.assertIn('52.9公里', second.stdout)

    def test_interactive_reset_and_exit(self):
        result = self.run_cli("--mode", "offline", user_input="1\n/state\n/reset\n/state\n/exit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("空调24℃", result.stdout)
        self.assertIn("空调26℃", result.stdout)
        self.assertIn("已退出", result.stdout)

    def test_location_command_validates_and_keeps_offline_chat_offline(self):
        result = self.run_cli('--mode', 'offline', user_input='/location 北京天安门（模拟）|116.397428,39.909230\n/location 错误位置|NaN,39\n导航到北京南站\n/reset\n/exit\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('已设置模拟车辆位置', result.stdout)
        self.assertIn('位置未设置', result.stdout)
        self.assertIn('当前离线，未调用高德或模型', result.stdout)
        self.assertNotIn('Traceback', result.stderr)

    def test_motion_command_exercises_safety_without_a_model(self):
        result = self.run_cli('--mode', 'offline', user_input='打开前照灯\n/motion driving\n关闭前照灯\n/motion parked\n关闭前照灯\n/motion unknown\n座椅调到60%\n/exit\n')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('SAFETY_BLOCKED', result.stdout)
        self.assertIn('行驶中', result.stdout)
        self.assertIn('状态未知', result.stdout)
        self.assertEqual(result.stdout.count('助手：模拟前照灯已关闭'), 1)
        self.assertNotIn('Traceback', result.stderr)

    def test_live_missing_config_then_nlu_still_works(self):
        result = self.run_cli("--mode", "live", user_input="去机场\n空调调到24度\n/exit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("未调用模型", result.stdout)
        self.assertIn("已设为24", result.stdout)
        self.assertNotIn("Traceback", result.stderr)

    def test_live_demo_flag_is_rejected(self):
        result = self.run_cli("--mode", "live", "--demo", "all")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("只用于offline", result.stderr)

    def test_extreme_temperature_is_rejected_without_logging_crash(self):
        result = self.run_cli("--mode", "offline", "--query", "空调调到" + "9" * 400 + "度")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("操作未执行", result.stdout)
        self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
