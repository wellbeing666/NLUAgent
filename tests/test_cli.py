"""通过实际子进程验证命令行入口与交互继续能力。"""

import os
import subprocess
import sys
import unittest
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

    def test_interactive_reset_and_exit(self):
        result = self.run_cli("--mode", "offline", user_input="1\n/state\n/reset\n/state\n/exit\n")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("空调24℃", result.stdout)
        self.assertIn("空调26℃", result.stdout)
        self.assertIn("已退出", result.stdout)

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
