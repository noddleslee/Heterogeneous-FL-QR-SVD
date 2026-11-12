"""Check child-process and evaluation entry points without loading ML libraries.

Run: python -m unittest discover -s tests -p test_launch_config.py -v
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))
import runtime_config
import run_simulation


class LaunchConfigTests(unittest.TestCase):
    def test_defaults_use_current_environment_and_included_evaluator(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(runtime_config.python_executable(), sys.executable)
            self.assertEqual(Path(runtime_config.evaluation_script()), ROOT / "eval" / "eval_pathvqa.py")
            self.assertTrue(Path(runtime_config.evaluation_script()).is_file())
            self.assertEqual(Path(runtime_config.dataset_cache_dir()), ROOT / ".dataset_cache")

    def test_environment_overrides(self):
        with patch.dict(os.environ, {"HFEDLORA_PYTHON": "/custom/python", "HFEDLORA_EVAL_SCRIPT": "/custom/eval.py", "HFEDLORA_CACHE_DIR": "/custom/cache"}):
            self.assertEqual(runtime_config.python_executable(), "/custom/python")
            self.assertEqual(runtime_config.evaluation_script(), "/custom/eval.py")
            self.assertEqual(runtime_config.dataset_cache_dir(), "/custom/cache")

    def test_client_command_points_to_existing_script_and_keeps_protocol(self):
        cmd = run_simulation._build_client_cmd(0, 4, 1, "results/example", "Qwen/Qwen3.5-0.8B", 8, 64, 1, 100, 10, "previous/global_model", True)
        self.assertEqual(cmd[0], run_simulation.PYTHON_EXE[0])
        self.assertTrue(Path(cmd[2]).is_file())
        self.assertEqual(cmd[cmd.index("--batch_size") + 1], "64")
        self.assertEqual(cmd[cmd.index("--rank") + 1], "8")
        self.assertIn("--flexlora", cmd)

    def test_evaluation_job_uses_validation_and_included_script(self):
        with tempfile.TemporaryDirectory(prefix=".launch-test-", dir=ROOT) as temporary:
            Path(temporary, "round_0").mkdir()
            with patch.object(run_simulation.subprocess, "check_output", return_value="Submitted batch job 12345") as submit:
                job = run_simulation._submit_eval(0, temporary, "Qwen/Qwen3.5-0.8B", str(Path(temporary, "global_model")))
            text = Path(temporary, "round_0", "submit_eval.sh").read_text()
            self.assertEqual(job, "12345")
            self.assertIn(run_simulation.EVAL_SCRIPT, text)
            self.assertIn("--split validation", text)
            self.assertIn("--n_samples none", text)
            self.assertNotIn("hfedlora_supervisor", text)
            self.assertEqual(submit.call_args.args[0][0], "sbatch")


if __name__ == "__main__":
    unittest.main()
