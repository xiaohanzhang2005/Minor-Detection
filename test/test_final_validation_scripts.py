import io
import json
import runpy
import shutil
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
TEST_TMP_ROOT = ROOT_DIR / ".tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)


def make_test_dir(test_case: unittest.TestCase) -> Path:
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    test_case.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


class FinalValidationScriptTests(unittest.TestCase):
    def test_run_final_test_outputs_expected_payload(self):
        root = make_test_dir(self)
        run_root = root / "reports" / "run"
        run_root.mkdir(parents=True, exist_ok=True)
        report_path = run_root / "report.json"
        report_path.write_text("{}", encoding="utf-8")

        runner_instance = mock.Mock()
        runner_instance.run_dataset.return_value = run_root
        stdout_buffer = io.StringIO()

        with (
            mock.patch("src.skill_loop.CodexSkillRunner", return_value=runner_instance),
            mock.patch(
                "src.skill_loop.judge_run_artifacts",
                return_value={
                    "report_path": report_path,
                    "report_payload": {
                        "final_validation_metrics": {
                            "metrics": {"accuracy": 0.9, "f1_score": 0.88},
                            "schema_validity_rate": 1.0,
                        },
                        "contract_check_preview": {
                            "release_gate_pass": True,
                            "schema_perfect_pass": True,
                        },
                    },
                },
            ),
            mock.patch.object(
                sys,
                "argv",
                [
                    "run_final_test.py",
                    "--version",
                    "minor-detection-v0.1.1",
                    "--dataset",
                    str(root / "data" / "test.jsonl"),
                    "--workspace",
                    str(root / "reports" / "final_tests"),
                ],
            ),
            redirect_stdout(stdout_buffer),
        ):
            runpy.run_path(str(ROOT_DIR / "scripts" / "run_final_test.py"), run_name="__main__")

        payload = json.loads(stdout_buffer.getvalue())
        self.assertEqual(payload["evaluation_role"], "standalone_formal_final_validation")
        self.assertFalse(payload["optimization_feedback_enabled"])
        self.assertEqual(payload["runner_mode"], "agent")
        self.assertEqual(payload["final_validation_metrics"]["metrics"]["f1_score"], 0.88)
        self.assertTrue(payload["contract_gate_all_green"])
        self.assertTrue(payload["contract_check_all_green"])
        self.assertFalse(payload["release_contract_gate_blocking"])
        self.assertIn("reports/run/report.json", payload["report_path"])


if __name__ == "__main__":
    unittest.main()
