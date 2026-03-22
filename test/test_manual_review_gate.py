import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
TEST_TMP_ROOT = ROOT_DIR / ".tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

from src.skill_loop.loop import SkillAgentLoop
from src.skill_loop.runner import CodexRunnerConfig


def make_test_dir(test_case: unittest.TestCase) -> Path:
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    test_case.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


class ManualReviewGateTests(unittest.TestCase):
    def test_final_promoted_version_requires_manual_review_after_loop(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        source_dir = skills_root / "minor-detection"
        baseline_dir = skills_root / "minor-detection-v0.1.0"
        legacy_stable_dir = skills_root / "minor-detection-v0.1.1-20260322_145202"
        legacy_candidate_dir = skills_root / "minor-detection-v0.1.1-rc001-20260322_145202"
        source_dir.mkdir(parents=True, exist_ok=True)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        legacy_stable_dir.mkdir(parents=True, exist_ok=True)
        legacy_candidate_dir.mkdir(parents=True, exist_ok=True)

        config = mock.Mock()
        config.baseline_source_dir = source_dir
        config.baseline_version = "minor-detection-v0.1.0"
        config.dataset_path = root / "data" / "val.jsonl"
        config.max_rounds = 1
        config.max_errors = None
        config.protected_count = 8
        config.workspace_root = root / "workspace"
        config.packages_root = root / "packages"
        config.refresh_baseline_version = False
        config.runner_config = CodexRunnerConfig()

        baseline_eval = {
            "report_path": root / "baseline-report.json",
            "failure_packets_dir": root / "failure_packets",
            "protected_packets_dir": root / "protected_packets",
            "protected_index_path": root / "protected_index.jsonl",
            "runtime_summary": {},
            "runtime_counts": {},
            "report_payload": {
                "sample_count": 2,
                "invocation_success_rate": 1.0,
            },
        }
        candidate_eval = {
            "report_path": root / "candidate-report.json",
            "error_index_path": root / "candidate_error_index.jsonl",
            "runtime_summary": {},
            "runtime_counts": {},
        }

        optimizer = mock.Mock()

        def optimize_side_effect(**kwargs):
            candidate_version = kwargs["new_version"]
            (skills_root / candidate_version).mkdir(parents=True, exist_ok=True)
            return {
                "success": True,
                "current_version": "minor-detection-v0.1.0",
                "new_version": candidate_version,
                "edited_files": ["references/evidence-rules.md"],
            }

        optimizer.optimize_from_judge_artifacts.side_effect = optimize_side_effect
        optimizer.create_formal_skill_review_artifact.return_value = {
            "base_version": "minor-detection-v0.1.0",
            "candidate_version": "minor-detection-v0.1.1-20260322_120000",
            "review_diff_path": "skills/minor-detection-v0.1.1-20260322_120000/review/diff.md",
        }

        loop = SkillAgentLoop(config=config, runner=mock.Mock(runner_mode="direct"), optimizer=optimizer)
        workspace = root / "workspace" / "20260322_120000"
        workspace.mkdir(parents=True, exist_ok=True)

        with (
            mock.patch("src.skill_loop.loop.SKILLS_DIR", skills_root),
            mock.patch("src.skill_loop.loop.ensure_version_snapshot"),
            mock.patch("src.skill_loop.loop.get_active_skill_version", return_value="minor-detection"),
            mock.patch.object(loop, "_workspace", return_value=workspace),
            mock.patch.object(loop, "_evaluate_version", side_effect=[baseline_eval, candidate_eval]),
            mock.patch(
                "src.skill_loop.loop.compare_reports",
                return_value={"decision": "promote", "accepted_f1": 0.9, "candidate_f1": 1.0, "f1_delta": 0.1},
            ),
        ):
            summary = loop.run()

        optimizer.create_formal_skill_review_artifact.assert_called_once_with(
            base_version="minor-detection-v0.1.0",
            candidate_version="minor-detection-v0.1.1-20260322_120000",
        )
        self.assertEqual(summary["final_version"], "minor-detection-v0.1.1-20260322_120000")
        self.assertTrue(summary["manual_review_required"])
        self.assertEqual(summary["manual_review_status"], "pending")
        self.assertEqual(summary["manual_review_base_version"], "minor-detection-v0.1.0")
        self.assertEqual(summary["manual_review_candidate_version"], "minor-detection-v0.1.1-20260322_120000")
        self.assertIn("run_final_test.py --version minor-detection-v0.1.1-20260322_120000", summary["manual_final_test_command"])
        self.assertIn("--review-decision approve", summary["manual_review_approve_command"])
        self.assertIn("--review-decision reject", summary["manual_review_reject_command"])
        self.assertEqual(summary["rounds"][0]["comparison"]["decision"], "promote")
        self.assertEqual(summary["rounds"][0]["promoted_to"], "minor-detection-v0.1.1-20260322_120000")
        self.assertFalse(summary["rounds"][0].get("manual_review_required", False))
        self.assertEqual(summary["version_management"]["run_tag"], "20260322_120000")
        self.assertEqual(summary["version_management"]["history_scope"], "current_run_only")
        self.assertIsNone(summary["version_management"]["inventory_before"]["active_version"])
        self.assertEqual(summary["version_management"]["inventory_before"]["stable_versions"], [])
        self.assertEqual(summary["version_management"]["inventory_before"]["candidate_versions"], [])
        self.assertIsNone(summary["version_management"]["inventory_after"]["active_version"])
        self.assertEqual(summary["version_management"]["inventory_after"]["stable_versions"], ["minor-detection-v0.1.1-20260322_120000"])
        self.assertEqual(summary["version_management"]["inventory_after"]["candidate_versions"], ["minor-detection-v0.1.1-rc001-20260322_120000"])
        self.assertIn("--only-run-tag 20260322_120000", summary["version_management"]["recommended_cleanup_command"])
        self.assertNotIn("all_inventory_after", summary["version_management"])
        self.assertTrue((skills_root / "minor-detection-v0.1.1-20260322_120000").exists())


if __name__ == "__main__":
    unittest.main()


