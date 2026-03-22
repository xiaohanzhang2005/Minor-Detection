import json
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

from src.evolution.optimizer import SkillOptimizer


def make_test_dir(test_case: unittest.TestCase) -> Path:
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    test_case.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


class FormalReviewFinalizeTests(unittest.TestCase):
    def test_approve_adopts_candidate_version_directly_without_syncing_base_dir(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        base_dir = skills_root / "minor-detection-v0.1.0"
        candidate_dir = skills_root / "minor-detection-v0.1.1-20260322_165515"

        for skill_dir in (base_dir, candidate_dir):
            (skill_dir / "references").mkdir(parents=True, exist_ok=True)
            (skill_dir / "review").mkdir(parents=True, exist_ok=True)
            (skill_dir / "references" / "output-schema.md").write_text("schema\n", encoding="utf-8")
            (skill_dir / "SKILL.md").write_text(f"# {skill_dir.name}\n", encoding="utf-8")

        (base_dir / "references" / "evidence-rules.md").write_text("base\n", encoding="utf-8")
        (candidate_dir / "references" / "evidence-rules.md").write_text("candidate\n", encoding="utf-8")

        optimizer = SkillOptimizer(skills_dir=str(skills_root))

        with mock.patch("src.evolution.optimizer.set_active_skill_version") as set_active:
            result = optimizer.finalize_formal_skill_review(
                base_version="minor-detection-v0.1.0",
                candidate_version="minor-detection-v0.1.1-20260322_165515",
                decision="approve",
            )

        set_active.assert_called_once_with("minor-detection-v0.1.1-20260322_165515")
        self.assertEqual(result["adopted_version"], "minor-detection-v0.1.1-20260322_165515")
        self.assertTrue(result["candidate_adopted_directly"])
        self.assertFalse(result["candidate_synced_to_base"])
        self.assertEqual(
            (base_dir / "references" / "evidence-rules.md").read_text(encoding="utf-8"),
            "base\n",
        )
        decision_path = candidate_dir / "review" / "review_decision_vs_minor-detection-v0.1.0.json"
        self.assertTrue(decision_path.exists())
        decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
        self.assertEqual(decision_payload["adopted_version"], "minor-detection-v0.1.1-20260322_165515")
        self.assertTrue(decision_payload["candidate_adopted_directly"])
        self.assertFalse(decision_payload["candidate_synced_to_base"])


if __name__ == "__main__":
    unittest.main()
