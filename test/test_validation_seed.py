# 模块说明：
# - Mode A validation seed 的回归测试。
# - 保护自迭代验收专用坏 baseline 的生成逻辑。

import json
import shutil
import unittest
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(ROOT_DIR))

TEST_TMP_ROOT = ROOT_DIR / ".tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

from src.skill_loop.validation_seed import (
    build_mode_a_validation_command,
    build_mode_a_validation_payload,
    build_validation_seed_version_name,
    create_mode_a_validation_seed,
)


def make_test_dir(test_case: unittest.TestCase) -> Path:
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    test_case.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


class ValidationSeedTests(unittest.TestCase):
    def _make_skill(self, root: Path) -> Path:
        skill_dir = root / "skills" / "minor-detection-v0.1.0"
        (skill_dir / "references").mkdir(parents=True, exist_ok=True)
        (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
        (skill_dir / "references" / "classifier-system.md").write_text("# Classifier System Prompt\n\nbase\n", encoding="utf-8")
        (skill_dir / "references" / "evidence-rules.md").write_text("# Evidence Rules\n\nbase\n", encoding="utf-8")
        (skill_dir / "scripts" / "run_minor_detection_pipeline.py").write_text("print('ok')\n", encoding="utf-8")
        return skill_dir

    def test_build_validation_seed_version_name_uses_stamped_stable_name(self):
        version_name, run_tag = build_validation_seed_version_name(base_name="minor-detection-verifya", run_tag="20260322_140000")
        self.assertEqual(run_tag, "20260322_140000")
        self.assertEqual(version_name, "minor-detection-verifya-v0.1.0-20260322_140000")

    def test_create_mode_a_validation_seed_only_changes_rule_assets(self):
        root = make_test_dir(self)
        source_dir = self._make_skill(root)
        seed = create_mode_a_validation_seed(
            source_dir=source_dir,
            output_root=root / "tmp" / "skill_validation",
            base_name="minor-detection-verifya",
            run_tag="20260322_140000",
        )

        target_dir = seed.source_dir
        classifier_text = (target_dir / "references" / "classifier-system.md").read_text(encoding="utf-8")
        evidence_text = (target_dir / "references" / "evidence-rules.md").read_text(encoding="utf-8")
        pipeline_text = (target_dir / "scripts" / "run_minor_detection_pipeline.py").read_text(encoding="utf-8")
        source_classifier_text = (source_dir / "references" / "classifier-system.md").read_text(encoding="utf-8")

        self.assertIn("Temporary Validation Seed Override", classifier_text)
        self.assertIn("Temporary Validation Seed Override", evidence_text)
        self.assertEqual(pipeline_text, "print('ok')\n")
        self.assertNotIn("Temporary Validation Seed Override", source_classifier_text)
        self.assertTrue((target_dir / ".validation_seed.json").exists())

    def test_build_mode_a_validation_payload_includes_run_and_cleanup_commands(self):
        root = make_test_dir(self)
        source_dir = self._make_skill(root)
        payload = build_mode_a_validation_payload(
            source_dir=source_dir,
            output_root=root / "tmp" / "skill_validation",
            dataset_path=root / "data" / "benchmark" / "val.jsonl",
            base_name="minor-detection-verifya",
            run_tag="20260322_140000",
            max_samples=5,
            codex_model="gpt-5.4",
        )

        run_command = payload["run_command"]
        cleanup = payload["cleanup"]

        self.assertIn("python scripts/run_skill_iteration_loop.py", run_command)
        self.assertIn("--baseline-version minor-detection-verifya-v0.1.0-20260322_140000", run_command)
        self.assertIn("--max-samples 5", run_command)
        self.assertIn("--codex-model gpt-5.4", run_command)
        self.assertIn("cleanup_skill_versions.py", cleanup["cleanup_skills_command"])
        self.assertIn("20260322_140000", cleanup["cleanup_skills_command"])
        self.assertIn("minor-detection-verifya-v0.1.0-20260322_140000", cleanup["cleanup_seed_source_command"])


if __name__ == "__main__":
    unittest.main()

