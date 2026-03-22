# 妯″潡璇存槑锛?
# - skill 鐗堟湰蹇収鍜屾竻鐞嗚鍒欑殑涓撻」娴嬭瘯銆?
# - 鍒犻櫎鏃?skill 鐩綍鍓嶆渶濂戒繚璇佸畠浠嶇劧閫氳繃銆?

import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
TEST_TMP_ROOT = ROOT_DIR / ".tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

from src.skill_loop.versioning import build_version_inventory, delete_skill_versions, ensure_version_snapshot, iter_skill_versions, select_cleanup_targets


def make_test_dir(test_case: unittest.TestCase) -> Path:
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    test_case.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


class VersionSnapshotTests(unittest.TestCase):
    def test_ensure_version_snapshot_reuses_matching_snapshot(self):
        root = make_test_dir(self)
        source = root / "skills" / "minor-detection"
        target = root / "skills" / "minor-detection-v0.1.0"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("# demo\n", encoding="utf-8")

        ensure_version_snapshot(source, target)
        first_snapshot = (target / "SKILL.md").read_text(encoding="utf-8")

        ensure_version_snapshot(source, target)
        second_snapshot = (target / "SKILL.md").read_text(encoding="utf-8")

        self.assertEqual(first_snapshot, second_snapshot)
        self.assertTrue((target / ".skill_snapshot_manifest.json").exists())

    def test_ensure_version_snapshot_raises_on_stale_snapshot(self):
        root = make_test_dir(self)
        source = root / "skills" / "minor-detection"
        target = root / "skills" / "minor-detection-v0.1.0"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("# v1\n", encoding="utf-8")

        ensure_version_snapshot(source, target)
        (source / "references").mkdir(parents=True)
        (source / "references" / "schema-repair-template.md").write_text("repair\n", encoding="utf-8")

        with self.assertRaises(FileExistsError):
            ensure_version_snapshot(source, target)

    def test_ensure_version_snapshot_allows_source_and_target_to_be_same_dir(self):
        root = make_test_dir(self)
        target = root / "skills" / "minor-detection-v0.1.1-20260322_165515"
        target.mkdir(parents=True)
        (target / "SKILL.md").write_text("# approved\n", encoding="utf-8")

        result = ensure_version_snapshot(target, target)

        self.assertEqual(result, target)
        self.assertTrue((target / ".skill_snapshot_manifest.json").exists())

    def test_iter_skill_versions_lists_stable_and_candidate_dirs(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        (skills_root / "minor-detection-v0.1.0").mkdir(parents=True)
        (skills_root / "minor-detection-v0.1.1-rc001-20260322_120000").mkdir(parents=True)
        (skills_root / "other-skill-v0.1.0").mkdir(parents=True)

        versions = iter_skill_versions(skills_root, base_name="minor-detection")

        self.assertEqual([item["version"] for item in versions], ["minor-detection-v0.1.1-rc001-20260322_120000", "minor-detection-v0.1.0"])
        self.assertTrue(versions[0]["is_candidate"])
        self.assertFalse(versions[1]["is_candidate"])

    def test_select_cleanup_targets_keeps_latest_stable_and_removes_candidates(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        for name in (
            "minor-detection-v0.1.0",
            "minor-detection-v0.1.1",
            "minor-detection-v0.1.2-rc001-20260322_120000",
        ):
            (skills_root / name).mkdir(parents=True)

        targets = select_cleanup_targets(
            skills_root,
            base_name="minor-detection",
            keep_latest_stable=1,
            delete_candidates=True,
        )

        self.assertEqual(
            [path.name for path in targets],
            ["minor-detection-v0.1.0", "minor-detection-v0.1.2-rc001-20260322_120000"],
        )

    def test_build_version_inventory_includes_cleanup_preview(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        for name in (
            "minor-detection",
            "minor-detection-v0.1.0",
            "minor-detection-v0.1.1",
            "minor-detection-v0.1.2-rc001-20260322_120000",
        ):
            (skills_root / name).mkdir(parents=True)

        inventory = build_version_inventory(
            skills_root,
            base_name="minor-detection",
            active_version="minor-detection-v0.1.1",
            keep_latest_stable=1,
            delete_candidates=True,
        )

        self.assertEqual(inventory["active_version"], "minor-detection-v0.1.1")
        self.assertEqual(inventory["stable_versions"], ["minor-detection-v0.1.1", "minor-detection-v0.1.0"])
        self.assertEqual(inventory["candidate_versions"], ["minor-detection-v0.1.2-rc001-20260322_120000"])
        self.assertEqual(
            [Path(path).name for path in inventory["cleanup_preview"]["targets"]],
            ["minor-detection-v0.1.0", "minor-detection-v0.1.2-rc001-20260322_120000"],
        )
        self.assertIsNone(inventory["cleanup_preview"]["only_run_tag"])
        self.assertIsNone(inventory["cleanup_preview"]["exclude_run_tag"])

    def test_select_cleanup_targets_can_filter_by_only_run_tag(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        for name in (
            "minor-detection-v0.1.0",
            "minor-detection-v0.1.1-20260322_120000",
            "minor-detection-v0.1.1-rc001-20260322_120000",
            "minor-detection-v0.1.1-rc001-20260322_130000",
        ):
            (skills_root / name).mkdir(parents=True)

        targets = select_cleanup_targets(
            skills_root,
            base_name="minor-detection",
            keep_latest_stable=2,
            delete_candidates=True,
            only_run_tag="20260322_120000",
        )

        self.assertEqual(
            [path.name for path in targets],
            ["minor-detection-v0.1.1-rc001-20260322_120000"],
        )

    def test_select_cleanup_targets_can_exclude_run_tag(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        for name in (
            "minor-detection-v0.1.0",
            "minor-detection-v0.1.1",
            "minor-detection-v0.1.1-rc001-20260322_120000",
            "minor-detection-v0.1.1-rc001-20260322_130000",
        ):
            (skills_root / name).mkdir(parents=True)

        targets = select_cleanup_targets(
            skills_root,
            base_name="minor-detection",
            keep_latest_stable=2,
            delete_candidates=True,
            exclude_run_tag="20260322_130000",
        )

        self.assertEqual(
            [path.name for path in targets],
            ["minor-detection-v0.1.1-rc001-20260322_120000"],
        )

    def test_build_version_inventory_can_filter_listed_versions_by_only_run_tag(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        for name in (
            "minor-detection",
            "minor-detection-v0.1.0",
            "minor-detection-v0.1.1-20260322_120000",
            "minor-detection-v0.1.1-rc001-20260322_120000",
            "minor-detection-v0.1.1-20260322_130000",
            "minor-detection-v0.1.1-rc001-20260322_130000",
        ):
            (skills_root / name).mkdir(parents=True)

        inventory = build_version_inventory(
            skills_root,
            base_name="minor-detection",
            active_version="minor-detection-v0.1.1-20260322_130000",
            only_run_tag="20260322_120000",
            scope_active_version=True,
        )

        self.assertIsNone(inventory["active_version"])
        self.assertEqual(inventory["stable_versions"], ["minor-detection-v0.1.1-20260322_120000"])
        self.assertEqual(inventory["candidate_versions"], ["minor-detection-v0.1.1-rc001-20260322_120000"])
        self.assertEqual([item["version"] for item in inventory["versions"]], [
            "minor-detection-v0.1.1-rc001-20260322_120000",
            "minor-detection-v0.1.1-20260322_120000",
        ])
        self.assertEqual(inventory["cleanup_preview"]["only_run_tag"], "20260322_120000")

    def test_delete_skill_versions_removes_target_dirs(self):
        root = make_test_dir(self)
        target = root / "skills" / "minor-detection-v0.1.0"
        target.mkdir(parents=True)

        removed = delete_skill_versions([target])

        self.assertEqual(removed, [target])
        self.assertFalse(target.exists())

    def test_ensure_version_snapshot_refreshes_when_requested(self):
        root = make_test_dir(self)
        source = root / "skills" / "minor-detection"
        target = root / "skills" / "minor-detection-v0.1.0"
        source.mkdir(parents=True)
        (source / "SKILL.md").write_text("# v1\n", encoding="utf-8")

        ensure_version_snapshot(source, target)
        (source / "references").mkdir(parents=True)
        (source / "references" / "schema-repair-template.md").write_text("repair\n", encoding="utf-8")

        ensure_version_snapshot(source, target, refresh=True)

        self.assertTrue((target / "references" / "schema-repair-template.md").exists())


if __name__ == "__main__":
    unittest.main()

