import json
import io
import runpy
import subprocess
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.evolution.optimizer import SkillOptimizer
from src.trigger_eval.judge import judge_trigger_full_smoke_artifacts, judge_trigger_run_artifacts
from src.trigger_eval.runner import TriggerEvalCodexRunner
from src.trigger_eval.loop import TriggerDescriptionLoop, TriggerDescriptionLoopConfig


class TriggerEvalJudgeTests(unittest.TestCase):
    def _formal_output(self, *, is_minor: bool) -> dict:
        return {
            "decision": {
                "is_minor": is_minor,
                "minor_confidence": 0.91 if is_minor else 0.12,
                "confidence_band": "high" if is_minor else "low",
                "risk_level": "Medium",
            },
            "user_profile": {
                "age_range": "13-17岁" if is_minor else "18岁以上",
                "education_stage": "高中" if is_minor else "大学/本科",
                "identity_markers": ["学生"],
            },
            "icbo_features": {
                "intention": "求助",
                "cognition": "压力较大",
                "behavior_style": "情绪化表达",
                "opportunity_time": "未明确",
            },
            "evidence": {
                "direct_evidence": ["样本证据"],
                "historical_evidence": [],
                "retrieval_evidence": [],
                "time_evidence": [],
                "conflicting_signals": [],
            },
            "reasoning_summary": "ok",
            "trend": {"trajectory": [], "trend_summary": ""},
            "uncertainty_notes": [],
            "recommended_next_step": "safe_to_continue",
        }

    def _write_sample(
        self,
        run_root: Path,
        name: str,
        *,
        should_trigger: bool,
        predicted,
        skill_invoked: bool,
        invocation_status: str,
        launcher_success: bool,
        slice_name: str,
        expected_is_minor=None,
        final_output_json=None,
    ) -> None:
        sample_dir = run_root / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        (sample_dir / "sample_input.json").write_text(
            json.dumps(
                {
                    "id": name,
                    "slice": slice_name,
                    "scenario": "window_scan",
                    "query": "test query",
                    "skill_input_turns": [{"role": "user", "content": "test"}],
                    "skill_input_base_sample_id": f"base-{name}",
                    "expected_is_minor": expected_is_minor if expected_is_minor is not None else should_trigger,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (sample_dir / "gold.json").write_text(
            json.dumps(
                {
                    "sample_id": name,
                    "should_trigger": should_trigger,
                    "slice": slice_name,
                    "scenario": "window_scan",
                    "expected_is_minor": expected_is_minor if expected_is_minor is not None else should_trigger,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        parsed = None if predicted is None else {
            "should_trigger": predicted,
            "skill_invoked": skill_invoked,
            "decision_confidence": 0.8 if predicted else 0.2,
            "decision_reason": f"reason for {name}",
            "invocation_status": invocation_status,
        }
        (sample_dir / "agent_output.json").write_text(
            json.dumps(
                {
                    "raw_text": json.dumps(parsed, ensure_ascii=False) if parsed is not None else "not-json",
                    "parsed_json": parsed,
                    "json_valid": parsed is not None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (sample_dir / "run_metadata.json").write_text(
            json.dumps(
                {
                    "sample_id": name,
                    "returncode": 0,
                    "launcher_invoked": skill_invoked,
                    "launcher_success": launcher_success,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (sample_dir / "timing.json").write_text(json.dumps({"total_duration_seconds": 1.25}, ensure_ascii=False, indent=2), encoding="utf-8")
        (sample_dir / "observability.json").write_text(
            json.dumps(
                {
                    "issues": [],
                    "launcher": {"invoked": skill_invoked, "success": launcher_success, "status": "ok" if launcher_success else "failed"},
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (sample_dir / "transcript.md").write_text("transcript", encoding="utf-8")
        (sample_dir / "transcript.jsonl").write_text("", encoding="utf-8")
        (sample_dir / "stderr.log").write_text("", encoding="utf-8")
        (sample_dir / "tool_trace.json").write_text("[]", encoding="utf-8")
        (sample_dir / "query.txt").write_text("query", encoding="utf-8")
        (sample_dir / "payload.json").write_text(json.dumps({"mode": "single_session"}, ensure_ascii=False, indent=2), encoding="utf-8")
        (sample_dir / "launcher_result.json").write_text(
            json.dumps({"invoked": skill_invoked, "success": launcher_success, "status": "ok" if launcher_success else "failed"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (sample_dir / "pipeline_observability.json").write_text("{}", encoding="utf-8")
        skill_output_payload = final_output_json
        if skill_output_payload is None and skill_invoked and launcher_success:
            skill_output_payload = self._formal_output(is_minor=bool(expected_is_minor if expected_is_minor is not None else should_trigger))
        (sample_dir / "skill_output.json").write_text(
            json.dumps(
                {
                    "raw_text": json.dumps(skill_output_payload, ensure_ascii=False) if skill_output_payload is not None else "",
                    "parsed_json": skill_output_payload,
                    "json_valid": skill_output_payload is not None,
                    "json_source": "launcher_output" if skill_output_payload is not None else None,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def test_judge_trigger_run_artifacts_reports_metrics_and_slice_stats(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "run"
            run_root.mkdir(parents=True, exist_ok=True)
            self._write_sample(
                run_root,
                "s1",
                should_trigger=True,
                predicted=True,
                skill_invoked=True,
                invocation_status="invoked_success",
                launcher_success=True,
                slice_name="identity_explicit",
            )
            self._write_sample(
                run_root,
                "s2",
                should_trigger=False,
                predicted=True,
                skill_invoked=True,
                invocation_status="invoked_success",
                launcher_success=True,
                slice_name="adult_near_miss",
            )
            self._write_sample(
                run_root,
                "s3",
                should_trigger=True,
                predicted=False,
                skill_invoked=False,
                invocation_status="not_invoked",
                launcher_success=False,
                slice_name="school_context_strong",
            )

            judged = judge_trigger_run_artifacts(
                run_root=run_root,
                skill_version="minor-detection-v0.1.0",
                parent_version=None,
                dataset_name="trigger_eval",
                project_root=ROOT_DIR,
            )

            report = judged["report_payload"]
            self.assertEqual(report["task_type"], "trigger_eval")
            self.assertEqual(report["optimization_focus"], "description")
            self.assertEqual(report["evaluation_contract"], "trigger_decision_and_skill_invocation_success_only")
            self.assertEqual(report["launcher_contract"], "skill_invocation_probe_invoked_when_triggered")
            self.assertEqual(report["sample_count"], 3)
            self.assertEqual(report["metrics"]["true_positive"], 1)
            self.assertEqual(report["metrics"]["false_positive"], 1)
            self.assertEqual(report["metrics"]["false_negative"], 1)
            self.assertAlmostEqual(report["metrics"]["accuracy"], 1 / 3)
            self.assertAlmostEqual(report["metrics"]["precision"], 0.5)
            self.assertAlmostEqual(report["metrics"]["recall"], 0.5)
            self.assertAlmostEqual(report["metrics"]["f1_score"], 0.5)
            self.assertIn("identity_explicit", report["slice_stats"])
            self.assertIn("adult_near_miss", report["slice_stats"])
            self.assertIn("school_context_strong", report["slice_stats"])
            self.assertEqual(report["failure_type_counts"]["false_positive"], 1)
            self.assertEqual(report["failure_type_counts"]["false_negative"], 1)
            self.assertTrue(judged["report_path"].exists())
            self.assertTrue(judged["error_index_path"].exists())
            self.assertTrue(judged["protected_index_path"].exists())

    def test_invocation_success_rate_requires_decision_and_launcher_action_to_match(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "run"
            run_root.mkdir(parents=True, exist_ok=True)
            self._write_sample(
                run_root,
                "s1",
                should_trigger=True,
                predicted=True,
                skill_invoked=False,
                invocation_status="not_invoked",
                launcher_success=False,
                slice_name="identity_explicit",
            )

            judged = judge_trigger_run_artifacts(
                run_root=run_root,
                skill_version="minor-detection-v0.1.0",
                parent_version=None,
                dataset_name="trigger_eval",
                project_root=ROOT_DIR,
            )

            report = judged["report_payload"]
            self.assertAlmostEqual(report["invocation_success_rate"], 0.0)
            self.assertAlmostEqual(report["step_compliance_rate"], 0.0)
            self.assertEqual(report["failure_type_counts"]["step_compliance_failure"], 1)

    def test_judge_trigger_full_smoke_artifacts_reports_end_to_end_metrics(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_root = Path(tmp_dir) / "run"
            run_root.mkdir(parents=True, exist_ok=True)
            self._write_sample(
                run_root,
                "s1",
                should_trigger=True,
                predicted=True,
                skill_invoked=True,
                invocation_status="invoked_success",
                launcher_success=True,
                slice_name="identity_explicit",
                expected_is_minor=True,
                final_output_json=self._formal_output(is_minor=True),
            )
            self._write_sample(
                run_root,
                "s2",
                should_trigger=True,
                predicted=True,
                skill_invoked=True,
                invocation_status="invoked_success",
                launcher_success=True,
                slice_name="school_context_strong",
                expected_is_minor=True,
                final_output_json=self._formal_output(is_minor=False),
            )
            self._write_sample(
                run_root,
                "s3",
                should_trigger=False,
                predicted=False,
                skill_invoked=False,
                invocation_status="not_invoked",
                launcher_success=False,
                slice_name="adult_near_miss",
                expected_is_minor=False,
                final_output_json=None,
            )

            judged = judge_trigger_full_smoke_artifacts(
                run_root=run_root,
                skill_version="minor-detection-v0.1.0",
                dataset_name="trigger_eval",
                project_root=ROOT_DIR,
            )

            report = judged["report_payload"]
            self.assertEqual(report["task_type"], "trigger_full_smoke")
            self.assertEqual(report["evaluation_role"], "standalone_full_smoke")
            self.assertFalse(report["optimizer_feedback_enabled"])
            self.assertEqual(report["invoked_sample_count"], 2)
            self.assertAlmostEqual(report["full_output_json_valid_rate_on_invoked"], 1.0)
            self.assertAlmostEqual(report["full_output_schema_valid_rate_on_invoked"], 1.0)
            self.assertAlmostEqual(report["final_minor_accuracy_rate_on_invoked"], 0.5)
            self.assertAlmostEqual(report["positive_path_end_to_end_success_rate"], 0.5)
            self.assertAlmostEqual(report["end_to_end_success_rate"], 2 / 3)
            self.assertEqual(report["failure_type_counts"]["full_smoke_label_incorrect"], 1)
            self.assertTrue(judged["report_path"].exists())
            self.assertTrue(judged["error_index_path"].exists())


class TriggerEvalOptimizerTargetTests(unittest.TestCase):
    def test_trigger_eval_decision_errors_include_skill_description_target(self):
        optimizer = SkillOptimizer()
        targets = optimizer.resolve_packet_edit_targets(
            {
                "task_type": "trigger_eval",
                "failure_type_counts": {"false_negative": 2},
                "observed_issue_counts": {},
            }
        )
        self.assertEqual(targets, ["SKILL.md"])


class TriggerEvalScriptContractTests(unittest.TestCase):
    def test_run_trigger_description_validation_outputs_expected_payload(self):
        tmp_parent = ROOT_DIR / ".tmp_tests"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmp_dir:
            root = Path(tmp_dir)
            run_root = root / "reports" / "run"
            run_root.mkdir(parents=True, exist_ok=True)
            report_path = run_root / "report.json"
            report_path.write_text("{}", encoding="utf-8")

            runner_instance = mock.Mock(runner_mode="trigger_agent")
            runner_instance.run_dataset.return_value = run_root
            stdout_buffer = io.StringIO()

            with (
                mock.patch("src.trigger_eval.TriggerEvalCodexRunner", return_value=runner_instance),
                mock.patch(
                    "src.trigger_eval.judge_trigger_run_artifacts",
                    return_value={
                        "report_path": report_path,
                        "report_payload": {
                            "evaluation_contract": "trigger_decision_and_skill_invocation_success_only",
                            "metrics": {"accuracy": 0.75, "f1_score": 0.8},
                            "invocation_success_rate": 0.9,
                            "step_compliance_rate": 0.85,
                            "schema_validity_rate": 1.0,
                            "slice_stats": {"identity_explicit": {"sample_count": 2}},
                            "failure_type_counts": {"false_negative": 1},
                        },
                    },
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_trigger_description_validation.py",
                        "--version",
                        "minor-detection-v0.1.1",
                        "--dataset",
                        str(root / "data" / "final_validation.json"),
                        "--workspace",
                        str(root / "reports" / "trigger_description_validations"),
                    ],
                ),
                redirect_stdout(stdout_buffer),
            ):
                runpy.run_path(str(ROOT_DIR / "scripts" / "run_trigger_description_validation.py"), run_name="__main__")

            payload = json.loads(stdout_buffer.getvalue())
            self.assertEqual(payload["evaluation_role"], "standalone_description_validation")
            self.assertFalse(payload["optimization_feedback_enabled"])
            self.assertEqual(payload["runner_mode"], "trigger_agent")
            self.assertEqual(payload["evaluation_contract"], "trigger_decision_and_skill_invocation_success_only")
            self.assertEqual(payload["metrics"]["f1_score"], 0.8)
            self.assertEqual(payload["invocation_success_rate"], 0.9)
            self.assertEqual(payload["step_compliance_rate"], 0.85)
            self.assertEqual(payload["schema_validity_rate"], 1.0)
            self.assertIn("identity_explicit", payload["slice_stats"])
            self.assertEqual(payload["failure_type_counts"]["false_negative"], 1)

    def test_run_trigger_eval_outputs_expected_full_smoke_payload(self):
        tmp_parent = ROOT_DIR / ".tmp_tests"
        tmp_parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_parent) as tmp_dir:
            root = Path(tmp_dir)
            run_root = root / "reports" / "run"
            run_root.mkdir(parents=True, exist_ok=True)
            report_path = run_root / "report.json"
            report_path.write_text("{}", encoding="utf-8")

            runner_instance = mock.Mock(runner_mode="trigger_agent")
            runner_instance.run_dataset.return_value = run_root
            stdout_buffer = io.StringIO()

            with (
                mock.patch("src.trigger_eval.TriggerEvalCodexRunner", return_value=runner_instance),
                mock.patch(
                    "src.trigger_eval.judge_trigger_full_smoke_artifacts",
                    return_value={
                        "report_path": report_path,
                        "report_payload": {
                            "trigger_metrics": {"accuracy": 0.7, "f1_score": 0.72},
                            "full_output_json_valid_rate_on_invoked": 1.0,
                            "full_output_schema_valid_rate_on_invoked": 0.95,
                            "final_minor_accuracy_rate_on_invoked": 0.8,
                            "positive_path_end_to_end_success_rate": 0.75,
                            "end_to_end_success_rate": 0.7,
                            "slice_stats": {"identity_explicit": {"sample_count": 2}},
                        },
                    },
                ),
                mock.patch.object(
                    sys,
                    "argv",
                    [
                        "run_trigger_eval.py",
                        "--version",
                        "minor-detection-v0.1.1",
                        "--dataset",
                        str(root / "data" / "full_smoke.json"),
                        "--workspace",
                        str(root / "reports" / "trigger_eval_runs"),
                    ],
                ),
                redirect_stdout(stdout_buffer),
            ):
                runpy.run_path(str(ROOT_DIR / "scripts" / "run_trigger_eval.py"), run_name="__main__")

            payload = json.loads(stdout_buffer.getvalue())
            self.assertEqual(payload["evaluation_role"], "standalone_full_smoke")
            self.assertFalse(payload["optimization_feedback_enabled"])
            self.assertEqual(payload["runner_mode"], "trigger_agent")
            self.assertEqual(payload["trigger_metrics"]["f1_score"], 0.72)
            self.assertEqual(payload["full_output_json_valid_rate_on_invoked"], 1.0)
            self.assertEqual(payload["full_output_schema_valid_rate_on_invoked"], 0.95)
            self.assertEqual(payload["final_minor_accuracy_rate_on_invoked"], 0.8)
            self.assertEqual(payload["positive_path_end_to_end_success_rate"], 0.75)
            self.assertEqual(payload["end_to_end_success_rate"], 0.7)
            self.assertIn("identity_explicit", payload["slice_stats"])


class TriggerEvalLoopHookTests(unittest.TestCase):
    def test_trigger_eval_runner_uses_trigger_dataset_strata(self):
        key = TriggerEvalCodexRunner._resolve_stratum_key(
            {
                "scenario": "window_scan",
                "should_trigger": True,
                "slice": "identity_explicit",
                "source": "benchmark_val:social_pos",
                "age_bucket": "adult_18_22",
            }
        )
        self.assertEqual(key, "window_scan::trigger::identity_explicit")

    def test_trigger_description_loop_uses_injected_judge_function(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            run_root = root / "workspace" / "baseline" / "run-trigger-minor"
            run_root.mkdir(parents=True, exist_ok=True)
            (run_root / "run_manifest.json").write_text(
                json.dumps({"timing": {"total_wall_seconds": 1.23}, "counts": {"sample_count": 2}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            runner = mock.Mock()
            runner.run_dataset.return_value = run_root
            judge_fn = mock.Mock(
                return_value={
                    "report_path": root / "report.json",
                    "failure_packets_dir": root / "failure_packets",
                    "protected_packets_dir": root / "protected_packets",
                    "protected_index_path": root / "protected_index.jsonl",
                    "error_index_path": root / "error_index.jsonl",
                    "report_payload": {"sample_count": 2, "invocation_success_rate": 1.0},
                }
            )
            config = TriggerDescriptionLoopConfig(
                baseline_source_dir=root / "skills" / "minor-detection",
                baseline_version="minor-detection-v0.1.0",
                optimization_set_path=root / "data" / "trigger_eval_optimization.json",
                final_validation_set_path=root / "data" / "trigger_eval_final_validation.json",
                judge_fn=judge_fn,
            )
            loop = TriggerDescriptionLoop(config=config, runner=runner)
            (root / "skills" / "minor-detection-v0.1.0").mkdir(parents=True, exist_ok=True)

            with mock.patch("src.trigger_eval.loop.SKILLS_DIR", root / "skills"):
                result = loop._evaluate_version(version_name="minor-detection-v0.1.0", parent_version=None, workspace=root / "workspace" / "baseline")

            judge_fn.assert_called_once()
            self.assertEqual(result["runtime_summary"], {"total_wall_seconds": 1.23})
            self.assertEqual(result["runtime_counts"], {"sample_count": 2})

    def test_trigger_description_loop_summary_includes_manual_review_commands_and_artifact(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            skills_root = root / "skills"
            source_dir = skills_root / "minor-detection"
            baseline_dir = skills_root / "minor-detection-v0.1.0"
            source_dir.mkdir(parents=True, exist_ok=True)
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_eval = {
                "report_path": root / "baseline-report.json",
                "failure_packets_dir": root / "failure_packets",
                "protected_packets_dir": root / "protected_packets",
                "protected_index_path": root / "protected_index.jsonl",
                "runtime_summary": {"total_wall_seconds": 1.0},
                "runtime_counts": {"sample_count": 2},
                "report_payload": {
                    "sample_count": 2,
                    "invocation_success_rate": 1.0,
                },
            }
            candidate_eval = {
                "report_path": root / "candidate-report.json",
                "error_index_path": root / "candidate_error_index.jsonl",
                "runtime_summary": {"total_wall_seconds": 1.2},
                "runtime_counts": {"sample_count": 2},
            }

            optimizer = mock.Mock()

            def optimize_side_effect(**kwargs):
                candidate_version = kwargs["new_version"]
                (skills_root / candidate_version).mkdir(parents=True, exist_ok=True)
                return {
                    "success": True,
                    "current_version": "minor-detection-v0.1.0",
                    "new_version": candidate_version,
                    "edited_files": ["SKILL.md"],
                }

            optimizer.optimize_from_judge_artifacts.side_effect = optimize_side_effect
            optimizer.create_formal_skill_review_artifact.return_value = {
                "base_version": "minor-detection-v0.1.0",
                "candidate_version": "minor-detection-v0.1.1-20260325_120000",
                "review_diff_path": "skills/minor-detection-v0.1.1-20260325_120000/review/diff.md",
                "review_summary_path": "skills/minor-detection-v0.1.1-20260325_120000/review/summary.json",
            }

            config = TriggerDescriptionLoopConfig(
                baseline_source_dir=source_dir,
                baseline_version="minor-detection-v0.1.0",
                optimization_set_path=root / "data" / "trigger_eval_optimization.json",
                final_validation_set_path=root / "data" / "trigger_eval_final_validation.json",
                max_rounds=1,
                workspace_root=root / "workspace",
                compare_fn=mock.Mock(
                    return_value={"decision": "promote", "accepted_f1": 0.7, "candidate_f1": 0.9, "f1_delta": 0.2}
                ),
                manual_final_test_command_template=(
                    "python scripts/run_trigger_description_validation.py --version {version} "
                    "--dataset data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json --codex-model gpt-5.2"
                ),
                manual_smoke_validation_command_template=(
                    "python scripts/run_trigger_eval.py --version {version} "
                    "--dataset data/trigger_eval/minor_detection_trigger_eval_v1_final_validation_set.json --codex-model gpt-5.2"
                ),
            )
            loop = TriggerDescriptionLoop(config=config, runner=mock.Mock(runner_mode="trigger_agent"), optimizer=optimizer)
            workspace = root / "workspace" / "20260325_120000"
            workspace.mkdir(parents=True, exist_ok=True)

            with (
                mock.patch("src.trigger_eval.loop.SKILLS_DIR", skills_root),
                mock.patch("src.trigger_eval.loop.ensure_version_snapshot"),
                mock.patch("src.trigger_eval.loop.get_active_skill_version", return_value="minor-detection"),
                mock.patch.object(loop, "_workspace", return_value=workspace),
                mock.patch.object(loop, "_evaluate_version", side_effect=[baseline_eval, candidate_eval]),
            ):
                summary = loop.run()

            self.assertEqual(summary["final_version"], "minor-detection-v0.1.1-20260325_120000")
            self.assertTrue(summary["manual_review_required"])
            self.assertEqual(summary["manual_review_status"], "pending")
            self.assertEqual(summary["manual_review_base_version"], "minor-detection-v0.1.0")
            self.assertEqual(summary["manual_review_candidate_version"], "minor-detection-v0.1.1-20260325_120000")
            self.assertEqual(summary["loop_type"], "trigger_description_optimization")
            self.assertEqual(summary["optimization_scope"], "trigger_decision_and_skill_invocation_success_only")
            self.assertEqual(summary["optimization_target"], "SKILL.md frontmatter description")
            self.assertEqual(summary["evaluation_contract"], "trigger_decision_and_skill_invocation_success_only")
            self.assertTrue(summary["optimizer_feedback_enabled"])
            self.assertEqual(summary["post_loop_validation_role"], "standalone_description_validation")
            self.assertTrue(str(summary["optimization_set"]).endswith("data/trigger_eval_optimization.json"))
            self.assertTrue(str(summary["final_validation_set"]).endswith("data/trigger_eval_final_validation.json"))
            self.assertEqual(summary["manual_smoke_validation_script"], "scripts/run_trigger_eval.py")
            self.assertIn("scripts/run_trigger_eval.py --version minor-detection-v0.1.1-20260325_120000", summary["manual_smoke_validation_command"])
            self.assertIn("scripts/run_trigger_description_validation.py --version minor-detection-v0.1.1-20260325_120000", summary["manual_final_test_command"])
            self.assertIn("--review-decision approve", summary["manual_review_approve_command"])
            self.assertIn("--review-decision reject", summary["manual_review_reject_command"])
            self.assertEqual(summary["review_artifact"]["candidate_version"], "minor-detection-v0.1.1-20260325_120000")
            self.assertEqual(summary["rounds"][0]["comparison"]["decision"], "promote")
            self.assertEqual(summary["rounds"][0]["promoted_to"], "minor-detection-v0.1.1-20260325_120000")
            self.assertEqual(summary["version_management"]["run_tag"], "20260325_120000")

            summary_path = workspace / "loop_summary.json"
            self.assertTrue(summary_path.exists())
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["loop_type"], "trigger_description_optimization")
            self.assertEqual(payload["optimization_scope"], "trigger_decision_and_skill_invocation_success_only")
            self.assertEqual(payload["evaluation_contract"], "trigger_decision_and_skill_invocation_success_only")
            self.assertTrue(str(payload["optimization_set"]).endswith("data/trigger_eval_optimization.json"))
            self.assertTrue(str(payload["final_validation_set"]).endswith("data/trigger_eval_final_validation.json"))
            self.assertIn("scripts/run_trigger_eval.py --version minor-detection-v0.1.1-20260325_120000", payload["manual_smoke_validation_command"])
            self.assertIn("scripts/run_trigger_description_validation.py --version minor-detection-v0.1.1-20260325_120000", payload["manual_final_test_command"])
            self.assertTrue(payload["manual_review_required"])
            self.assertEqual(payload["manual_review_candidate_version"], "minor-detection-v0.1.1-20260325_120000")
            self.assertIn("--review-decision approve", payload["manual_review_approve_command"])
            self.assertEqual(payload["review_artifact"]["candidate_version"], "minor-detection-v0.1.1-20260325_120000")

    def test_trigger_description_loop_without_promotion_has_no_manual_review_outputs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            skills_root = root / "skills"
            source_dir = skills_root / "minor-detection"
            baseline_dir = skills_root / "minor-detection-v0.1.0"
            source_dir.mkdir(parents=True, exist_ok=True)
            baseline_dir.mkdir(parents=True, exist_ok=True)

            baseline_eval = {
                "report_path": root / "baseline-report.json",
                "failure_packets_dir": root / "failure_packets",
                "protected_packets_dir": root / "protected_packets",
                "protected_index_path": root / "protected_index.jsonl",
                "runtime_summary": {"total_wall_seconds": 1.0},
                "runtime_counts": {"sample_count": 2},
                "report_payload": {
                    "sample_count": 2,
                    "invocation_success_rate": 1.0,
                },
            }
            candidate_eval = {
                "report_path": root / "candidate-report.json",
                "error_index_path": root / "candidate_error_index.jsonl",
                "runtime_summary": {"total_wall_seconds": 1.2},
                "runtime_counts": {"sample_count": 2},
            }

            optimizer = mock.Mock()

            def optimize_side_effect(**kwargs):
                candidate_version = kwargs["new_version"]
                (skills_root / candidate_version).mkdir(parents=True, exist_ok=True)
                return {
                    "success": True,
                    "current_version": "minor-detection-v0.1.0",
                    "new_version": candidate_version,
                    "edited_files": ["SKILL.md"],
                }

            optimizer.optimize_from_judge_artifacts.side_effect = optimize_side_effect

            config = TriggerDescriptionLoopConfig(
                baseline_source_dir=source_dir,
                baseline_version="minor-detection-v0.1.0",
                optimization_set_path=root / "data" / "trigger_eval_optimization.json",
                final_validation_set_path=root / "data" / "trigger_eval_final_validation.json",
                max_rounds=1,
                workspace_root=root / "workspace",
                compare_fn=mock.Mock(
                    return_value={"decision": "rollback", "accepted_f1": 0.8, "candidate_f1": 0.7, "f1_delta": -0.1}
                ),
            )
            loop = TriggerDescriptionLoop(config=config, runner=mock.Mock(runner_mode="trigger_agent"), optimizer=optimizer)
            workspace = root / "workspace" / "20260325_121500"
            workspace.mkdir(parents=True, exist_ok=True)

            with (
                mock.patch("src.trigger_eval.loop.SKILLS_DIR", skills_root),
                mock.patch("src.trigger_eval.loop.ensure_version_snapshot"),
                mock.patch("src.trigger_eval.loop.get_active_skill_version", return_value="minor-detection"),
                mock.patch.object(loop, "_workspace", return_value=workspace),
                mock.patch.object(loop, "_evaluate_version", side_effect=[baseline_eval, candidate_eval]),
            ):
                summary = loop.run()

            self.assertEqual(summary["final_version"], "minor-detection-v0.1.0")
            self.assertFalse(summary["manual_review_required"])
            self.assertEqual(summary["manual_review_status"], "not_required")
            self.assertIsNone(summary["manual_review_candidate_version"])
            self.assertIsNone(summary["manual_smoke_validation_command"])
            self.assertIsNone(summary["manual_final_test_command"])
            self.assertIsNone(summary["manual_review_approve_command"])
            self.assertIsNone(summary["manual_review_reject_command"])
            self.assertIsNone(summary["review_artifact"])


class TriggerEvalRunnerLoggingTests(unittest.TestCase):
    def test_trigger_eval_runner_emits_progress_logs(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            skill_dir = root / "skills" / "minor-detection-v0.1.0"
            skill_dir.mkdir(parents=True, exist_ok=True)
            dataset_path = root / "data" / "trigger_eval.json"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("{}", encoding="utf-8")

            runner = TriggerEvalCodexRunner()

            def fake_command_runner(command, **kwargs):
                return subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

            runner.command_runner = fake_command_runner

            samples = [
                {
                    "id": "sample-alpha",
                    "query": "query a",
                    "should_trigger": True,
                    "slice": "identity_explicit",
                    "scenario": "window_scan",
                    "skill_input_turns": [{"role": "user", "content": "我高二了"}],
                    "skill_input_base_sample_id": "alpha-base",
                    "expected_is_minor": True,
                },
                {
                    "id": "sample-beta",
                    "query": "query b",
                    "should_trigger": False,
                    "slice": "adult_near_miss",
                    "scenario": "direct_request",
                    "skill_input_turns": [{"role": "user", "content": "我大四了"}],
                    "skill_input_base_sample_id": "beta-base",
                    "expected_is_minor": False,
                },
            ]

            log_stream = io.StringIO()
            with (
                mock.patch("src.trigger_eval.runner.validate_skill_schema_contract", return_value={"ok": True, "warnings": []}),
                mock.patch.object(runner, "_build_isolated_codex_home", return_value=root / "isolated" / ".codex"),
                mock.patch.object(runner, "_install_skill", return_value=root / "installed-skill"),
                mock.patch.object(runner, "_output_schema_path", return_value=root / "schema.json"),
                mock.patch.object(runner, "_load_dataset_samples", return_value=samples),
                mock.patch.object(runner, "_load_json_file", return_value={"invoked": True, "success": True, "status": "ok"}),
                mock.patch.object(runner, "_write_skill_launcher"),
                mock.patch.object(runner, "_write_agent_output", return_value={"json_valid": True}),
                redirect_stderr(log_stream),
            ):
                run_root = runner.run_dataset(
                    project_root=root,
                    skill_source_dir=skill_dir,
                    skill_version="minor-detection-v0.1.0",
                    dataset_path=dataset_path,
                    workspace_dir=root / "workspace",
                )

            stderr_text = log_stream.getvalue()
            self.assertIn("[trigger-eval-runner] schema_consistency ok=True warnings=0", stderr_text)
            self.assertIn("dataset=", stderr_text)
            self.assertIn("total_samples=2", stderr_text)
            self.assertIn("selected_samples=2 strategy=all execution_mode=sandbox", stderr_text)
            self.assertIn("sample 1/2 start sample-alpha", stderr_text)
            self.assertIn("sample 1/2 done returncode=0 json_valid=True launcher_invoked=True launcher_success=True", stderr_text)
            self.assertIn("sample 2/2 start sample-beta", stderr_text)
            self.assertIn("completed samples=2 json_valid=2 launcher_invoked=2 launcher_success=2", stderr_text)

            manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["counts"]["sample_count"], 2)
            self.assertEqual(manifest["counts"]["json_valid_count"], 2)
            self.assertEqual(manifest["counts"]["launcher_invoked_count"], 2)
            self.assertEqual(manifest["counts"]["launcher_success_count"], 2)


if __name__ == "__main__":
    unittest.main()
