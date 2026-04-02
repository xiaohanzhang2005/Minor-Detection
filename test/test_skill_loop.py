# 模块说明：
# - 当前主链最重要的回归测试文件。
# - 覆盖 runner、judge、compare、versioning 和 loop orchestration。

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
TEST_TMP_ROOT = ROOT_DIR / ".tmp_tests"
TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)

from src.evolution.optimizer import SkillOptimizer
from src.skill_loop.compare import compare_reports
from src.skill_loop.judge import calc_default_max_errors, judge_run_artifacts
from src.skill_loop.loop import SkillAgentLoop
from src.skill_loop.packaging import install_skill_snapshot
from src.skill_loop.direct_runner import DirectRunnerConfig, DirectSkillRunner
from src.skill_loop.schema_consistency import validate_skill_schema_contract
from src.skill_loop.runner import (
    PIPELINE_OBSERVABILITY_PREFIX,
    CodexRunnerConfig,
    CodexSkillRunner,
    _build_timing_summary,
    _detect_fatal_agent_error,
    _run_dirname,
    _strict_json_schema,
)
from src.skill_loop.versioning import (
    build_candidate_version_name,
    next_patch_version_name,
    next_available_candidate_version_name,
    parse_version_name,
)


class DummyLLMClient:
    def chat(self, messages, temperature=0.7):
        return messages[-1]["content"]


def make_test_dir(test_case: unittest.TestCase) -> Path:
    path = TEST_TMP_ROOT / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    test_case.addCleanup(lambda: shutil.rmtree(path, ignore_errors=True))
    return path


class SkillLoopVersionTests(unittest.TestCase):
    def test_version_helpers(self):
        parsed = parse_version_name("minor-detection-v0.1.1-rc003-20260322_123456")
        self.assertEqual(parsed["patch"], 1)
        self.assertEqual(parsed["rc"], 3)
        self.assertEqual(parsed["run_tag"], "20260322_123456")
        self.assertEqual(next_patch_version_name("minor-detection-v0.1.1-20260322_123456"), "minor-detection-v0.1.2")
        self.assertEqual(
            build_candidate_version_name("minor-detection-v0.1.2", 4, run_tag="20260322_123456"),
            "minor-detection-v0.1.2-rc004-20260322_123456",
        )

    def test_next_available_candidate_version_name_skips_existing_dirs(self):
        root = make_test_dir(self)
        (root / "minor-detection-v0.1.1-rc001-20260322_120000").mkdir()
        (root / "minor-detection-v0.1.1-rc002-20260322_120000").mkdir()
        self.assertEqual(
            next_available_candidate_version_name("minor-detection-v0.1.1", root, run_tag="20260322_120000"),
            "minor-detection-v0.1.1-rc003-20260322_120000",
        )
        self.assertEqual(
            next_available_candidate_version_name("minor-detection-v0.1.1", root, run_tag="20260322_130000"),
            "minor-detection-v0.1.1-rc001-20260322_130000",
        )

    def test_default_max_errors_formula(self):
        self.assertEqual(calc_default_max_errors(total_errors=80, eval_size=100), 12)
        self.assertEqual(calc_default_max_errors(total_errors=80, eval_size=400), 36)
        self.assertEqual(calc_default_max_errors(total_errors=5, eval_size=400), 5)


class SkillLoopRunnerTests(unittest.TestCase):
    def test_codex_command_contains_output_schema_and_add_dir(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex"))
        command = runner._build_codex_command(
            workspace_dir=Path("workspace"),
            output_schema_path=Path("workspace/schema.json"),
            final_output_path=Path("workspace/final.txt"),
            installed_skill_dir=Path(".codex/skills/minor-detection-v0.1.0"),
        )
        self.assertNotIn("--output-schema", command)
        self.assertIn("--add-dir", command)
        self.assertTrue(command[0].lower().endswith("cmd.exe"))

    def test_codex_command_wraps_cmd_entrypoint_on_windows(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="tools/codex.cmd"))
        command = runner._build_codex_command(
            workspace_dir=Path("workspace"),
            output_schema_path=Path("workspace/schema.json"),
            final_output_path=Path("workspace/final.txt"),
            installed_skill_dir=Path(".codex/skills/minor-detection-v0.1.0"),
        )
        self.assertTrue(command[0].lower().endswith("cmd.exe"))
        self.assertEqual(command[1], "/c")
        self.assertEqual(command[2], "tools/codex.cmd")
        self.assertIn("exec", command)

    def test_codex_command_resolves_bare_codex_to_powershell_shim(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex"))

        def fake_which(name: str):
            mapping = {
                "codex.ps1": "tools/codex.ps1",
            }
            return mapping.get(name)

        with mock.patch("src.skill_loop.runner.shutil.which", side_effect=fake_which):
            command = runner._build_codex_command(
                workspace_dir=Path("workspace"),
                output_schema_path=Path("workspace/schema.json"),
                final_output_path=Path("workspace/final.txt"),
                installed_skill_dir=Path(".codex/skills/minor-detection-v0.1.0"),
            )

        self.assertTrue(command[0].lower().endswith("powershell.exe"))
        self.assertEqual(command[1], "-NoProfile")
        self.assertEqual(command[2], "-File")
        self.assertEqual(command[3], "tools/codex.ps1")
        self.assertIn("exec", command)

    def test_codex_command_uses_bypass_mode_without_sandbox_flag(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex", execution_mode="bypass"))
        command = runner._build_codex_command(
            workspace_dir=Path("workspace"),
            output_schema_path=Path("workspace/schema.json"),
            final_output_path=Path("workspace/final.txt"),
            installed_skill_dir=Path(".codex/skills/minor-detection-v0.1.0"),
        )
        self.assertIn("--dangerously-bypass-approvals-and-sandbox", command)
        self.assertNotIn("--sandbox", command)

    def test_build_prompt_points_agent_to_prepared_launcher(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig())
        prompt = runner._build_prompt(
            payload_path=Path("payload.json"),
            launcher_path=Path("run_skill_once.py"),
        )
        self.assertIn("payload.json", prompt)
        self.assertIn("run_skill_once.py", prompt)
        self.assertIn("exactly once", prompt)

    def test_codex_command_includes_model_override_when_configured(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex", codex_model="gpt-5-mini"))
        command = runner._build_codex_command(
            workspace_dir=Path("workspace"),
            output_schema_path=Path("workspace/schema.json"),
            final_output_path=Path("workspace/final.txt"),
            installed_skill_dir=Path(".codex/skills/minor-detection-v0.1.0"),
        )
        self.assertIn("--model", command)
        self.assertIn("gpt-5-mini", command)

    def test_cli_agent_backend_uses_plain_executable_by_default(self):
        runner = CodexSkillRunner(
            config=CodexRunnerConfig(
                agent_backend="cli",
                agent_cmd="vendor-agent",
            )
        )
        command = runner._build_agent_command(
            workspace_dir=Path("workspace"),
            output_schema_path=Path("workspace/schema.json"),
            final_output_path=Path("workspace/final.txt"),
            installed_skill_dir=Path(".codex/skills/minor-detection-v0.1.0"),
            prompt_file_path=Path("workspace/agent_prompt.txt"),
        )
        self.assertEqual(command, ["vendor-agent"])

    def test_cli_agent_backend_formats_template_placeholders(self):
        runner = CodexSkillRunner(
            config=CodexRunnerConfig(
                agent_backend="cli",
                agent_cmd="vendor-agent",
                agent_args_template="{agent_cmd} --workspace {workspace_dir} --prompt-file {prompt_file} --output-file {final_output_path} --model {agent_model}",
                agent_model="vendor-pro-1",
            )
        )
        command = runner._build_agent_command(
            workspace_dir=Path("workspace"),
            output_schema_path=Path("workspace/schema.json"),
            final_output_path=Path("workspace/final.txt"),
            installed_skill_dir=Path(".codex/skills/minor-detection-v0.1.0"),
            prompt_file_path=Path("workspace/agent_prompt.txt"),
        )
        self.assertEqual(
            command,
            [
                "vendor-agent",
                "--workspace",
                "workspace",
                "--prompt-file",
                "workspace/agent_prompt.txt",
                "--output-file",
                "workspace/final.txt",
                "--model",
                "vendor-pro-1",
            ],
        )

    def test_write_agent_output_strips_observability_marker_lines(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig())
        root = make_test_dir(self)
        final_output_path = root / "final_output.txt"
        target_path = root / "agent_output.json"
        final_output_path.write_text(
            json.dumps({"decision": {"is_minor": True}}, ensure_ascii=False)
            + "\n"
            + f"{PIPELINE_OBSERVABILITY_PREFIX} {{\"summary\":{{}}}}",
            encoding="utf-8",
        )

        payload = runner._write_agent_output(
            final_output_path,
            target_path,
            launcher_result={"success": True, "status": "ok"},
        )

        self.assertTrue(payload["json_valid"])
        self.assertEqual(payload["parsed_json"]["decision"]["is_minor"], True)

    def test_strict_json_schema_sets_additional_properties_false(self):
        schema = _strict_json_schema(
            {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "object",
                        "properties": {
                            "is_minor": {"type": "boolean"},
                        },
                    }
                },
                "$defs": {
                    "Nested": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                        },
                    }
                },
            }
        )
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["required"], ["decision"])
        self.assertFalse(schema["properties"]["decision"]["additionalProperties"])
        self.assertEqual(schema["properties"]["decision"]["required"], ["is_minor"])
        self.assertFalse(schema["$defs"]["Nested"]["additionalProperties"])
        self.assertEqual(schema["$defs"]["Nested"]["required"], ["value"])

    def test_select_samples_stratified_matches_old_eval_behavior(self):
        samples = [
            {"sample_id": "a1", "source": "social", "is_minor": True, "age": 14},
            {"sample_id": "a2", "source": "social", "is_minor": True, "age": 15},
            {"sample_id": "b1", "source": "social", "is_minor": False, "age": 21},
            {"sample_id": "b2", "source": "social", "is_minor": False, "age": 22},
            {"sample_id": "c1", "source": "knowledge", "is_minor": True, "age": 11},
            {"sample_id": "c2", "source": "knowledge", "is_minor": True, "age": 12},
            {"sample_id": "d1", "source": "knowledge", "is_minor": False, "age": 28},
            {"sample_id": "d2", "source": "knowledge", "is_minor": False, "age": 31},
        ]

        selected, sampling = CodexSkillRunner._select_samples(
            samples,
            max_samples=4,
            strategy="stratified",
            sample_seed=42,
        )

        self.assertEqual(len(selected), 4)
        self.assertEqual(sampling["strategy"], "stratified")
        self.assertEqual(sampling["selected"], 4)
        self.assertEqual(
            sampling["by_stratum"],
            {
                "knowledge::adult_27_plus": 1,
                "knowledge::minor_07_12": 1,
                "social::adult_18_22": 1,
                "social::minor_13_15": 1,
            },
        )

    def test_build_timing_summary_aggregates_samples(self):
        summary = _build_timing_summary(
            [
                {"duration_seconds": 10.0},
                {"duration_seconds": 20.0},
                {"duration_seconds": 30.0},
            ],
            total_wall_seconds=65.0,
        )
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["sample_total_seconds"], 60.0)
        self.assertEqual(summary["avg_sample_seconds"], 20.0)
        self.assertEqual(summary["median_sample_seconds"], 20.0)
        self.assertEqual(summary["min_sample_seconds"], 10.0)
        self.assertEqual(summary["max_sample_seconds"], 30.0)
        self.assertEqual(summary["total_wall_seconds"], 65.0)

    def test_strict_json_schema_strips_ref_siblings_and_metadata(self):
        schema = _strict_json_schema(
            {
                "type": "object",
                "title": "Root",
                "description": "root schema",
                "properties": {
                    "confidence_band": {
                        "$ref": "#/$defs/ConfidenceBand",
                        "description": "band",
                    }
                },
                "$defs": {
                    "ConfidenceBand": {
                        "type": "string",
                        "title": "ConfidenceBand",
                        "enum": ["low", "medium", "high"],
                    }
                },
            }
        )
        self.assertNotIn("title", schema)
        self.assertNotIn("description", schema)
        self.assertEqual(schema["properties"]["confidence_band"], {"$ref": "#/$defs/ConfidenceBand"})
        self.assertNotIn("title", schema["$defs"]["ConfidenceBand"])

    def test_install_skill_snapshot_copies_validated_skill_source(self):
        root = make_test_dir(self)
        project_root = root / "project"
        skill_dir = project_root / "skills" / "demo-skill"
        validator_dir = project_root / "claude-skill-creator" / "scripts"
        validator_dir.mkdir(parents=True, exist_ok=True)
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: demo\n---\n\n# Demo\n",
            encoding="utf-8",
        )
        (skill_dir / "notes.txt").write_text("keep", encoding="utf-8")
        pycache_dir = skill_dir / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "tmp.pyc").write_text("compiled", encoding="utf-8")
        (validator_dir / "quick_validate.py").write_text(
            "from pathlib import Path\n\ndef validate_skill(skill_path):\n    return (Path(skill_path) / 'SKILL.md').exists(), 'ok'\n",
            encoding="utf-8",
        )

        installed_dir = install_skill_snapshot(
            project_root=project_root,
            skill_dir=skill_dir,
            target_dir=root / "isolated" / ".codex" / "skills",
        )

        self.assertTrue((installed_dir / "SKILL.md").exists())
        self.assertTrue((installed_dir / "notes.txt").exists())
        self.assertFalse((installed_dir / "__pycache__").exists())

    def test_install_skill_snapshot_shortens_installed_dir_name(self):
        root = make_test_dir(self)
        project_root = root / "project"
        long_name = "minor-detection-verifya-v0.1.1-rc001-20260322_142814"
        skill_dir = project_root / "skills" / long_name
        validator_dir = project_root / "claude-skill-creator" / "scripts"
        validator_dir.mkdir(parents=True, exist_ok=True)
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: demo\n---\n\n# Demo\n",
            encoding="utf-8",
        )
        (validator_dir / "quick_validate.py").write_text(
            "from pathlib import Path\n\ndef validate_skill(skill_path):\n    return (Path(skill_path) / 'SKILL.md').exists(), 'ok'\n",
            encoding="utf-8",
        )

        installed_dir = install_skill_snapshot(
            project_root=project_root,
            skill_dir=skill_dir,
            target_dir=root / "isolated" / ".codex" / "skills",
        )

        self.assertTrue((installed_dir / "SKILL.md").exists())
        self.assertLess(len(installed_dir.name), len(long_name))
        self.assertLessEqual(len(installed_dir.name), 24)

    def test_run_dirname_shortens_long_skill_versions(self):
        dirname = _run_dirname("minor-detection-verifya-v0.1.1-rc001-20260322_142814")
        self.assertTrue(dirname.startswith("run-"))
        self.assertLessEqual(len(dirname), 28)

    def test_runner_observability_captures_fallback_and_quoting_failure(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex.cmd"))
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "python scripts/extract_time_features.py --timestamp '2025-11-20 23:20'",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": '{"local_time": "2025-11-20 23:20:00"}',
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "python scripts/retrieve_cases.py --query 'x' --time-features-json '{broken}'",
                    "status": "failed",
                    "exit_code": 1,
                    "aggregated_output": "retrieve_cases.py: error: unrecognized arguments: broken",
                },
            },
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "python scripts/retrieve_cases.py --query 'x'",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": json.dumps(
                        {
                            "status": "ok",
                            "mode": "fallback:ConnectError",
                            "count": 3,
                            "message": "[WinError 10013] socket access was blocked",
                        },
                        ensure_ascii=False,
                    ),
                },
            },
        ]
        observability = runner._build_observability(
            events=events,
            stderr_text="",
            embedding_runtime={"api_key_present": True, "base_url": "https://aihubmix.com/v1", "embedding_model": "text-embedding-3-small"},
        )
        self.assertIn("shell_quoting_failure", observability["issues"])
        self.assertIn("retrieval_fallback", observability["issues"])
        self.assertIn("retrieval_network_blocked", observability["issues"])
        self.assertEqual(observability["retrieval"]["mode"], "fallback:ConnectError")
        self.assertTrue(observability["time_processing"]["successful"])

    def test_detect_fatal_agent_error_flags_provider_outage(self):
        fatal_error = _detect_fatal_agent_error(
            "Reconnecting... 1/5 (unexpected status 503 Service Unavailable: model unavailable, url: https://code.ddsst.online/v1/responses)"
        )
        self.assertEqual(fatal_error, "agent_model_unavailable")

    def test_detect_fatal_agent_error_flags_skill_load_failure(self):
        fatal_error = _detect_fatal_agent_error(
            "failed to load skill x\\SKILL.md: missing YAML frontmatter delimited by ---"
        )
        self.assertEqual(fatal_error, "skill_load_failure")

    def test_runner_observability_merges_pipeline_marker_from_command_output(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex.cmd"))
        marker = '[MINOR_PIPELINE_OBSERVABILITY] ' + json.dumps(
            {
                "summary": {"failed_script_calls": 0},
                "time_processing": {"attempted": True, "successful": True, "failure_count": 0, "mode": "script"},
                "retrieval": {
                    "attempted": True,
                    "successful": True,
                    "mode": "embedding",
                    "used_fallback": False,
                    "fallback_reason": None,
                    "retrieved_count": 3,
                    "failure_count": 0,
                    "quoting_failure_detected": False,
                    "network_error_detected": False,
                    "message": None,
                },
                "issues": [],
                "script_calls": [
                    {"script_name": "run_minor_detection_pipeline.py", "failed": False},
                    {"script_name": "extract_time_features.py", "failed": False},
                    {"script_name": "retrieve_cases.py", "failed": False},
                ],
            },
            ensure_ascii=False,
        )
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "python scripts/run_minor_detection_pipeline.py --payload-file payload.json",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": '{\"decision\": {\"is_minor\": true}}\n' + marker,
                },
            },
        ]
        observability = runner._build_observability(
            events=events,
            stderr_text="",
            embedding_runtime={"api_key_present": True, "base_url": "https://aihubmix.com/v1", "embedding_model": "text-embedding-3-small"},
        )
        self.assertEqual(observability["retrieval"]["mode"], "embedding")
        self.assertTrue(observability["time_processing"]["successful"])

    def test_runner_observability_merges_pipeline_marker_from_launcher_command_output(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex.cmd"))
        marker = '[MINOR_PIPELINE_OBSERVABILITY] ' + json.dumps(
            {
                "summary": {"failed_script_calls": 0},
                "time_processing": {"attempted": True, "successful": True, "failure_count": 0, "mode": "script"},
                "retrieval": {
                    "attempted": True,
                    "successful": True,
                    "mode": "embedding",
                    "used_fallback": False,
                    "fallback_reason": None,
                    "retrieved_count": 3,
                    "failure_count": 0,
                    "quoting_failure_detected": False,
                    "network_error_detected": False,
                    "message": None,
                },
                "issues": [],
                "script_calls": [
                    {"script_name": "run_minor_detection_pipeline.py", "failed": False},
                    {"script_name": "extract_time_features.py", "failed": False},
                    {"script_name": "retrieve_cases.py", "failed": False},
                ],
            },
            ensure_ascii=False,
        )
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "\"powershell.exe\" -Command 'python run_skill_once.py'",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": '{"decision": {"is_minor": true}}' + marker,
                },
            },
        ]
        observability = runner._build_observability(
            events=events,
            stderr_text="",
            embedding_runtime={"api_key_present": True, "base_url": "https://aihubmix.com/v1", "embedding_model": "text-embedding-3-small"},
        )
        self.assertEqual(observability["retrieval"]["mode"], "embedding")
        self.assertTrue(observability["time_processing"]["successful"])

    def test_runner_observability_merges_pipeline_marker(self):
        runner = CodexSkillRunner(config=CodexRunnerConfig(codex_cmd="codex.cmd"))
        events = [
            {
                "type": "item.completed",
                "item": {
                    "type": "command_execution",
                    "command": "python scripts/run_minor_detection_pipeline.py --payload-file payload.json",
                    "status": "completed",
                    "exit_code": 0,
                    "aggregated_output": '{"decision": {"is_minor": true}}',
                },
            },
        ]
        stderr_text = '[MINOR_PIPELINE_OBSERVABILITY] ' + json.dumps(
            {
                "summary": {"failed_script_calls": 0},
                "time_processing": {"attempted": True, "successful": True, "failure_count": 0, "mode": "script"},
                "retrieval": {
                    "attempted": True,
                    "successful": True,
                    "mode": "embedding",
                    "used_fallback": False,
                    "fallback_reason": None,
                    "retrieved_count": 2,
                    "failure_count": 0,
                    "quoting_failure_detected": False,
                    "network_error_detected": False,
                    "message": None,
                },
                "issues": [],
                "script_calls": [
                    {"script_name": "run_minor_detection_pipeline.py", "failed": False},
                    {"script_name": "extract_time_features.py", "failed": False},
                    {"script_name": "retrieve_cases.py", "failed": False},
                ],
            },
            ensure_ascii=False,
        )
        observability = runner._build_observability(
            events=events,
            stderr_text=stderr_text,
            embedding_runtime={"api_key_present": True, "base_url": "https://aihubmix.com/v1", "embedding_model": "text-embedding-3-small"},
        )
        self.assertEqual(observability["retrieval"]["mode"], "embedding")
        self.assertTrue(observability["time_processing"]["successful"])
        self.assertFalse(observability["summary"]["stderr_non_empty"])
        self.assertTrue(any(item.get("script_name") == "retrieve_cases.py" for item in observability["script_calls"]))


class DirectSkillRunnerTests(unittest.TestCase):
    def test_direct_runner_writes_judge_compatible_artifacts(self):
        root = make_test_dir(self)
        skill_dir = root / "skills" / "minor-detection-v0.1.0"
        scripts_dir = skill_dir / "scripts"
        references_dir = skill_dir / "references"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        references_dir.mkdir(parents=True, exist_ok=True)
        (scripts_dir / "run_minor_detection_pipeline.py").write_text("# placeholder\n", encoding="utf-8")
        (references_dir / "output-schema.md").write_text(
            (ROOT_DIR / "skills" / "minor-detection" / "references" / "output-schema.md").read_text(encoding="utf-8"),
            encoding="utf-8",
        )

        dataset_path = root / "dataset.jsonl"
        dataset_path.write_text(
            json.dumps(
                {
                    "sample_id": "sample_001",
                    "is_minor": True,
                    "source": "social",
                    "conversation": [{"role": "user", "content": "2025-11-20 23:20 ??????????"}],
                },
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

        parsed_json = {
            "decision": {
                "is_minor": True,
                "minor_confidence": 0.91,
                "confidence_band": "high",
                "risk_level": "High",
            },
            "user_profile": {"age_range": "13-15?", "education_stage": "??", "identity_markers": []},
            "icbo_features": {
                "intention": "support",
                "cognition": "peer-focused",
                "behavior_style": "emotional",
                "opportunity_time": "2025-11-20 23:20?time_bucket=late_night",
            },
            "evidence": {
                "direct_evidence": ["??"],
                "historical_evidence": [],
                "retrieval_evidence": ["similar cases"],
                "time_evidence": ["late night"],
                "conflicting_signals": [],
            },
            "reasoning_summary": "ok",
            "trend": {"trajectory": [], "trend_summary": "stable"},
            "uncertainty_notes": [],
            "recommended_next_step": "safe_to_continue",
        }
        observability = {
            "summary": {"script_call_count": 3, "failed_script_calls": 0, "stderr_non_empty": False},
            "time_processing": {"attempted": True, "successful": True, "failure_count": 0, "mode": "script"},
            "retrieval": {
                "attempted": True,
                "successful": True,
                "mode": "embedding",
                "used_fallback": False,
                "fallback_reason": None,
                "retrieved_count": 3,
                "failure_count": 0,
                "quoting_failure_detected": False,
                "network_error_detected": False,
                "message": None,
            },
            "issues": [],
            "script_calls": [
                {"script_name": "run_minor_detection_pipeline.py", "status": "completed", "exit_code": 0, "failed": False, "command": "run_minor_detection_pipeline.py"},
                {"script_name": "extract_time_features.py", "status": "completed", "exit_code": 0, "failed": False, "command": "extract_time_features.py"},
                {"script_name": "retrieve_cases.py", "status": "completed", "exit_code": 0, "failed": False, "command": "retrieve_cases.py"},
            ],
        }

        def fake_run(command, **kwargs):
            return subprocess.CompletedProcess(
                args=command,
                returncode=0,
                stdout=json.dumps(parsed_json, ensure_ascii=False),
                stderr=f"{PIPELINE_OBSERVABILITY_PREFIX} {json.dumps(observability, ensure_ascii=False)}",
            )

        runner = DirectSkillRunner(
            config=DirectRunnerConfig(timeout_sec=30, max_samples=1, sample_strategy="stratified", sample_seed=42),
            command_runner=fake_run,
        )
        run_root = runner.run_dataset(
            project_root=root,
            skill_source_dir=skill_dir,
            skill_version="minor-detection-v0.1.0",
            dataset_path=dataset_path,
            workspace_dir=root / "workspace",
        )

        sample_dir = next(path for path in run_root.iterdir() if path.is_dir())
        agent_output = json.loads((sample_dir / "agent_output.json").read_text(encoding="utf-8"))
        direct_observability = json.loads((sample_dir / "observability.json").read_text(encoding="utf-8"))
        manifest = json.loads((run_root / "run_manifest.json").read_text(encoding="utf-8"))

        self.assertTrue(agent_output["json_valid"])
        self.assertEqual(direct_observability["retrieval"]["mode"], "embedding")
        self.assertEqual(direct_observability["time_processing"]["mode"], "script")
        self.assertEqual(manifest["runner_mode"], "direct")


class SkillSchemaConsistencyTests(unittest.TestCase):
    def test_validate_skill_schema_contract_matches_current_minor_detection_skill(self):
        payload = validate_skill_schema_contract(ROOT_DIR / "skills" / "minor-detection")
        self.assertTrue(payload["ok"])
        self.assertIn("top_level_fields", payload["checks"])
        self.assertIn("decision_fields", payload["checks"])
        self.assertTrue(payload["python_enforces_next_step_enum"])
        self.assertEqual(payload["warnings"], [])


class SkillAgentLoopTests(unittest.TestCase):
    def test_skip_comparison_uses_skipped_decision(self):
        comparison = SkillAgentLoop()._skip_comparison("no errors to optimize on current eval slice")
        self.assertEqual(comparison["decision"], "skipped")
        self.assertEqual(comparison["reason"], "no errors to optimize on current eval slice")

    def test_loop_skips_optimizer_when_baseline_invocation_fully_fails(self):
        root = make_test_dir(self)
        config = mock.Mock()
        config.baseline_source_dir = root / "skills" / "minor-detection"
        config.baseline_version = "minor-detection-v0.1.0"
        config.dataset_path = root / "data" / "val.jsonl"
        config.max_rounds = 1
        config.max_errors = None
        config.protected_count = 8
        config.workspace_root = root / "workspace"
        config.packages_root = root / "packages"
        config.refresh_baseline_version = False
        config.runner_config = CodexRunnerConfig()

        optimizer = mock.Mock()
        loop = SkillAgentLoop(config=config, runner=mock.Mock(), optimizer=optimizer)
        workspace = root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        accepted_eval = {
            "report_payload": {
                "sample_count": 2,
                "invocation_success_rate": 0.0,
            }
        }

        with mock.patch("src.skill_loop.loop.ensure_version_snapshot"), mock.patch.object(loop, "_workspace", return_value=workspace), mock.patch.object(loop, "_evaluate_version", return_value=accepted_eval):
            summary = loop.run()

        optimizer.optimize_from_judge_artifacts.assert_not_called()
        self.assertEqual(summary["final_version"], "minor-detection-v0.1.0")
        self.assertFalse(summary["manual_review_required"])
        self.assertIsNone(summary["review_artifact"])
        self.assertEqual(summary["rounds"][0]["comparison"]["decision"], "skipped")
        self.assertIn("baseline", summary["rounds"][0]["comparison"]["reason"])
        self.assertIn("invocation failed", summary["rounds"][0]["comparison"]["reason"])


class SkillLoopJudgeTests(unittest.TestCase):
    def _make_sample_dir(
        self,
        root: Path,
        name: str,
        *,
        is_minor: bool,
        predicted: bool,
        include_time_trace: bool = True,
        include_script_trace: bool = True,
        observability: dict | None = None,
    ) -> Path:
        sample_dir = root / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_input = {
            "mode": "single_session",
            "conversation": [{"role": "user", "content": "今天是 2025-11-20 23:20，我很烦。"}],
            "context": {"raw_time_hint": "2025-11-20 23:20"},
        }
        gold = {"sample_id": name, "is_minor": is_minor}
        parsed_json = {
            "decision": {
                "is_minor": predicted,
                "minor_confidence": 0.82 if predicted else 0.18,
                "confidence_band": "high" if predicted else "low",
                "risk_level": "Low",
            },
            "user_profile": {"age_range": "13-15岁", "education_stage": "初中", "identity_markers": []},
            "icbo_features": {
                "intention": "intent",
                "cognition": "cognition",
                "behavior_style": "style",
                "opportunity_time": "2025-11-20 23:20",
            },
            "evidence": {
                "direct_evidence": [],
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
        transcript = []
        if include_time_trace:
            transcript.append("time_features generated")
        if include_script_trace:
            transcript.append("scripts/extract_time_features.py invoked")
        transcript_text = "\n".join(transcript)

        (sample_dir / "sample_input.json").write_text(json.dumps(sample_input, ensure_ascii=False), encoding="utf-8")
        (sample_dir / "gold.json").write_text(json.dumps(gold, ensure_ascii=False), encoding="utf-8")
        (sample_dir / "agent_output.json").write_text(
            json.dumps({"raw_text": json.dumps(parsed_json), "parsed_json": parsed_json, "json_valid": True, "launcher_success": True}, ensure_ascii=False),
            encoding="utf-8",
        )
        (sample_dir / "transcript.md").write_text(transcript_text, encoding="utf-8")
        (sample_dir / "tool_trace.json").write_text("[]", encoding="utf-8")
        if observability is not None:
            (sample_dir / "observability.json").write_text(json.dumps(observability, ensure_ascii=False), encoding="utf-8")
        (sample_dir / "timing.json").write_text(json.dumps({"total_duration_seconds": 1.2}), encoding="utf-8")
        (sample_dir / "run_metadata.json").write_text(json.dumps({"returncode": 0, "sample_id": name}), encoding="utf-8")
        return sample_dir

    def test_judge_respects_max_errors_cap(self):
        run_root = make_test_dir(self)
        for idx in range(15):
            self._make_sample_dir(run_root, f"sample_{idx}", is_minor=True, predicted=False)
        judged = judge_run_artifacts(
            run_root=run_root,
            skill_version="minor-detection-v0.1.0",
            parent_version=None,
            dataset_name="val",
            max_errors=3,
        )
        lines = judged["error_index_path"].read_text(encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)

    def test_judge_flags_missing_time_and_script_usage(self):
        run_root = make_test_dir(self)
        self._make_sample_dir(
            run_root,
            "sample_x",
            is_minor=True,
            predicted=True,
            include_time_trace=False,
            include_script_trace=False,
        )
        judged = judge_run_artifacts(
            run_root=run_root,
            skill_version="minor-detection-v0.1.0",
            parent_version=None,
            dataset_name="val",
            max_errors=4,
        )
        report = json.loads(judged["report_path"].read_text(encoding="utf-8"))
        self.assertGreater(report["failure_type_counts"]["missing_time_handling"], 0)
        self.assertGreater(report["failure_type_counts"]["missing_script_usage"], 0)

    def test_judge_tracks_observed_log_issues_without_turning_them_into_main_errors(self):
        run_root = make_test_dir(self)
        self._make_sample_dir(
            run_root,
            "sample_obs",
            is_minor=True,
            predicted=True,
            observability={
                "summary": {"failed_script_calls": 1},
                "time_processing": {"attempted": True, "successful": True, "failure_count": 0},
                "retrieval": {
                    "attempted": True,
                    "successful": True,
                    "mode": "fallback:ConnectError",
                    "used_fallback": True,
                    "network_error_detected": True,
                    "quoting_failure_detected": True,
                },
                "issues": [
                    "script_command_failure",
                    "shell_quoting_failure",
                    "retrieval_fallback",
                    "retrieval_network_blocked",
                ],
                "script_calls": [{"script_name": "extract_time_features.py"}],
            },
        )
        judged = judge_run_artifacts(
            run_root=run_root,
            skill_version="minor-detection-v0.1.0",
            parent_version=None,
            dataset_name="val",
            max_errors=4,
        )
        report = json.loads(judged["report_path"].read_text(encoding="utf-8"))
        self.assertEqual(report["total_errors"], 0)
        self.assertEqual(report["observed_issue_counts"]["retrieval_fallback"], 1)
        self.assertEqual(report["observability_stats"]["shell_quoting_failure_count"], 1)
        self.assertEqual(report["retrieval_mode_counts"]["fallback:ConnectError"], 1)


    def test_judge_penalizes_invalid_outputs_in_report_metrics(self):
        run_root = make_test_dir(self)
        self._make_sample_dir(run_root, "sample_minor_ok", is_minor=True, predicted=True)

        invalid_dir = run_root / "sample_adult_invalid"
        invalid_dir.mkdir(parents=True, exist_ok=True)
        (invalid_dir / "sample_input.json").write_text(
            json.dumps({"mode": "single_session", "conversation": [{"role": "user", "content": "鎴戝凡缁忓伐浣滀簡"}]}, ensure_ascii=False),
            encoding="utf-8",
        )
        (invalid_dir / "gold.json").write_text(
            json.dumps({"sample_id": "sample_adult_invalid", "is_minor": False}, ensure_ascii=False),
            encoding="utf-8",
        )
        (invalid_dir / "agent_output.json").write_text(
            json.dumps({"raw_text": "", "parsed_json": None, "json_valid": False, "launcher_success": False}, ensure_ascii=False),
            encoding="utf-8",
        )
        (invalid_dir / "transcript.md").write_text("", encoding="utf-8")
        (invalid_dir / "tool_trace.json").write_text("[]", encoding="utf-8")
        (invalid_dir / "timing.json").write_text(json.dumps({"total_duration_seconds": 1.0}), encoding="utf-8")
        (invalid_dir / "run_metadata.json").write_text(
            json.dumps({"returncode": 1, "sample_id": "sample_adult_invalid"}, ensure_ascii=False),
            encoding="utf-8",
        )

        judged = judge_run_artifacts(
            run_root=run_root,
            skill_version="minor-detection-v0.1.0",
            parent_version=None,
            dataset_name="val",
            max_errors=4,
        )
        report = json.loads(judged["report_path"].read_text(encoding="utf-8"))
        self.assertEqual(report["output_parse_failure_stats"]["output_parse_failure"], 1)
        self.assertEqual(report["metrics"]["false_positive"], 1)
        self.assertLess(report["metrics"]["f1_score"], 1.0)


class SkillLoopCompareTests(unittest.TestCase):
    def test_compare_reports_blocks_protected_regression(self):
        root = make_test_dir(self)
        accepted_report = root / "accepted.json"
        candidate_report = root / "candidate.json"
        accepted_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.8},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 1.0,
                    "step_compliance_rate": 1.0,
                }
            ),
            encoding="utf-8",
        )
        candidate_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.9},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 1.0,
                    "step_compliance_rate": 1.0,
                }
            ),
            encoding="utf-8",
        )
        protected_index = root / "protected.jsonl"
        protected_index.write_text(json.dumps({"sample_id": "s1"}) + "\n", encoding="utf-8")
        candidate_error_index = root / "errors.jsonl"
        candidate_error_index.write_text(json.dumps({"sample_id": "s1"}) + "\n", encoding="utf-8")
        result = compare_reports(
            accepted_report_path=accepted_report,
            candidate_report_path=candidate_report,
            accepted_protected_index_path=protected_index,
            candidate_error_index_path=candidate_error_index,
        )
        self.assertEqual(result["decision"], "rollback")

    def test_compare_reports_allows_invocation_improvement_without_f1_gain(self):
        root = make_test_dir(self)
        accepted_report = root / "accepted.json"
        candidate_report = root / "candidate.json"
        accepted_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.8},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 0.6,
                    "step_compliance_rate": 0.7,
                }
            ),
            encoding="utf-8",
        )
        candidate_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.8},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 0.85,
                    "step_compliance_rate": 0.7,
                }
            ),
            encoding="utf-8",
        )
        protected_index = root / "protected.jsonl"
        protected_index.write_text("", encoding="utf-8")
        candidate_error_index = root / "errors.jsonl"
        candidate_error_index.write_text("", encoding="utf-8")

        result = compare_reports(
            accepted_report_path=accepted_report,
            candidate_report_path=candidate_report,
            accepted_protected_index_path=protected_index,
            candidate_error_index_path=candidate_error_index,
        )

        self.assertEqual(result["decision"], "promote")
        self.assertFalse(result["gates"]["f1_improved"])
        self.assertTrue(result["gates"]["invocation_improved"])
        self.assertTrue(result["gates"]["core_metric_improved"])

    def test_compare_reports_blocks_step_compliance_regression_even_when_f1_improves(self):
        root = make_test_dir(self)
        accepted_report = root / "accepted.json"
        candidate_report = root / "candidate.json"
        accepted_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.8},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 0.9,
                    "step_compliance_rate": 0.95,
                }
            ),
            encoding="utf-8",
        )
        candidate_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.88},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 0.9,
                    "step_compliance_rate": 0.8,
                }
            ),
            encoding="utf-8",
        )
        protected_index = root / "protected.jsonl"
        protected_index.write_text("", encoding="utf-8")
        candidate_error_index = root / "errors.jsonl"
        candidate_error_index.write_text("", encoding="utf-8")

        result = compare_reports(
            accepted_report_path=accepted_report,
            candidate_report_path=candidate_report,
            accepted_protected_index_path=protected_index,
            candidate_error_index_path=candidate_error_index,
        )

        self.assertEqual(result["decision"], "rollback")
        self.assertTrue(result["gates"]["f1_improved"])
        self.assertFalse(result["gates"]["step_compliance_non_regression"])

    def test_compare_reports_blocks_when_no_core_metric_improves(self):
        root = make_test_dir(self)
        accepted_report = root / "accepted.json"
        candidate_report = root / "candidate.json"
        accepted_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.8},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 0.9,
                    "step_compliance_rate": 0.95,
                }
            ),
            encoding="utf-8",
        )
        candidate_report.write_text(
            json.dumps(
                {
                    "metrics": {"f1_score": 0.8},
                    "schema_validity_rate": 1.0,
                    "invocation_success_rate": 0.9,
                    "step_compliance_rate": 0.95,
                }
            ),
            encoding="utf-8",
        )
        protected_index = root / "protected.jsonl"
        protected_index.write_text("", encoding="utf-8")
        candidate_error_index = root / "errors.jsonl"
        candidate_error_index.write_text("", encoding="utf-8")

        result = compare_reports(
            accepted_report_path=accepted_report,
            candidate_report_path=candidate_report,
            accepted_protected_index_path=protected_index,
            candidate_error_index_path=candidate_error_index,
        )

        self.assertEqual(result["decision"], "rollback")
        self.assertFalse(result["gates"]["core_metric_improved"])


class PacketOptimizerTests(unittest.TestCase):
    def test_packet_routing_limits_editable_targets(self):
        optimizer = SkillOptimizer(llm_client=DummyLLMClient())
        report_payload = {
            "failure_type_counts": {
                "false_positive": 2,
                "missing_time_handling": 1,
            }
        }
        targets = optimizer.resolve_packet_edit_targets(report_payload)
        self.assertEqual(
            targets,
            [
                "SKILL.md",
                "references/evidence-rules.md",
            ],
        )

    def test_packet_routing_prefers_retrieval_template_over_config(self):
        optimizer = SkillOptimizer(llm_client=DummyLLMClient())
        report_payload = {
            "observed_issue_counts": {
                "retrieval_fallback": 1,
                "retrieval_network_blocked": 1,
            }
        }
        targets = optimizer.resolve_packet_edit_targets(report_payload)
        self.assertEqual(targets, ["references/retrieval-query-template.md"])

    def test_optimizer_strips_prose_around_markdown_block(self):
        optimizer = SkillOptimizer(llm_client=DummyLLMClient())
        cleaned = optimizer._strip_markdown_fence(
            "Here is the optimized file.\n\n```markdown\n---\nname: minor-detection\ndescription: test\n---\n\n# Minor Detection\n```\n"
        )
        self.assertTrue(cleaned.startswith("---"))
        self.assertIn("# Minor Detection", cleaned)

    def test_description_substantive_change_rejects_quote_only_edits(self):
        optimizer = SkillOptimizer(llm_client=DummyLLMClient())
        base_skill = (
            "---\n"
            "name: minor-detection\n"
            "description: 即使用户没有直接说“未成年人识别”，但需求本质上是判断“像不像未成年用户”时也应激活此技能。\n"
            "---\n\n"
            "# Minor Detection\n"
        )
        candidate_skill = (
            "---\n"
            "name: minor-detection\n"
            "description: 即使用户没有直接说\"未成年人识别\"，但需求本质上是判断\"像不像未成年用户\"时也应激活此技能。\n"
            "---\n\n"
            "# Minor Detection\n"
        )

        self.assertTrue(optimizer._is_description_only_skill_change(base_skill, candidate_skill))
        self.assertFalse(optimizer._has_substantive_description_change(base_skill, candidate_skill))
        self.assertEqual(
            optimizer._analyze_description_change(base_skill, candidate_skill)["trivial_change_reason"],
            "punctuation_or_format_only",
        )

    def test_trigger_eval_packet_examples_use_trigger_labels_and_decision_confidence(self):
        root = make_test_dir(self)
        packets_dir = root / "failure_packets"
        packet_dir = packets_dir / "failure_001"
        packet_dir.mkdir(parents=True, exist_ok=True)
        (packet_dir / "sample_input.json").write_text(
            json.dumps({"query": "帮我判断要不要触发这个 skill"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (packet_dir / "gold.json").write_text(
            json.dumps(
                {
                    "sample_id": "trigger-001",
                    "should_trigger": True,
                    "expected_is_minor": False,
                    "slice": "identity_explicit",
                    "scenario": "window_scan",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (packet_dir / "agent_output.json").write_text(
            json.dumps(
                {
                    "parsed_json": {
                        "should_trigger": False,
                        "skill_invoked": False,
                        "decision_confidence": 0.73,
                        "decision_reason": "no trigger",
                        "invocation_status": "not_invoked",
                    },
                    "json_valid": True,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (packet_dir / "judge_findings.json").write_text(
            json.dumps(
                {
                    "sample_id": "trigger-001",
                    "failure_types": ["false_negative"],
                    "missing_fields": [],
                    "slice": "identity_explicit",
                    "scenario": "window_scan",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        (packet_dir / "artifact_summary.json").write_text(
            json.dumps(
                {
                    "sample_id": "trigger-001",
                    "slice": "identity_explicit",
                    "scenario": "window_scan",
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        optimizer = SkillOptimizer(llm_client=DummyLLMClient())
        examples = optimizer._load_packet_examples(packets_dir, task_type="trigger_eval")

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0]["ground_truth"], "trigger")
        self.assertEqual(examples[0]["predicted"], "no_trigger")
        self.assertEqual(examples[0]["label"], "trigger")
        self.assertAlmostEqual(examples[0]["confidence"], 0.73)

    def test_trigger_eval_optimizer_skips_non_substantive_description_candidate(self):
        root = make_test_dir(self)
        skills_root = root / "skills"
        current_dir = skills_root / "minor-detection-v0.1.0"
        current_dir.mkdir(parents=True, exist_ok=True)
        skill_text = (
            "---\n"
            "name: minor-detection\n"
            "description: 即使用户没有直接说“未成年人识别”，但需求本质上是判断“像不像未成年用户”时也应激活此技能。\n"
            "---\n\n"
            "# Minor Detection\n"
        )
        (current_dir / "SKILL.md").write_text(skill_text, encoding="utf-8")
        report_path = root / "report.json"
        report_path.write_text(
            json.dumps(
                {
                    "task_type": "trigger_eval",
                    "optimization_focus": "description",
                    "total_errors": 1,
                    "max_errors": 1,
                    "failure_type_counts": {"false_positive": 1},
                    "metrics": {
                        "accuracy": 0.8,
                        "precision": 0.7,
                        "recall": 1.0,
                        "f1_score": 0.82,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        failure_packets_dir = root / "failure_packets"
        protected_packets_dir = root / "protected_packets"
        failure_packets_dir.mkdir(parents=True, exist_ok=True)
        protected_packets_dir.mkdir(parents=True, exist_ok=True)

        optimizer = SkillOptimizer(skills_dir=str(skills_root), llm_client=DummyLLMClient())
        revised_skill = (
            "---\n"
            "name: minor-detection\n"
            "description: 即使用户没有直接说\"未成年人识别\"，但需求本质上是判断\"像不像未成年用户\"时也应激活此技能。\n"
            "---\n\n"
            "# Minor Detection\n"
        )
        failure_examples = [
            {
                "packet_id": "failure_001",
                "sample_id": "trigger-001",
                "ground_truth": "adult",
                "predicted": "unknown",
                "label": "adult",
                "confidence": 0.9,
                "failure_types": ["false_positive"],
                "reasoning": "fake reasoning",
                "missing_fields": [],
            }
        ]

        with (
            mock.patch.object(optimizer, "_load_packet_examples", side_effect=[failure_examples, []]),
            mock.patch.object(optimizer, "_load_reference_materials", return_value={}),
            mock.patch.object(optimizer, "_request_revised_markdown", return_value=revised_skill),
        ):
            result = optimizer.optimize_from_judge_artifacts(
                report_path=report_path,
                failure_packets_dir=failure_packets_dir,
                protected_packets_dir=protected_packets_dir,
                current_version="minor-detection-v0.1.0",
                new_version="minor-detection-v0.1.1-rc001-20260330_190000",
                dry_run=False,
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["message"], "description revision is not substantive")
        self.assertEqual(
            result["description_change"]["trivial_change_reason"],
            "punctuation_or_format_only",
        )
        self.assertFalse((skills_root / "minor-detection-v0.1.1-rc001-20260330_190000").exists())

    def test_trigger_eval_packet_prompt_includes_rewrite_contract_and_slice_contrast(self):
        optimizer = SkillOptimizer(llm_client=DummyLLMClient())
        prompt = optimizer._build_packet_prompt_sections(
            editable_target="SKILL.md",
            report_payload={"task_type": "trigger_eval", "failure_type_counts": {"false_positive": 2}},
            failure_examples=[
                {
                    "packet_id": "failure_001",
                    "sample_id": "trigger-topic_adjacent_not_identity-102",
                    "ground_truth": "adult",
                    "predicted": "unknown",
                    "label": "adult",
                    "confidence": 0.88,
                    "slice": "topic_adjacent_not_identity",
                    "scenario": "window_scan",
                    "failure_types": ["false_positive"],
                    "reasoning": "failure reasoning",
                }
            ],
            protected_examples=[
                {
                    "packet_id": "protected_001",
                    "sample_id": "trigger-topic_adjacent_not_identity-104",
                    "label": "adult",
                    "confidence": 0.9,
                    "slice": "topic_adjacent_not_identity",
                    "scenario": "window_scan",
                    "reasoning": "protected reasoning",
                }
            ],
        )

        self.assertIn("## Trigger Description Rewrite Contract", prompt)
        self.assertIn("Include at least 2 explicit non-trigger boundaries", prompt)
        self.assertIn("Current failure slices to address: topic_adjacent_not_identity", prompt)
        self.assertIn("Current protected slices to preserve: topic_adjacent_not_identity", prompt)
        self.assertIn("## Trigger Slice Contrast Checks", prompt)
        self.assertIn("Failure sample IDs to fix: trigger-topic_adjacent_not_identity-102", prompt)
        self.assertIn("Protected sample IDs to preserve: trigger-topic_adjacent_not_identity-104", prompt)


