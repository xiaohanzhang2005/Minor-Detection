from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.runtime import build_formal_single_session_payload
from src.skill_loop.runner import (
    LAUNCHER_OBSERVABILITY_FILENAME,
    LAUNCHER_RESULT_FILENAME,
    LAUNCHER_SCRIPT_NAME,
    PIPELINE_SCRIPT_NAME,
    CodexSkillRunner,
    _detect_fatal_agent_error,
    _extract_raw_time_hint,
    _json_dump,
    _parse_json_payload,
    _safe_slug,
    _strip_observability_marker_lines,
    _truncate_text,
)
from src.skill_loop.schema_consistency import validate_skill_schema_contract
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path


TRIGGER_LAUNCHER_SCRIPT_NAME = LAUNCHER_SCRIPT_NAME
TRIGGER_LAUNCHER_RESULT_FILENAME = LAUNCHER_RESULT_FILENAME
TRIGGER_LAUNCHER_OBSERVABILITY_FILENAME = LAUNCHER_OBSERVABILITY_FILENAME
TRIGGER_SKILL_OUTPUT_FILENAME = "skill_output.json"


def _display_path(value: Any) -> str:
    try:
        path = Path(value)
    except TypeError:
        return str(value)
    if not path.is_absolute():
        return str(value).replace("\\", "/")
    try:
        return to_relative_posix_path(path, Path.cwd())
    except ValueError:
        return path.as_posix()


def _trigger_runner_log(message: str) -> None:
    print(f"[trigger-eval-runner] {message}", file=sys.stderr, flush=True)


@dataclass
class TriggerEvalRunnerConfig:
    codex_cmd: str = "codex"
    timeout_sec: int = 600
    max_samples: Optional[int] = None
    sample_strategy: str = "stratified"
    sample_seed: int = 42
    execution_mode: str = "sandbox"
    sandbox_mode: str = "workspace-write"
    skill_execution_mode: str = "probe"
    codex_model: Optional[str] = None
    agent_backend: str = "codex"
    agent_cmd: Optional[str] = None
    agent_args_template: Optional[str] = None
    agent_model: Optional[str] = None
    actual_codex_home: Optional[Path] = None
    prompt_template: str = (
        "The installed `minor-detection` skill is available in this workspace.\n"
        "Your task is to decide whether this user request should trigger the `minor-detection` skill.\n"
        "The request is already prepared at `{query_file}`. Read that file before deciding.\n"
        "A prepared launcher script is already available locally at `{launcher_file}`.\n"
        "If and only if the request should trigger `minor-detection`, run `python {launcher_file}` exactly once.\n"
        "If the request should not trigger the skill, do not run the launcher.\n"
        "Do not inspect SKILL.md, do not rewrite files, and do not explore directories.\n"
        "Return only one JSON object with these fields: `should_trigger` (bool), `skill_invoked` (bool), `decision_confidence` (0-1), `decision_reason` (string), `invocation_status` (`not_invoked` or `invoked_success` or `invoked_failed`).\n"
        "Make sure `skill_invoked` matches whether you actually ran the launcher.\n"
        "If you invoked the launcher and it succeeded, set `invocation_status` to `invoked_success`.\n"
        "If you did not invoke the launcher, set `invocation_status` to `not_invoked`.\n"
        "Do not include code fences or any extra commentary."
    )


class TriggerEvalCodexRunner(CodexSkillRunner):
    runner_mode = "trigger_agent"
    runner_label = "trigger agent runtime"

    def __init__(self, *, config: Optional[TriggerEvalRunnerConfig] = None, command_runner=None):
        super().__init__(config=config or TriggerEvalRunnerConfig(), command_runner=command_runner)

    @classmethod
    def _resolve_stratum_key(cls, sample: Dict[str, Any]) -> str:
        scenario = str(sample.get("scenario", "unknown") or "unknown")
        should_trigger = "trigger" if bool(sample.get("should_trigger", False)) else "no_trigger"
        slice_name = str(sample.get("slice", "unknown") or "unknown")
        return f"{scenario}::{should_trigger}::{slice_name}"

    def _script_name_from_command(self, command: str) -> Optional[str]:
        lowered = str(command or "").lower()
        if TRIGGER_LAUNCHER_SCRIPT_NAME.lower() in lowered:
            return TRIGGER_LAUNCHER_SCRIPT_NAME
        return super()._script_name_from_command(command)

    def _load_dataset_samples(self, dataset_path: Path) -> List[Dict[str, Any]]:
        if dataset_path.suffix.lower() == ".jsonl":
            with dataset_path.open("r", encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]

        payload = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
            return [dict(item) for item in payload["samples"] if isinstance(item, dict)]
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        raise ValueError(f"Unsupported trigger dataset format: {dataset_path}")

    def _output_schema_path(self, workspace_dir: Path) -> Path:
        schema_path = workspace_dir / "trigger_eval_output_schema.json"
        if schema_path.exists():
            return schema_path
        schema = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "should_trigger",
                "skill_invoked",
                "decision_confidence",
                "decision_reason",
                "invocation_status",
            ],
            "properties": {
                "should_trigger": {"type": "boolean"},
                "skill_invoked": {"type": "boolean"},
                "decision_confidence": {"type": "number"},
                "decision_reason": {"type": "string"},
                "invocation_status": {
                    "type": "string",
                    "enum": ["not_invoked", "invoked_success", "invoked_failed"],
                },
            },
        }
        schema_path.write_text(_json_dump(schema), encoding="utf-8")
        return schema_path

    def _write_query_file(self, sample: Dict[str, Any], target_path: Path) -> None:
        target_path.write_text(str(sample.get("query", "") or "").strip(), encoding="utf-8")

    def _resolve_skill_input_turns(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        turns = sample.get("skill_input_turns") or sample.get("window_turns") or []
        if not isinstance(turns, list):
            return []
        resolved: List[Dict[str, Any]] = []
        for turn in turns:
            if not isinstance(turn, dict):
                continue
            resolved.append(
                {
                    "role": str(turn.get("role", "user") or "user"),
                    "content": str(turn.get("content", "") or ""),
                }
            )
        return resolved

    def _infer_expected_is_minor(self, sample: Dict[str, Any]) -> Optional[bool]:
        if "expected_is_minor" not in sample:
            return None
        return bool(sample.get("expected_is_minor"))

    def _build_skill_payload(self, sample: Dict[str, Any]):
        turns = self._resolve_skill_input_turns(sample)
        if not turns:
            raise ValueError(f"trigger sample {sample.get('id') or '<unknown>'} is missing skill_input_turns")
        raw_time_hint = _extract_raw_time_hint({"conversation": turns})
        context: Dict[str, Any] = {}
        if raw_time_hint:
            context["raw_time_hint"] = raw_time_hint
        return build_formal_single_session_payload(
            turns,
            user_id=str(sample.get("skill_input_base_sample_id") or sample.get("base_sample_id") or sample.get("id") or ""),
            request_id=str(sample.get("id") or sample.get("sample_id") or ""),
            source="trigger_eval_runner",
            context=context or None,
        )

    def _write_prepared_payload(self, payload, target_path: Path) -> None:
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        target_path.write_text(_json_dump(payload_dict), encoding="utf-8")

    def _write_trigger_probe_launcher(self, *, target_path: Path, result_path: Path) -> None:
        launcher = f'''#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

RESULT_PATH = Path(r"{str(result_path.resolve())}")

payload = {{
    "success": True,
    "invoked": True,
    "status": "ok",
    "message": "skill invocation probe executed",
}}
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
'''
        target_path.write_text(launcher, encoding="utf-8")

    def _build_prompt(self, *, query_path: Path, launcher_path: Path) -> str:
        return self.config.prompt_template.format(
            query_file=query_path.name,
            launcher_file=launcher_path.name,
        )

    def _write_skill_output(self, *, events, target_path: Path, launcher_invoked: bool) -> Dict[str, Any]:
        raw_text = ""
        parsed_json: Optional[Dict[str, Any]] = None
        parsed_source: Optional[str] = None
        if launcher_invoked:
            for item in reversed(self._extract_command_execution_items(events)):
                command = str(item.get("command", "") or "")
                if self._script_name_from_command(command) != TRIGGER_LAUNCHER_SCRIPT_NAME:
                    continue
                aggregated_output = _strip_observability_marker_lines(str(item.get("aggregated_output", "") or "").strip())
                parsed_json = _parse_json_payload(aggregated_output)
                if parsed_json is not None:
                    raw_text = aggregated_output
                    parsed_source = "launcher_aggregated_output"
                break
        payload = {
            "raw_text": raw_text,
            "parsed_json": parsed_json,
            "json_valid": parsed_json is not None,
            "json_source": parsed_source,
        }
        target_path.write_text(_json_dump(payload), encoding="utf-8")
        return payload

    def _extract_event_error_messages(self, events: List[Dict[str, Any]]) -> List[str]:
        messages: List[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") == "error":
                message = str(event.get("message", "") or "").strip()
                if message:
                    messages.append(message)
                continue
            if event.get("type") == "turn.failed":
                error_payload = event.get("error") or {}
                if isinstance(error_payload, dict):
                    message = str(error_payload.get("message", "") or "").strip()
                    if message:
                        messages.append(message)
        return messages

    def _build_observability(self, *, events, stderr_text: str, launcher_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        script_calls: List[Dict[str, Any]] = []
        issues: set[str] = set()
        failed_script_calls = 0
        for item in self._extract_command_execution_items(events):
            command = str(item.get("command", "") or "")
            script_name = self._script_name_from_command(command)
            if not script_name:
                continue
            aggregated_output = str(item.get("aggregated_output", "") or "")
            status = str(item.get("status", "") or "")
            exit_code = item.get("exit_code")
            failed = status == "failed" or (isinstance(exit_code, int) and exit_code != 0)
            if failed:
                failed_script_calls += 1
                issues.add("script_command_failure")
            script_calls.append(
                {
                    "script_name": script_name,
                    "status": status,
                    "exit_code": exit_code,
                    "failed": failed,
                    "command": command,
                    "aggregated_output": _truncate_text(aggregated_output, max_chars=1200),
                    "output_json": _parse_json_payload(aggregated_output),
                }
            )

        launcher_payload = launcher_result if isinstance(launcher_result, dict) else {}
        if launcher_payload and not bool(launcher_payload.get("success")):
            issues.add("launcher_failed")
            failed_script_calls += 1

        event_error_messages = self._extract_event_error_messages(list(events))
        if event_error_messages:
            issues.add("agent_event_error")

        stderr_non_empty = bool(str(stderr_text or "").strip())
        if stderr_non_empty:
            issues.add("stderr_non_empty")
        fatal_agent_error = _detect_fatal_agent_error("\n".join([str(stderr_text or ""), *event_error_messages]))
        if fatal_agent_error:
            issues.add(fatal_agent_error)

        return {
            "fatal_agent_error": fatal_agent_error,
            "summary": {
                "script_call_count": len(script_calls),
                "failed_script_calls": failed_script_calls,
                "stderr_non_empty": stderr_non_empty,
            },
            "issues": sorted(issues),
            "script_calls": script_calls,
            "launcher": launcher_payload,
        }

    def _write_agent_output(self, final_output_path: Path, target_path: Path, *, launcher_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        raw_text = final_output_path.read_text(encoding="utf-8").strip() if final_output_path.exists() else ""
        parsed_json = _parse_json_payload(raw_text)
        payload = {
            "raw_text": raw_text,
            "parsed_json": parsed_json,
            "json_valid": parsed_json is not None,
            "launcher_success": bool((launcher_result or {}).get("success")),
            "launcher_status": (launcher_result or {}).get("status"),
            "json_source": "final_output" if parsed_json is not None else None,
        }
        target_path.write_text(_json_dump(payload), encoding="utf-8")
        return payload

    def run_dataset(
        self,
        *,
        project_root: Path,
        skill_source_dir: Path,
        skill_version: str,
        dataset_path: Path,
        workspace_dir: Path,
    ) -> Path:
        workspace_dir.mkdir(parents=True, exist_ok=True)
        run_root = workspace_dir / f"run-trigger-{_safe_slug(skill_version)[:24]}"
        if run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)

        isolated_codex_dir = self._build_isolated_codex_home(run_root)
        installed_skill_dir = self._install_skill(
            project_root=project_root,
            skill_source_dir=skill_source_dir,
            isolated_codex_dir=isolated_codex_dir,
        )
        output_schema_path = self._output_schema_path(run_root)

        schema_consistency = validate_skill_schema_contract(skill_source_dir)
        schema_consistency_path = run_root / "schema_consistency.json"
        schema_consistency_path.write_text(_json_dump(schema_consistency), encoding="utf-8")
        _trigger_runner_log(
            f"schema_consistency ok={schema_consistency.get('ok')} warnings={len(schema_consistency.get('warnings') or [])}"
        )
        if not schema_consistency.get("ok"):
            raise ValueError(f"skill schema contract drift detected: {schema_consistency_path}")

        samples = self._load_dataset_samples(dataset_path)
        _trigger_runner_log(f"dataset={_display_path(dataset_path)} total_samples={len(samples)}")
        samples, sampling_info = self._select_samples(
            samples,
            max_samples=self.config.max_samples,
            strategy=self.config.sample_strategy,
            sample_seed=self.config.sample_seed,
        )
        _trigger_runner_log(
            f"selected_samples={len(samples)} strategy={sampling_info.get('strategy')} execution_mode={self.config.execution_mode}"
        )

        run_started_at = time.time()
        manifest: Dict[str, Any] = {
            "runner_mode": self.runner_mode,
            "skill_version": skill_version,
            "skill_source_dir": to_relative_posix_path(skill_source_dir, run_root),
            "dataset_path": to_relative_posix_path(dataset_path, run_root),
            "run_root": ".",
            "isolated_codex_dir": to_relative_posix_path(isolated_codex_dir, run_root),
            "installed_skill_dir": to_relative_posix_path(installed_skill_dir, run_root),
            "execution_mode": self.config.execution_mode,
            "sandbox_mode": self.config.sandbox_mode if self.config.execution_mode == "sandbox" else None,
            "codex_model": self.config.codex_model,
            "agent_backend": self._agent_backend(),
            "agent_model": str(getattr(self.config, "agent_model", None) or getattr(self.config, "codex_model", None) or "") or None,
            "sampling": sampling_info,
            "schema_consistency_path": to_relative_posix_path(schema_consistency_path, run_root),
            "schema_consistency_ok": bool(schema_consistency.get("ok")),
            "schema_consistency_warnings": schema_consistency.get("warnings", []),
            "samples": [],
        }

        json_valid_count = 0
        launcher_invoked_count = 0
        launcher_success_count = 0

        for index, sample in enumerate(samples):
            sample_id = str(sample.get("id") or sample.get("sample_id") or f"sample_{index:05d}")
            _trigger_runner_log(f"sample {index + 1}/{len(samples)} start {sample_id}")
            sample_dir = run_root / f"eval-{index:04d}-{_safe_slug(sample_id)}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            sample_input_path = sample_dir / "sample_input.json"
            sample_input_path.write_text(_json_dump(sample), encoding="utf-8")
            query_path = sample_dir / "query.txt"
            self._write_query_file(sample, query_path)
            launcher_path = sample_dir / TRIGGER_LAUNCHER_SCRIPT_NAME
            launcher_result_path = sample_dir / TRIGGER_LAUNCHER_RESULT_FILENAME
            payload_path: Optional[Path] = None
            pipeline_observability_path: Optional[Path] = None
            skill_output_path: Optional[Path] = None
            if self.config.skill_execution_mode == "full":
                payload = self._build_skill_payload(sample)
                payload_path = sample_dir / "payload.json"
                self._write_prepared_payload(payload, payload_path)
                pipeline_observability_path = sample_dir / TRIGGER_LAUNCHER_OBSERVABILITY_FILENAME
                skill_output_path = sample_dir / TRIGGER_SKILL_OUTPUT_FILENAME
                expected_is_minor = self._infer_expected_is_minor(sample)
                if expected_is_minor is None:
                    raise ValueError(
                        f"trigger sample {sample_id} is missing expected_is_minor required for full smoke validation"
                    )
                self._write_skill_launcher(
                    target_path=launcher_path,
                    pipeline_script_path=installed_skill_dir / "scripts" / PIPELINE_SCRIPT_NAME,
                    payload_path=payload_path,
                    launcher_result_path=launcher_result_path,
                    pipeline_observability_path=pipeline_observability_path,
                )
            else:
                expected_is_minor = None
                self._write_trigger_probe_launcher(target_path=launcher_path, result_path=launcher_result_path)
            prompt = self._build_prompt(query_path=query_path, launcher_path=launcher_path)
            prompt_file_path = sample_dir / "agent_prompt.txt"
            prompt_file_path.write_text(prompt, encoding="utf-8")

            gold_payload = {
                "sample_id": sample_id,
                "should_trigger": bool(sample.get("should_trigger", False)),
                "slice": str(sample.get("slice", "") or ""),
                "scenario": str(sample.get("scenario", "") or ""),
                "source": str(sample.get("source", "") or ""),
                "skill_input_base_sample_id": str(sample.get("skill_input_base_sample_id") or sample.get("base_sample_id") or ""),
                "expected_is_minor": expected_is_minor,
            }
            (sample_dir / "gold.json").write_text(_json_dump(gold_payload), encoding="utf-8")

            final_output_path = sample_dir / "final_output.txt"
            stdout_path = sample_dir / "transcript.jsonl"
            stderr_path = sample_dir / "stderr.log"
            transcript_md_path = sample_dir / "transcript.md"
            tool_trace_path = sample_dir / "tool_trace.json"
            observability_path = sample_dir / "observability.json"
            timing_path = sample_dir / "timing.json"
            metadata_path = sample_dir / "run_metadata.json"
            agent_output_path = sample_dir / "agent_output.json"

            command = self._build_agent_command(
                workspace_dir=sample_dir,
                output_schema_path=output_schema_path,
                final_output_path=final_output_path,
                installed_skill_dir=installed_skill_dir,
                prompt_file_path=prompt_file_path,
            )
            env = self._build_env(isolated_codex_dir)
            started_at = time.time()
            completed = self.command_runner(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                timeout=self.config.timeout_sec,
                env=env,
            )
            elapsed = time.time() - started_at

            stdout_text = completed.stdout or ""
            stderr_text = completed.stderr or ""
            if not final_output_path.exists() and stdout_text.strip():
                final_output_path.write_text(stdout_text.strip(), encoding="utf-8")
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(stderr_text, encoding="utf-8")
            transcript_md_path.write_text(stdout_text, encoding="utf-8")

            events = self._parse_jsonl_events(stdout_text)
            self._write_tool_trace(events, tool_trace_path)
            launcher_invoked = any(
                self._script_name_from_command(str(item.get("command", "") or "")) == TRIGGER_LAUNCHER_SCRIPT_NAME
                for item in self._extract_command_execution_items(events)
            )
            launcher_result = self._load_json_file(launcher_result_path)
            pipeline_observability = self._load_json_file(pipeline_observability_path) if pipeline_observability_path else None
            if isinstance(launcher_result, dict):
                launcher_result["invoked"] = True
            elif launcher_invoked:
                launcher_result = {
                    "invoked": True,
                    "success": False,
                    "status": "process_error",
                }
            observability = self._build_observability(events=events, stderr_text=stderr_text, launcher_result=launcher_result)
            if isinstance(pipeline_observability, dict) and pipeline_observability:
                observability["pipeline_observability"] = pipeline_observability
            observability_path.write_text(_json_dump(observability), encoding="utf-8")
            agent_output = self._write_agent_output(final_output_path, agent_output_path, launcher_result=launcher_result)

            timing_payload = {
                "duration_ms": int(elapsed * 1000),
                "total_duration_seconds": round(elapsed, 3),
            }
            timing_path.write_text(_json_dump(timing_payload), encoding="utf-8")

            launcher_invoked = bool((launcher_result or {}).get("invoked"))
            launcher_success = bool((launcher_result or {}).get("success"))
            skill_output = (
                self._write_skill_output(events=events, target_path=skill_output_path, launcher_invoked=launcher_invoked)
                if skill_output_path is not None
                else {"json_valid": False}
            )
            if agent_output.get("json_valid"):
                json_valid_count += 1
            if launcher_invoked:
                launcher_invoked_count += 1
            if launcher_success:
                launcher_success_count += 1

            metadata_payload = {
                "sample_id": sample_id,
                "skill_version": skill_version,
                "skill_source_dir": to_relative_posix_path(skill_source_dir, sample_dir),
                "installed_skill_dir": to_relative_posix_path(installed_skill_dir, sample_dir),
                "command": command,
                "execution_mode": self.config.execution_mode,
                "sandbox_mode": self.config.sandbox_mode if self.config.execution_mode == "sandbox" else None,
                "codex_model": self.config.codex_model,
                "returncode": completed.returncode,
                "output_schema_path": to_relative_posix_path(output_schema_path, sample_dir),
                "stdout_path": to_relative_posix_path(stdout_path, sample_dir),
                "stderr_path": to_relative_posix_path(stderr_path, sample_dir),
                "tool_trace_path": to_relative_posix_path(tool_trace_path, sample_dir),
                "observability_path": to_relative_posix_path(observability_path, sample_dir),
                "agent_output_path": to_relative_posix_path(agent_output_path, sample_dir),
                "timing_path": to_relative_posix_path(timing_path, sample_dir),
                "sample_input_path": to_relative_posix_path(sample_input_path, sample_dir),
                "payload_path": to_relative_posix_path(payload_path, sample_dir) if payload_path else None,
                "query_path": to_relative_posix_path(query_path, sample_dir),
                "prompt_file_path": to_relative_posix_path(prompt_file_path, sample_dir),
                "launcher_path": to_relative_posix_path(launcher_path, sample_dir),
                "launcher_result_path": to_relative_posix_path(launcher_result_path, sample_dir),
                "pipeline_observability_path": to_relative_posix_path(pipeline_observability_path, sample_dir) if pipeline_observability_path else None,
                "skill_output_path": to_relative_posix_path(skill_output_path, sample_dir) if skill_output_path else None,
                "launcher_success": launcher_success,
                "launcher_invoked": launcher_invoked,
                "json_valid": agent_output["json_valid"],
                "agent_backend": self._agent_backend(),
                "agent_model": str(getattr(self.config, "agent_model", None) or getattr(self.config, "codex_model", None) or "") or None,
                "skill_execution_mode": self.config.skill_execution_mode,
                "skill_output_json_valid": bool(skill_output.get("json_valid")),
            }
            metadata_path.write_text(_json_dump(metadata_payload), encoding="utf-8")
            _trigger_runner_log(
                "sample "
                f"{index + 1}/{len(samples)} done "
                f"returncode={completed.returncode} "
                f"json_valid={agent_output.get('json_valid')} "
                f"launcher_invoked={launcher_invoked} "
                f"launcher_success={launcher_success}"
            )

            manifest["samples"].append(
                {
                    "sample_id": sample_id,
                    "slice": gold_payload["slice"],
                    "scenario": gold_payload["scenario"],
                    "should_trigger": gold_payload["should_trigger"],
                    "returncode": completed.returncode,
                    "json_valid": bool(agent_output.get("json_valid")),
                    "launcher_invoked": launcher_invoked,
                    "launcher_success": launcher_success,
                    "skill_output_json_valid": bool(skill_output.get("json_valid")),
                    "duration_seconds": round(elapsed, 3),
                }
            )

        total_wall_seconds = time.time() - run_started_at
        manifest["counts"] = {
            "sample_count": len(samples),
            "json_valid_count": json_valid_count,
            "launcher_invoked_count": launcher_invoked_count,
            "launcher_success_count": launcher_success_count,
        }
        manifest["timing"] = {
            "total_wall_seconds": round(total_wall_seconds, 3),
            "avg_sample_seconds": round(total_wall_seconds / len(samples), 3) if samples else 0.0,
        }
        _trigger_runner_log(
            "completed "
            f"samples={len(samples)} "
            f"json_valid={json_valid_count} "
            f"launcher_invoked={launcher_invoked_count} "
            f"launcher_success={launcher_success_count} "
            f"total_wall_seconds={manifest['timing']['total_wall_seconds']}"
        )

        run_manifest_path = run_root / "run_manifest.json"
        run_manifest_path.write_text(
            json.dumps(normalize_project_paths(manifest, project_root=project_root, start=run_root), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return run_root
