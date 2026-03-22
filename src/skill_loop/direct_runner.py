# 模块说明：
# - Mode B 的数据集执行器，直接调用 bundled skill pipeline。
# - 产物结构和 agent runner 基本对齐。

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.config import ROOT_DIR
from src.runtime import build_formal_single_session_payload
from src.utils.path_utils import to_relative_posix_path

from .runner import (
    PIPELINE_OBSERVABILITY_PREFIX,
    PIPELINE_SCRIPT_NAME,
    CodexSkillRunner,
    _build_timing_summary,
    _extract_raw_time_hint,
    _json_dump,
    _parse_json_payload,
    _run_dirname,
    _safe_slug,
    _strip_observability_marker_lines,
)
from .schema_consistency import validate_skill_schema_contract


@dataclass
class DirectRunnerConfig:
    timeout_sec: int = 600
    max_samples: Optional[int] = None
    sample_strategy: str = "stratified"
    sample_seed: int = 42


def _display_path(value: Any) -> str:
    try:
        path = Path(value)
    except TypeError:
        return str(value)
    return to_relative_posix_path(path, ROOT_DIR) if path.is_absolute() else str(value).replace("\\", "/")


def _log(message: str) -> None:
    print(f"[direct-runner] {message}", file=sys.stderr, flush=True)


class DirectSkillRunner:
    runner_mode = "direct"
    runner_label = "direct runtime"

    def __init__(
        self,
        *,
        config: Optional[DirectRunnerConfig] = None,
        command_runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
    ):
        self.config = config or DirectRunnerConfig()
        self.command_runner = command_runner or subprocess.run

    def _build_analysis_payload(self, sample: Dict[str, Any]):
        raw_time_hint = _extract_raw_time_hint(sample)
        context: Dict[str, Any] = {}
        if raw_time_hint:
            context["raw_time_hint"] = raw_time_hint
        return build_formal_single_session_payload(
            sample.get("conversation", []) or [],
            user_id=str(sample.get("sample_id", "")),
            request_id=str(sample.get("sample_id", "")),
            source="direct_skill_runner",
            context=context or None,
        )

    @staticmethod
    def _default_observability() -> Dict[str, Any]:
        return {
            "fatal_agent_error": None,
            "summary": {"script_call_count": 0, "failed_script_calls": 0, "stderr_non_empty": False},
            "time_processing": {"attempted": False, "successful": False, "failure_count": 0, "mode": None},
            "retrieval": {
                "attempted": False,
                "successful": False,
                "mode": None,
                "used_fallback": False,
                "fallback_reason": None,
                "retrieved_count": 0,
                "failure_count": 0,
                "quoting_failure_detected": False,
                "network_error_detected": False,
                "message": None,
            },
            "issues": [],
            "script_calls": [],
            "launcher": {},
        }

    @staticmethod
    def _extract_pipeline_observability(stderr_text: str) -> Optional[Dict[str, Any]]:
        prefix = f"{PIPELINE_OBSERVABILITY_PREFIX} "
        for line in reversed((stderr_text or "").splitlines()):
            if not line.startswith(prefix):
                continue
            try:
                payload = json.loads(line[len(prefix):].strip())
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
        return None

    @staticmethod
    def _clean_stderr(stderr_text: str) -> str:
        prefix = f"{PIPELINE_OBSERVABILITY_PREFIX} "
        lines = [line for line in (stderr_text or "").splitlines() if not line.startswith(prefix)]
        return "\n".join(lines).strip()

    def _write_tool_trace(self, observability: Dict[str, Any], target_path: Path) -> List[Dict[str, Any]]:
        tool_events: List[Dict[str, Any]] = []
        for item in observability.get("script_calls") or []:
            if not isinstance(item, dict):
                continue
            script_name = str(item.get("script_name", "") or "").strip()
            if not script_name:
                continue
            entry = {
                "script_name": script_name,
                "status": str(item.get("status", "") or ""),
                "exit_code": item.get("exit_code"),
                "failed": bool(item.get("failed", False)),
                "command": str(item.get("command", "") or ""),
            }
            output_json = item.get("output_json")
            if isinstance(output_json, dict):
                entry["output_json"] = output_json
            tool_events.append(entry)
        target_path.write_text(_json_dump(tool_events), encoding="utf-8")
        return tool_events

    def _write_agent_output(self, stdout_text: str, target_path: Path, *, success: bool) -> Dict[str, Any]:
        raw_text = str(stdout_text or "").strip()
        candidate_text = _strip_observability_marker_lines(raw_text)
        parsed_json = _parse_json_payload(candidate_text)
        payload = {
            "raw_text": raw_text,
            "parsed_json": parsed_json,
            "json_valid": bool(success and parsed_json is not None),
            "launcher_success": bool(success),
            "launcher_status": "ok" if success else "process_error",
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
        del project_root
        workspace_dir.mkdir(parents=True, exist_ok=True)
        run_root = workspace_dir / _run_dirname(skill_version)
        run_root.mkdir(parents=True, exist_ok=True)

        pipeline_script_path = skill_source_dir / "scripts" / PIPELINE_SCRIPT_NAME
        if not pipeline_script_path.exists():
            raise FileNotFoundError(f"Direct pipeline script does not exist: {pipeline_script_path}")

        schema_consistency = validate_skill_schema_contract(skill_source_dir)
        schema_consistency_path = run_root / "schema_consistency.json"
        schema_consistency_path.write_text(_json_dump(schema_consistency), encoding="utf-8")
        _log(f"schema_consistency ok={schema_consistency.get('ok')} warnings={len(schema_consistency.get('warnings') or [])}")
        if not schema_consistency.get('ok'):
            raise ValueError(f"skill schema contract drift detected: {schema_consistency_path}")

        with dataset_path.open("r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        _log(f"dataset={_display_path(dataset_path)} total_samples={len(samples)}")
        samples, sampling_info = CodexSkillRunner._select_samples(
            samples,
            max_samples=self.config.max_samples,
            strategy=self.config.sample_strategy,
            sample_seed=self.config.sample_seed,
        )

        _log(f"selected_samples={len(samples)} strategy={sampling_info.get('strategy')}")
        run_started_at = time.time()
        manifest: Dict[str, Any] = {
            "runner_mode": self.runner_mode,
            "skill_version": skill_version,
            "aborted_early": False,
            "fatal_agent_error": None,
            "skill_source_dir": to_relative_posix_path(skill_source_dir, run_root),
            "dataset_path": to_relative_posix_path(dataset_path, run_root),
            "run_root": ".",
            "installed_skill_dir": to_relative_posix_path(skill_source_dir, run_root),
            "install_mode": "version_snapshot_direct",
            "execution_mode": "direct",
            "sandbox_mode": None,
            "codex_model": None,
            "sampling": sampling_info,
            "schema_consistency_path": to_relative_posix_path(schema_consistency_path, run_root),
            "schema_consistency_ok": bool(schema_consistency.get('ok')),
            "schema_consistency_warnings": schema_consistency.get("warnings", []),
            "samples": [],
        }

        for index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", f"sample_{index:05d}"))
            _log(f"sample {index + 1}/{len(samples)} start {sample_id}")
            sample_dir = run_root / f"eval-{index:04d}-{_safe_slug(sample_id)}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            payload = self._build_analysis_payload(sample)
            sample_input_path = sample_dir / "sample_input.json"
            payload_path = sample_dir / "payload.json"
            payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
            sample_input_path.write_text(_json_dump(payload_dict), encoding="utf-8")
            payload_path.write_text(_json_dump(payload_dict), encoding="utf-8")

            gold_payload = {
                "sample_id": sample_id,
                "is_minor": bool(sample.get("is_minor", False)),
                "source": sample.get("source", "unknown"),
            }
            (sample_dir / "gold.json").write_text(_json_dump(gold_payload), encoding="utf-8")

            stdout_path = sample_dir / "transcript.jsonl"
            stderr_path = sample_dir / "stderr.log"
            transcript_md_path = sample_dir / "transcript.md"
            tool_trace_path = sample_dir / "tool_trace.json"
            observability_path = sample_dir / "observability.json"
            timing_path = sample_dir / "timing.json"
            metadata_path = sample_dir / "run_metadata.json"
            agent_output_path = sample_dir / "agent_output.json"

            command = [sys.executable, str(pipeline_script_path), "--payload-file", str(payload_path)]
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")

            started_at = time.time()
            timed_out = False
            try:
                completed = self.command_runner(
                    command,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    check=False,
                    timeout=self.config.timeout_sec,
                    env=env,
                    cwd=str(sample_dir),
                )
                returncode = completed.returncode
                stdout_text = completed.stdout or ""
                stderr_text = completed.stderr or ""
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                returncode = 124
                stdout_text = exc.stdout or ""
                stderr_text = exc.stderr or f"direct pipeline timeout after {self.config.timeout_sec} seconds"
            elapsed = time.time() - started_at

            pipeline_observability = self._extract_pipeline_observability(stderr_text) or {}
            clean_stderr = self._clean_stderr(stderr_text)
            stdout_path.write_text(stdout_text, encoding="utf-8")
            stderr_path.write_text(clean_stderr, encoding="utf-8")
            transcript_md_path.write_text(stdout_text, encoding="utf-8")

            agent_output = self._write_agent_output(
                stdout_text,
                agent_output_path,
                success=(returncode == 0 and not timed_out),
            )
            observability = self._default_observability()
            observability.update({key: value for key, value in pipeline_observability.items() if key in observability or key.endswith("_runtime")})
            observability["summary"] = dict(observability.get("summary") or {})
            observability["summary"]["script_call_count"] = len(observability.get("script_calls") or [])
            observability["summary"]["failed_script_calls"] = sum(1 for item in (observability.get("script_calls") or []) if isinstance(item, dict) and item.get("failed"))
            observability["summary"]["stderr_non_empty"] = bool(clean_stderr)
            issues = set(str(item) for item in (observability.get("issues") or []) if str(item).strip())
            if clean_stderr:
                issues.add("stderr_non_empty")
            if timed_out or returncode != 0:
                issues.add("script_command_failure")
            observability["issues"] = sorted(issues)
            observability["launcher"] = {
                "success": agent_output["json_valid"],
                "status": "timeout" if timed_out else ("ok" if returncode == 0 and agent_output["json_valid"] else "process_error"),
                "pipeline_returncode": returncode,
                "timed_out": timed_out,
                "stdout_json_valid": agent_output["parsed_json"] is not None,
                "stdout_excerpt": _strip_observability_marker_lines(stdout_text)[:2000],
                "stderr_excerpt": clean_stderr[:2000],
            }
            observability_path.write_text(_json_dump(observability), encoding="utf-8")
            tool_trace = self._write_tool_trace(observability, tool_trace_path)

            timing_payload = {
                "duration_ms": int(elapsed * 1000),
                "total_duration_seconds": round(elapsed, 3),
            }
            timing_path.write_text(_json_dump(timing_payload), encoding="utf-8")

            metadata_payload = {
                "runner_mode": self.runner_mode,
                "sample_id": sample_id,
                "fatal_agent_error": None,
                "skill_version": skill_version,
                "skill_source_dir": to_relative_posix_path(skill_source_dir, sample_dir),
                "installed_skill_dir": to_relative_posix_path(skill_source_dir, sample_dir),
                "install_mode": "version_snapshot_direct",
                "command": command,
                "execution_mode": "direct",
                "sandbox_mode": None,
                "codex_model": None,
                "returncode": returncode,
                "timed_out": timed_out,
                "stdout_path": to_relative_posix_path(stdout_path, sample_dir),
                "stderr_path": to_relative_posix_path(stderr_path, sample_dir),
                "tool_trace_path": to_relative_posix_path(tool_trace_path, sample_dir),
                "observability_path": to_relative_posix_path(observability_path, sample_dir),
                "agent_output_path": to_relative_posix_path(agent_output_path, sample_dir),
                "timing_path": to_relative_posix_path(timing_path, sample_dir),
                "sample_input_path": to_relative_posix_path(sample_input_path, sample_dir),
                "payload_path": to_relative_posix_path(payload_path, sample_dir),
                "launcher_status": observability["launcher"]["status"],
                "launcher_success": bool(observability["launcher"]["success"]),
                "json_valid": agent_output["json_valid"],
            }
            metadata_path.write_text(_json_dump(metadata_payload), encoding="utf-8")

            _log(f"sample {index + 1}/{len(samples)} done returncode={returncode} json_valid={agent_output.get('json_valid')} retrieval_mode={((observability.get('retrieval') or {}).get('mode'))}")
            manifest["samples"].append(
                {
                    "sample_id": sample_id,
                    "sample_dir": to_relative_posix_path(sample_dir, run_root),
                    "returncode": returncode,
                    "json_valid": agent_output["json_valid"],
                    "launcher_success": bool(observability["launcher"]["success"]),
                    "duration_ms": int(elapsed * 1000),
                    "duration_seconds": round(elapsed, 3),
                    "observed_issues": observability.get("issues", []),
                    "fatal_agent_error": None,
                    "retrieval_mode": ((observability.get('retrieval') or {}).get('mode')),
                    "tool_trace_entries": len(tool_trace),
                }
            )

        _log(f"run_root={_display_path(run_root)}")
        manifest["counts"] = {
            "sample_count": len(manifest["samples"]),
            "json_valid_count": sum(1 for item in manifest["samples"] if item.get("json_valid")),
            "launcher_success_count": sum(1 for item in manifest["samples"] if item.get("launcher_success")),
            "fatal_agent_error_count": 0,
        }
        manifest["timing"] = _build_timing_summary(manifest["samples"], total_wall_seconds=time.time() - run_started_at)
        manifest_path = run_root / "run_manifest.json"
        manifest_path.write_text(_json_dump(manifest), encoding="utf-8")
        return run_root
