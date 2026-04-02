# 模块说明：
# - Mode A 的数据集执行器，让 Agent 只运行一次预制 launcher。
# - 会产出 transcript、tool trace、observability 和 run manifest。

from __future__ import annotations

import json
import math
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from src.config import ROOT_DIR
from src.models import AnalysisPayload, FormalSkillOutput
from src.runtime import build_formal_single_session_payload
from src.utils.path_utils import normalize_project_paths, to_relative_posix_path

from .packaging import install_skill_snapshot
from .schema_consistency import validate_skill_schema_contract


TIMESTAMP_PATTERN = re.compile(r"(\d{4}[-/]\d{1,2}[-/]\d{1,2}[^\n]{0,24}\d{1,2}:\d{2}(?::\d{2})?)")
TIME_SCRIPT_NAME = "extract_time_features.py"
RETRIEVE_SCRIPT_NAME = "retrieve_cases.py"
PIPELINE_SCRIPT_NAME = "run_minor_detection_pipeline.py"
PIPELINE_OBSERVABILITY_PREFIX = "[MINOR_PIPELINE_OBSERVABILITY]"
LAUNCHER_SCRIPT_NAME = "run_skill_once.py"
LAUNCHER_RESULT_FILENAME = "launcher_result.json"
LAUNCHER_OBSERVABILITY_FILENAME = "pipeline_observability.json"
DEFAULT_EMBEDDING_BASE_URL = "https://aihubmix.com/v1"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


def _run_dirname(skill_version: str) -> str:
    return f"run-{_safe_slug(skill_version)[:24]}"


def _display_path(value: Any) -> str:
    try:
        path = Path(value)
    except TypeError:
        return str(value)
    return to_relative_posix_path(path, ROOT_DIR) if path.is_absolute() else str(value).replace("\\", "/")


def _runner_log(message: str) -> None:
    print(f"[agent-runner] {message}", file=sys.stderr, flush=True)


def _extract_raw_time_hint(sample: Dict[str, Any]) -> str:
    icbo_features = sample.get("icbo_features")
    if isinstance(icbo_features, dict):
        opportunity_time = str(icbo_features.get("opportunity_time", "") or "").strip()
        if opportunity_time:
            return opportunity_time
    for turn in sample.get("conversation", []) or []:
        content = str(turn.get("content", "") or "")
        match = TIMESTAMP_PATTERN.search(content)
        if match:
            return match.group(1).strip()
    return ""


def _json_dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2)


def _strict_json_schema(node: Any) -> Any:
    if isinstance(node, dict):
        normalized = {key: _strict_json_schema(value) for key, value in node.items()}
        for metadata_key in ("title", "description", "default", "examples"):
            normalized.pop(metadata_key, None)
        if "$ref" in normalized:
            return {"$ref": normalized["$ref"]}
        if normalized.get("type") == "object":
            normalized["additionalProperties"] = False
            properties = normalized.get("properties")
            if isinstance(properties, dict):
                normalized["required"] = list(properties.keys())
        return normalized
    if isinstance(node, list):
        return [_strict_json_schema(item) for item in node]
    return node


def _safe_slug(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_-]+", "-", text.strip()).strip("-")
    return normalized or "sample"


def _truncate_text(text: str, *, max_chars: int = 800) -> str:
    value = str(text or "")
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 14] + "\n...[truncated]"


def _build_timing_summary(samples: List[Dict[str, Any]], *, total_wall_seconds: float) -> Dict[str, Any]:
    durations = [float(item.get("duration_seconds", 0.0) or 0.0) for item in samples if isinstance(item, dict)]
    if durations:
        ordered = sorted(durations)
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 == 1 else (ordered[middle - 1] + ordered[middle]) / 2.0
        min_seconds = ordered[0]
        max_seconds = ordered[-1]
        avg_seconds = sum(ordered) / len(ordered)
        total_sample_seconds = sum(ordered)
    else:
        median = 0.0
        min_seconds = 0.0
        max_seconds = 0.0
        avg_seconds = 0.0
        total_sample_seconds = 0.0
    return {
        "sample_count": len(samples),
        "timed_sample_count": len(durations),
        "total_wall_seconds": round(float(total_wall_seconds or 0.0), 3),
        "sample_total_seconds": round(total_sample_seconds, 3),
        "avg_sample_seconds": round(avg_seconds, 3),
        "median_sample_seconds": round(median, 3),
        "min_sample_seconds": round(min_seconds, 3),
        "max_sample_seconds": round(max_seconds, 3),
    }


def _parse_json_payload(text: str) -> Optional[Dict[str, Any]]:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _strip_observability_marker_lines(text: str) -> str:
    prefix = f"{PIPELINE_OBSERVABILITY_PREFIX} "
    raw_text = str(text or "")
    marker_index = raw_text.find(prefix)
    if marker_index >= 0:
        raw_text = raw_text[:marker_index]
    lines = [line for line in raw_text.splitlines() if not line.startswith(prefix)]
    return "\n".join(lines).strip()


def _detect_fatal_agent_error(stderr_text: str) -> Optional[str]:
    lowered = str(stderr_text or "").lower()
    if "failed to load skill" in lowered and "missing yaml frontmatter" in lowered:
        return "skill_load_failure"
    if "unexpected status 503" in lowered and "service unavailable" in lowered:
        return "agent_model_unavailable"
    if "reconnecting..." in lowered and "/responses" in lowered:
        return "agent_model_unavailable"
    if any(token in lowered for token in ("unauthorized", "authentication", "forbidden", "invalid api key")):
        return "agent_auth_failure"
    return None


@dataclass
class CodexRunnerConfig:
    codex_cmd: str = "codex"
    timeout_sec: int = 600
    max_samples: Optional[int] = None
    sample_strategy: str = "stratified"
    sample_seed: int = 42
    execution_mode: str = "sandbox"
    sandbox_mode: str = "workspace-write"
    codex_model: Optional[str] = None
    agent_backend: str = "codex"
    agent_cmd: Optional[str] = None
    agent_args_template: Optional[str] = None
    agent_model: Optional[str] = None
    actual_codex_home: Optional[Path] = None
    prompt_template: str = (
        "The installed `minor-detection` skill has already been wired into the prepared launcher.\n"
        "The exact payload JSON is already prepared at `{payload_file}`.\n"
        "A one-shot launcher script is already prepared at `{launcher_file}`.\n"
        "Do not create, rewrite, or transcode payload files.\n"
        "Do not inspect SKILL.md, do not inspect directories, and do not rewrite the pipeline implementation.\n"
        "Do not change directories and do not run any command other than the prepared launcher.\n"
        "Run `python {launcher_file}` exactly once from the current working directory.\n"
        "If it succeeds, return the stdout JSON object verbatim and nothing else.\n"
        "Do not include observability markers, logs, code fences, or any extra commentary.\n"
        "If it fails, return the error only and stop.\n"
    )


class CodexSkillRunner:
    def __init__(
        self,
        *,
        config: Optional[CodexRunnerConfig] = None,
        command_runner: Optional[Callable[..., subprocess.CompletedProcess[str]]] = None,
    ):
        self.config = config or CodexRunnerConfig()
        self.command_runner = command_runner or subprocess.run

    def _build_isolated_codex_home(self, run_root: Path) -> Path:
        actual_home = self.config.actual_codex_home or (Path.home() / ".codex")
        isolated_home = run_root / "codex_home"
        isolated_codex_dir = isolated_home / ".codex"
        isolated_codex_dir.mkdir(parents=True, exist_ok=True)

        for file_name in ("auth.json", "config.toml", "version.json"):
            source = actual_home / file_name
            if source.exists():
                shutil.copy2(source, isolated_codex_dir / file_name)

        actual_skills_dir = actual_home / "skills"
        if actual_skills_dir.exists() and (actual_skills_dir / ".system").exists():
            target_system_dir = isolated_codex_dir / "skills" / ".system"
            target_system_dir.parent.mkdir(parents=True, exist_ok=True)
            if not target_system_dir.exists():
                shutil.copytree(actual_skills_dir / ".system", target_system_dir)

        return isolated_codex_dir

    def _install_skill(self, *, project_root: Path, skill_source_dir: Path, isolated_codex_dir: Path) -> Path:
        skills_root = isolated_codex_dir / "skills"
        return install_skill_snapshot(project_root=project_root, skill_dir=skill_source_dir, target_dir=skills_root)

    def _build_analysis_payload(self, sample: Dict[str, Any]) -> AnalysisPayload:
        raw_time_hint = _extract_raw_time_hint(sample)
        context: Dict[str, Any] = {}
        if raw_time_hint:
            context["raw_time_hint"] = raw_time_hint
        return build_formal_single_session_payload(
            sample.get("conversation", []) or [],
            user_id=str(sample.get("sample_id", "")),
            request_id=str(sample.get("sample_id", "")),
            source="codex_skill_runner",
            context=context or None,
        )

    @staticmethod
    def _resolve_age_bucket(sample: Dict[str, Any]) -> str:
        age_bucket = sample.get("age_bucket")
        if age_bucket:
            return str(age_bucket)

        age = sample.get("age")
        is_minor = bool(sample.get("is_minor", False))
        if isinstance(age, (int, float)):
            age = int(age)
            if is_minor:
                if age <= 12:
                    return "minor_07_12"
                if age <= 15:
                    return "minor_13_15"
                return "minor_16_17"
            if age <= 22:
                return "adult_18_22"
            if age <= 26:
                return "adult_23_26"
            return "adult_27_plus"

        return "minor_unknown" if is_minor else "adult_unknown"

    @classmethod
    def _resolve_stratum_key(cls, sample: Dict[str, Any]) -> str:
        source = str(sample.get("source", "unknown"))
        age_bucket = cls._resolve_age_bucket(sample)
        return f"{source}::{age_bucket}"

    @staticmethod
    def _allocate_stratified_counts(group_sizes: Dict[str, int], target_total: int) -> Dict[str, int]:
        if target_total <= 0:
            return {key: 0 for key in group_sizes}

        keys = sorted(group_sizes.keys())
        allocations = {key: 0 for key in keys}
        if not keys:
            return allocations

        if target_total >= len(keys):
            for key in keys:
                allocations[key] = 1
            target_total -= len(keys)

        if target_total <= 0:
            return allocations

        remaining_capacity = {key: group_sizes[key] - allocations[key] for key in keys}
        total_capacity = sum(max(0, value) for value in remaining_capacity.values())
        if total_capacity <= 0:
            return allocations

        base_additions: Dict[str, int] = {}
        remainders: List[tuple[float, int, str]] = []
        assigned = 0
        for key in keys:
            capacity = max(0, remaining_capacity[key])
            quota = (capacity / total_capacity) * target_total if total_capacity else 0.0
            floor_value = min(capacity, math.floor(quota))
            base_additions[key] = floor_value
            assigned += floor_value
            remainders.append((quota - floor_value, capacity - floor_value, key))

        for key, floor_value in base_additions.items():
            allocations[key] += floor_value

        remaining = target_total - assigned
        remainders.sort(key=lambda item: (-item[0], -item[1], item[2]))
        for _, spare_capacity, key in remainders:
            if remaining <= 0:
                break
            if spare_capacity <= 0:
                continue
            allocations[key] += 1
            remaining -= 1

        return allocations

    @classmethod
    def _select_samples(
        cls,
        samples: List[Dict[str, Any]],
        *,
        max_samples: Optional[int],
        strategy: str,
        sample_seed: int,
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        available = len(samples)
        if max_samples is None or max_samples >= available:
            return list(samples), {
                "strategy": "all",
                "seed": sample_seed,
                "requested": max_samples,
                "selected": available,
                "available": available,
                "by_stratum": {},
            }

        rng = random.Random(sample_seed)
        if strategy == "sequential":
            selected = list(samples[:max_samples])
        elif strategy == "random":
            selected = list(samples)
            rng.shuffle(selected)
            selected = selected[:max_samples]
        elif strategy == "stratified":
            grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            for sample in samples:
                grouped[cls._resolve_stratum_key(sample)].append(sample)

            allocations = cls._allocate_stratified_counts(
                {key: len(group) for key, group in grouped.items()},
                max_samples,
            )

            selected = []
            for key in sorted(grouped.keys()):
                group = list(grouped[key])
                rng.shuffle(group)
                selected.extend(group[: allocations.get(key, 0)])
            rng.shuffle(selected)
        else:
            raise ValueError(f"Unsupported sample strategy: {strategy}")

        by_stratum: Dict[str, int] = {}
        for sample in selected:
            key = cls._resolve_stratum_key(sample)
            by_stratum[key] = by_stratum.get(key, 0) + 1

        return selected, {
            "strategy": strategy,
            "seed": sample_seed,
            "requested": max_samples,
            "selected": len(selected),
            "available": available,
            "by_stratum": dict(sorted(by_stratum.items())),
        }

    def _output_schema_path(self, workspace_dir: Path) -> Path:
        schema_path = workspace_dir / "formal_skill_output_schema.json"
        if schema_path.exists():
            return schema_path
        schema = _strict_json_schema(FormalSkillOutput.model_json_schema())
        schema_path.write_text(_json_dump(schema), encoding="utf-8")
        return schema_path

    def _build_prompt(self, *, payload_path: Path, launcher_path: Path) -> str:
        return self.config.prompt_template.format(
            payload_file=payload_path.name,
            launcher_file=launcher_path.name,
        )

    def _write_prepared_payload(self, payload: AnalysisPayload, target_path: Path) -> None:
        payload_dict = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
        target_path.write_text(_json_dump(payload_dict), encoding="utf-8")

    def _write_skill_launcher(
        self,
        *,
        target_path: Path,
        pipeline_script_path: Path,
        payload_path: Path,
        launcher_result_path: Path,
        pipeline_observability_path: Path,
    ) -> None:
        pipeline_timeout_sec = max(30, min(90, self.config.timeout_sec))
        launcher_base_dir = target_path.parent.resolve()

        def _launcher_embedded_path(path: Path) -> str:
            resolved = path.resolve()
            try:
                return os.path.relpath(resolved, launcher_base_dir)
            except ValueError:
                return str(resolved)

        pipeline_script_ref = _launcher_embedded_path(pipeline_script_path)
        payload_ref = _launcher_embedded_path(payload_path)
        launcher_result_ref = _launcher_embedded_path(launcher_result_path)
        pipeline_observability_ref = _launcher_embedded_path(pipeline_observability_path)
        launcher = f'''#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PIPELINE = (BASE_DIR / Path(r"{pipeline_script_ref}")).resolve()
PAYLOAD = (BASE_DIR / Path(r"{payload_ref}")).resolve()
RESULT_PATH = (BASE_DIR / Path(r"{launcher_result_ref}")).resolve()
PIPELINE_OBSERVABILITY_PATH = (BASE_DIR / Path(r"{pipeline_observability_ref}")).resolve()
OBSERVABILITY_PREFIX = "{PIPELINE_OBSERVABILITY_PREFIX}"
PIPELINE_TIMEOUT_SEC = {pipeline_timeout_sec}

def _write_result(payload):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _split_stdout(stdout_text):
    json_lines = []
    observability_payload = None
    for raw_line in (stdout_text or "").splitlines():
        if raw_line.startswith(OBSERVABILITY_PREFIX + " "):
            try:
                observability_payload = json.loads(raw_line[len(OBSERVABILITY_PREFIX) + 1:].strip())
            except json.JSONDecodeError:
                observability_payload = {{"parse_error": "invalid_observability_marker"}}
            continue
        json_lines.append(raw_line)
    json_text = "\\n".join(json_lines).strip()
    return json_text, observability_payload

try:
    completed = subprocess.run(
        [sys.executable, str(PIPELINE), "--payload-file", str(PAYLOAD)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=dict(os.environ),
        timeout=PIPELINE_TIMEOUT_SEC,
    )
except subprocess.TimeoutExpired:
    _write_result({{
        "success": False,
        "status": "timeout",
        "pipeline_returncode": 124,
        "timed_out": True,
        "stdout_json_valid": False,
        "stdout_excerpt": "",
        "stderr_excerpt": f"launcher timeout after {{PIPELINE_TIMEOUT_SEC}} seconds",
    }})
    sys.stderr.write(f"launcher timeout after {{PIPELINE_TIMEOUT_SEC}} seconds\\n")
    raise SystemExit(124)

stdout_text = completed.stdout or ""
stderr_text = completed.stderr or ""
stdout_json, observability_payload = _split_stdout(stdout_text)
if observability_payload is not None:
    PIPELINE_OBSERVABILITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    PIPELINE_OBSERVABILITY_PATH.write_text(
        json.dumps(observability_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

stdout_json_valid = False
if stdout_json:
    try:
        json.loads(stdout_json)
        stdout_json_valid = True
    except json.JSONDecodeError:
        stdout_json_valid = False

_write_result({{
    "success": completed.returncode == 0 and stdout_json_valid,
    "status": "ok" if completed.returncode == 0 and stdout_json_valid else "process_error",
    "pipeline_returncode": completed.returncode,
    "timed_out": False,
    "stdout_json_valid": stdout_json_valid,
    "stdout_excerpt": stdout_json[:2000],
    "stderr_excerpt": stderr_text[:2000],
}})

if stdout_json:
    sys.stdout.write(stdout_json)
if stderr_text:
    sys.stderr.write(stderr_text)
raise SystemExit(0 if completed.returncode == 0 and stdout_json_valid else (completed.returncode or 1))
'''
        target_path.write_text(launcher, encoding="utf-8")

    def _resolved_codex_cmd(self) -> str:
        codex_cmd = str(self.config.codex_cmd)
        if Path(codex_cmd).suffix:
            return codex_cmd
        candidates = [
            codex_cmd,
            f"{codex_cmd}.cmd",
            f"{codex_cmd}.bat",
            f"{codex_cmd}.ps1",
        ]
        for candidate in candidates:
            resolved = shutil.which(candidate)
            if resolved:
                return resolved
        return codex_cmd

    def _resolved_agent_cmd(self) -> str:
        configured = str(self.config.agent_cmd or self.config.codex_cmd or "codex").strip()
        if not configured:
            configured = "codex"
        if Path(configured).suffix:
            return configured
        resolved = shutil.which(configured)
        return resolved or configured

    def _agent_backend(self) -> str:
        backend = str(getattr(self.config, "agent_backend", "codex") or "codex").strip().lower()
        if backend not in {"codex", "cli"}:
            raise ValueError(f"Unsupported agent backend: {backend}")
        return backend

    def _codex_entrypoint(self) -> List[str]:
        codex_cmd = self._resolved_codex_cmd()
        suffix = Path(codex_cmd).suffix.lower()
        if suffix in {".cmd", ".bat"}:
            return [os.environ.get("COMSPEC") or "cmd.exe", "/c", codex_cmd]
        if suffix == ".ps1":
            powershell_cmd = shutil.which("pwsh") or shutil.which("powershell")
            if powershell_cmd:
                return [powershell_cmd, "-NoProfile", "-File", codex_cmd]
            return ["powershell.exe", "-NoProfile", "-File", codex_cmd]
        return [codex_cmd]

    def _build_cli_agent_command(
        self,
        *,
        workspace_dir: Path,
        output_schema_path: Path,
        final_output_path: Path,
        installed_skill_dir: Path,
        prompt_file_path: Path,
    ) -> List[str]:
        template = str(getattr(self.config, "agent_args_template", "") or "").strip()
        agent_cmd = self._resolved_agent_cmd()
        agent_model = str(getattr(self.config, "agent_model", None) or getattr(self.config, "codex_model", None) or "").strip()
        format_payload = {
            "agent_cmd": agent_cmd,
            "workspace_dir": workspace_dir.as_posix(),
            "output_schema_path": output_schema_path.as_posix(),
            "final_output_path": final_output_path.as_posix(),
            "installed_skill_dir": installed_skill_dir.as_posix(),
            "prompt_file": prompt_file_path.as_posix(),
            "sandbox_mode": str(self.config.sandbox_mode),
            "execution_mode": str(self.config.execution_mode),
            "agent_model": agent_model,
        }
        if not template:
            return [agent_cmd]
        return shlex.split(template.format(**format_payload), posix=(os.name != "nt"))

    def _build_codex_command(
        self,
        *,
        workspace_dir: Path,
        output_schema_path: Path,
        final_output_path: Path,
        installed_skill_dir: Path,
    ) -> List[str]:
        command = [
            *self._codex_entrypoint(),
            "exec",
            "-",
            "--json",
            "--skip-git-repo-check",
        ]
        if self.config.codex_model:
            command.extend(["--model", self.config.codex_model])
        if self.config.execution_mode == "bypass":
            command.append("--dangerously-bypass-approvals-and-sandbox")
        elif self.config.execution_mode == "sandbox":
            command.extend(["--full-auto", "--sandbox", self.config.sandbox_mode])
        else:
            raise ValueError(f"Unsupported execution mode: {self.config.execution_mode}")

        command.extend(
            [
                "--cd",
                str(workspace_dir),
                "--output-last-message",
                str(final_output_path),
                "--add-dir",
                str(installed_skill_dir),
                "--add-dir",
                str(workspace_dir),
            ]
        )
        return command

    def _build_agent_command(
        self,
        *,
        workspace_dir: Path,
        output_schema_path: Path,
        final_output_path: Path,
        installed_skill_dir: Path,
        prompt_file_path: Path,
    ) -> List[str]:
        if self._agent_backend() == "codex":
            return self._build_codex_command(
                workspace_dir=workspace_dir,
                output_schema_path=output_schema_path,
                final_output_path=final_output_path,
                installed_skill_dir=installed_skill_dir,
            )
        return self._build_cli_agent_command(
            workspace_dir=workspace_dir,
            output_schema_path=output_schema_path,
            final_output_path=final_output_path,
            installed_skill_dir=installed_skill_dir,
            prompt_file_path=prompt_file_path,
        )

    def _build_env(self, isolated_codex_dir: Path) -> Dict[str, str]:
        env = dict(os.environ)
        if self._agent_backend() != "codex":
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            return env
        home_root = str(isolated_codex_dir.parent)
        env["HOME"] = home_root
        env["USERPROFILE"] = home_root
        env["CODEX_HOME"] = str(isolated_codex_dir)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        return env

    def _embedding_runtime_snapshot(self, env: Dict[str, str]) -> Dict[str, Any]:
        return {
            "api_key_present": bool(
                env.get("SKILL_EMBEDDING_API_KEY") or env.get("AIHUBMIX_API_KEY") or env.get("OPENAI_API_KEY")
            ),
            "base_url": (
                env.get("SKILL_EMBEDDING_BASE_URL")
                or env.get("AIHUBMIX_BASE_URL")
                or env.get("OPENAI_BASE_URL")
                or DEFAULT_EMBEDDING_BASE_URL
            ).rstrip("/"),
            "embedding_model": env.get("SKILL_EMBEDDING_MODEL") or DEFAULT_EMBEDDING_MODEL,
        }

    def _parse_jsonl_events(self, stdout_text: str) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        for raw_line in (stdout_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    def _extract_command_execution_items(self, events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        command_items: List[Dict[str, Any]] = []
        for event in events:
            if not isinstance(event, dict) or event.get("type") != "item.completed":
                continue
            item = event.get("item")
            if isinstance(item, dict) and item.get("type") == "command_execution":
                command_items.append(item)
        return command_items

    def _script_name_from_command(self, command: str) -> Optional[str]:
        lowered = str(command or "").lower()
        if LAUNCHER_SCRIPT_NAME.lower() in lowered:
            return LAUNCHER_SCRIPT_NAME
        if PIPELINE_SCRIPT_NAME.lower() in lowered:
            return PIPELINE_SCRIPT_NAME
        if TIME_SCRIPT_NAME.lower() in lowered:
            return TIME_SCRIPT_NAME
        if RETRIEVE_SCRIPT_NAME.lower() in lowered:
            return RETRIEVE_SCRIPT_NAME
        return None

    def _load_json_file(self, path: Path) -> Optional[Dict[str, Any]]:
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _extract_pipeline_observability(self, text: str) -> Optional[Dict[str, Any]]:
        prefix = f"{PIPELINE_OBSERVABILITY_PREFIX} "
        raw_text = str(text or "")
        marker_index = raw_text.rfind(prefix)
        if marker_index >= 0:
            candidate = raw_text[marker_index + len(prefix):].strip()
            try:
                payload = json.loads(candidate)
            except json.JSONDecodeError:
                pass
            else:
                return payload if isinstance(payload, dict) else None
        for line in reversed((text or "").splitlines()):
            if not line.startswith(prefix):
                continue
            try:
                payload = json.loads(line[len(prefix):].strip())
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
        return None

    def _write_tool_trace(self, events: Iterable[Dict[str, Any]], target_path: Path) -> List[Dict[str, Any]]:
        tool_events: List[Dict[str, Any]] = []
        for item in self._extract_command_execution_items(events):
            command = str(item.get("command", "") or "")
            script_name = self._script_name_from_command(command)
            if not script_name:
                continue
            aggregated_output = str(item.get("aggregated_output", "") or "")
            tool_events.append(
                {
                    "script_name": script_name,
                    "status": str(item.get("status", "") or ""),
                    "exit_code": item.get("exit_code"),
                    "command": command,
                    "aggregated_output": _truncate_text(aggregated_output, max_chars=1200),
                    "output_json": _parse_json_payload(aggregated_output),
                }
            )
        target_path.write_text(_json_dump(tool_events), encoding="utf-8")
        return tool_events

    def _build_observability(
        self,
        *,
        events: Iterable[Dict[str, Any]],
        stderr_text: str,
        embedding_runtime: Dict[str, Any],
        pipeline_observability: Optional[Dict[str, Any]] = None,
        launcher_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        issues: set[str] = set()
        script_calls: List[Dict[str, Any]] = []
        failed_script_calls = 0
        time_processing = {"attempted": False, "successful": False, "failure_count": 0}
        retrieval = {
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
        }

        pipeline_observability_from_tools: Optional[Dict[str, Any]] = None

        for item in self._extract_command_execution_items(events):
            command = str(item.get("command", "") or "")
            script_name = self._script_name_from_command(command)
            if not script_name:
                continue
            aggregated_output = str(item.get("aggregated_output", "") or "")
            parsed_output = _parse_json_payload(aggregated_output)
            if script_name in {PIPELINE_SCRIPT_NAME, LAUNCHER_SCRIPT_NAME} and pipeline_observability_from_tools is None:
                pipeline_observability_from_tools = self._extract_pipeline_observability(aggregated_output)
            status = str(item.get("status", "") or "")
            exit_code = item.get("exit_code")
            failed = status == "failed" or (isinstance(exit_code, int) and exit_code != 0)
            if failed:
                failed_script_calls += 1
                issues.add("script_command_failure")

            entry = {
                "script_name": script_name,
                "status": status,
                "exit_code": exit_code,
                "failed": failed,
                "command": command,
                "aggregated_output": _truncate_text(aggregated_output, max_chars=1200),
            }
            if parsed_output is not None:
                entry["output_json"] = parsed_output
            script_calls.append(entry)

            lowered_output = aggregated_output.lower()
            if script_name == TIME_SCRIPT_NAME:
                time_processing["attempted"] = True
                if not failed:
                    time_processing["successful"] = True
                else:
                    time_processing["failure_count"] += 1
                continue

            if script_name == RETRIEVE_SCRIPT_NAME:
                retrieval["attempted"] = True
                if failed:
                    retrieval["failure_count"] += 1
                if "unrecognized arguments:" in lowered_output:
                    retrieval["quoting_failure_detected"] = True
                    issues.add("shell_quoting_failure")
                if parsed_output is not None and parsed_output.get("status") == "ok":
                    retrieval["successful"] = True
                    mode = str(parsed_output.get("mode", "") or "") or None
                    retrieval["mode"] = mode
                    retrieval["retrieved_count"] = int(parsed_output.get("count", 0) or 0)
                    message = str(parsed_output.get("message", "") or "") or None
                    retrieval["message"] = message
                    if mode and mode.startswith("fallback:"):
                        retrieval["used_fallback"] = True
                        retrieval["fallback_reason"] = mode.split(":", 1)[1] or None
                        issues.add("retrieval_fallback")
                    lowered_mode = (mode or "").lower()
                    lowered_message = (message or "").lower()
                    if any(token in lowered_mode for token in ("connecterror", "connect_error", "network")) or any(
                        token in lowered_message for token in ("winerror 10013", "socket", "connecterror", "network")
                    ):
                        retrieval["network_error_detected"] = True
                        issues.add("retrieval_network_blocked")

        pipeline_observability = (
            pipeline_observability
            or pipeline_observability_from_tools
            or self._extract_pipeline_observability(stderr_text)
        )
        if pipeline_observability:
            for item in pipeline_observability.get("script_calls") or []:
                if not isinstance(item, dict):
                    continue
                if item.get("script_name") == PIPELINE_SCRIPT_NAME and any(existing.get("script_name") == PIPELINE_SCRIPT_NAME for existing in script_calls):
                    continue
                script_calls.append(item)
            failed_script_calls += int(((pipeline_observability.get("summary") or {}).get("failed_script_calls", 0)) or 0)

            pipeline_time = pipeline_observability.get("time_processing") or {}
            if isinstance(pipeline_time, dict):
                time_processing["attempted"] = time_processing["attempted"] or bool(pipeline_time.get("attempted"))
                time_processing["successful"] = time_processing["successful"] or bool(pipeline_time.get("successful"))
                time_processing["failure_count"] += int(pipeline_time.get("failure_count", 0) or 0)
                if pipeline_time.get("mode") is not None:
                    time_processing["mode"] = pipeline_time.get("mode")

            pipeline_retrieval = pipeline_observability.get("retrieval") or {}
            if isinstance(pipeline_retrieval, dict):
                retrieval["attempted"] = retrieval["attempted"] or bool(pipeline_retrieval.get("attempted"))
                retrieval["successful"] = retrieval["successful"] or bool(pipeline_retrieval.get("successful"))
                retrieval["used_fallback"] = retrieval["used_fallback"] or bool(pipeline_retrieval.get("used_fallback"))
                retrieval["quoting_failure_detected"] = retrieval["quoting_failure_detected"] or bool(pipeline_retrieval.get("quoting_failure_detected"))
                retrieval["network_error_detected"] = retrieval["network_error_detected"] or bool(pipeline_retrieval.get("network_error_detected"))
                retrieval["failure_count"] += int(pipeline_retrieval.get("failure_count", 0) or 0)
                if pipeline_retrieval.get("mode") is not None:
                    retrieval["mode"] = pipeline_retrieval.get("mode")
                if pipeline_retrieval.get("fallback_reason") is not None:
                    retrieval["fallback_reason"] = pipeline_retrieval.get("fallback_reason")
                if pipeline_retrieval.get("message") is not None:
                    retrieval["message"] = pipeline_retrieval.get("message")
                retrieval["retrieved_count"] = max(retrieval["retrieved_count"], int(pipeline_retrieval.get("retrieved_count", 0) or 0))

            issues.update(str(item) for item in (pipeline_observability.get("issues") or []) if str(item).strip())

        launcher_payload = launcher_result if isinstance(launcher_result, dict) else {}
        if launcher_payload:
            launcher_status = str(launcher_payload.get("status", "") or "").strip() or None
            if launcher_status == "timeout":
                issues.add("launcher_timeout")
                issues.add("script_command_failure")
                failed_script_calls += 1
            elif not bool(launcher_payload.get("success")):
                issues.add("launcher_failed")
                issues.add("script_command_failure")
                failed_script_calls += 1

        stderr_lines = []
        for line in (stderr_text or "").splitlines():
            if line.startswith(f"{PIPELINE_OBSERVABILITY_PREFIX} "):
                continue
            if line.strip():
                stderr_lines.append(line)
        stderr_non_empty = bool("\n".join(stderr_lines).strip())
        if stderr_non_empty:
            issues.add("stderr_non_empty")

        fatal_agent_error = _detect_fatal_agent_error(stderr_text)
        if fatal_agent_error:
            issues.add(fatal_agent_error)

        return {
            "fatal_agent_error": fatal_agent_error,
            "embedding_runtime": embedding_runtime,
            "summary": {
                "script_call_count": len(script_calls),
                "failed_script_calls": failed_script_calls,
                "stderr_non_empty": stderr_non_empty,
            },
            "time_processing": time_processing,
            "retrieval": retrieval,
            "issues": sorted(issues),
            "script_calls": script_calls,
            "launcher": launcher_payload,
        }

    def _write_agent_output(
        self,
        final_output_path: Path,
        target_path: Path,
        *,
        launcher_result: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_text = final_output_path.read_text(encoding="utf-8").strip() if final_output_path.exists() else ""
        launcher_payload = launcher_result if isinstance(launcher_result, dict) else {}
        launcher_stdout_text = _strip_observability_marker_lines(str(launcher_payload.get("stdout_excerpt", "") or ""))
        candidate_text = _strip_observability_marker_lines(raw_text)

        parsed_json = None
        parsed_source = None
        for source_name, source_text in (
            ("launcher_result.stdout_excerpt", launcher_stdout_text),
            ("final_output", candidate_text),
        ):
            parsed_json = _parse_json_payload(source_text)
            if parsed_json is not None:
                parsed_source = source_name
                break

        launcher_success = bool(launcher_payload.get('success')) if launcher_result is not None else (parsed_json is not None)
        payload = {
            "raw_text": raw_text,
            "parsed_json": parsed_json,
            "json_valid": parsed_json is not None and launcher_success,
            "launcher_success": launcher_success,
            "launcher_status": launcher_payload.get("status"),
            "json_source": parsed_source,
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
        run_root = workspace_dir / _run_dirname(skill_version)
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
        _runner_log(f"schema_consistency ok={schema_consistency.get('ok')} warnings={len(schema_consistency.get('warnings') or [])}")
        if not schema_consistency.get('ok'):
            raise ValueError(f"skill schema contract drift detected: {schema_consistency_path}")

        with dataset_path.open("r", encoding="utf-8") as f:
            samples = [json.loads(line) for line in f if line.strip()]
        _runner_log(f"dataset={_display_path(dataset_path)} total_samples={len(samples)}")
        samples, sampling_info = self._select_samples(
            samples,
            max_samples=self.config.max_samples,
            strategy=self.config.sample_strategy,
            sample_seed=self.config.sample_seed,
        )

        _runner_log(f"selected_samples={len(samples)} strategy={sampling_info.get('strategy')} execution_mode={self.config.execution_mode}")
        run_started_at = time.time()
        manifest: Dict[str, Any] = {
            "runner_mode": "agent",
            "skill_version": skill_version,
            "aborted_early": False,
            "fatal_agent_error": None,
            "skill_source_dir": to_relative_posix_path(skill_source_dir, run_root),
            "dataset_path": to_relative_posix_path(dataset_path, run_root),
            "run_root": ".",
            "isolated_codex_dir": to_relative_posix_path(isolated_codex_dir, run_root),
            "installed_skill_dir": to_relative_posix_path(installed_skill_dir, run_root),
            "install_mode": "source_snapshot",
            "execution_mode": self.config.execution_mode,
            "sandbox_mode": self.config.sandbox_mode if self.config.execution_mode == "sandbox" else None,
            "codex_model": self.config.codex_model,
            "agent_backend": self._agent_backend(),
            "agent_model": str(getattr(self.config, "agent_model", None) or getattr(self.config, "codex_model", None) or "") or None,
            "sampling": sampling_info,
            "schema_consistency_path": to_relative_posix_path(schema_consistency_path, run_root),
            "schema_consistency_ok": bool(schema_consistency.get('ok')),
            "schema_consistency_warnings": schema_consistency.get("warnings", []),
            "samples": [],
        }

        for index, sample in enumerate(samples):
            sample_id = str(sample.get("sample_id", f"sample_{index:05d}"))
            _runner_log(f"sample {index + 1}/{len(samples)} start {sample_id}")
            sample_dir = run_root / f"eval-{index:04d}-{_safe_slug(sample_id)}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            payload = self._build_analysis_payload(sample)

            sample_input_path = sample_dir / "sample_input.json"
            self._write_prepared_payload(payload, sample_input_path)
            payload_path = sample_dir / "payload.json"
            self._write_prepared_payload(payload, payload_path)
            launcher_path = sample_dir / LAUNCHER_SCRIPT_NAME
            launcher_result_path = sample_dir / LAUNCHER_RESULT_FILENAME
            pipeline_observability_path = sample_dir / LAUNCHER_OBSERVABILITY_FILENAME
            self._write_skill_launcher(
                target_path=launcher_path,
                pipeline_script_path=installed_skill_dir / "scripts" / PIPELINE_SCRIPT_NAME,
                payload_path=payload_path,
                launcher_result_path=launcher_result_path,
                pipeline_observability_path=pipeline_observability_path,
            )
            prompt = self._build_prompt(payload_path=payload_path, launcher_path=launcher_path)
            prompt_file_path = sample_dir / "agent_prompt.txt"
            prompt_file_path.write_text(prompt, encoding="utf-8")

            gold_payload = {
                "sample_id": sample_id,
                "is_minor": bool(sample.get("is_minor", False)),
                "source": sample.get("source", "unknown"),
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
            embedding_runtime = self._embedding_runtime_snapshot(env)
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
            tool_trace = self._write_tool_trace(events, tool_trace_path)
            launcher_result = self._load_json_file(launcher_result_path)
            pipeline_observability = self._load_json_file(pipeline_observability_path)
            observability = self._build_observability(
                events=events,
                stderr_text=stderr_text,
                embedding_runtime=embedding_runtime,
                pipeline_observability=pipeline_observability,
                launcher_result=launcher_result,
            )
            observability_path.write_text(_json_dump(observability), encoding="utf-8")
            agent_output = self._write_agent_output(
                final_output_path,
                agent_output_path,
                launcher_result=launcher_result,
            )

            timing_payload = {
                "duration_ms": int(elapsed * 1000),
                "total_duration_seconds": round(elapsed, 3),
            }
            timing_path.write_text(_json_dump(timing_payload), encoding="utf-8")

            metadata_payload = {
                "sample_id": sample_id,
                "fatal_agent_error": observability.get("fatal_agent_error"),
                "skill_version": skill_version,
                "skill_source_dir": to_relative_posix_path(skill_source_dir, sample_dir),
                "installed_skill_dir": to_relative_posix_path(installed_skill_dir, sample_dir),
                "install_mode": "source_snapshot",
                "command": command,
                "execution_mode": self.config.execution_mode,
                "sandbox_mode": self.config.sandbox_mode if self.config.execution_mode == "sandbox" else None,
                "codex_model": self.config.codex_model,
                "embedding_runtime": embedding_runtime,
                "returncode": completed.returncode,
                "output_schema_path": to_relative_posix_path(output_schema_path, sample_dir),
                "stdout_path": to_relative_posix_path(stdout_path, sample_dir),
                "stderr_path": to_relative_posix_path(stderr_path, sample_dir),
                "tool_trace_path": to_relative_posix_path(tool_trace_path, sample_dir),
                "observability_path": to_relative_posix_path(observability_path, sample_dir),
                "agent_output_path": to_relative_posix_path(agent_output_path, sample_dir),
                "timing_path": to_relative_posix_path(timing_path, sample_dir),
                "sample_input_path": to_relative_posix_path(sample_input_path, sample_dir),
                "payload_path": to_relative_posix_path(payload_path, sample_dir),
                "prompt_file_path": to_relative_posix_path(prompt_file_path, sample_dir),
                "launcher_path": to_relative_posix_path(launcher_path, sample_dir),
                "launcher_result_path": to_relative_posix_path(launcher_result_path, sample_dir),
                "pipeline_observability_path": to_relative_posix_path(pipeline_observability_path, sample_dir),
                "launcher_status": (launcher_result or {}).get("status"),
                "launcher_success": bool((launcher_result or {}).get('success')),
                "json_valid": agent_output["json_valid"],
                "agent_backend": self._agent_backend(),
                "agent_model": str(getattr(self.config, "agent_model", None) or getattr(self.config, "codex_model", None) or "") or None,
            }
            metadata_path.write_text(_json_dump(metadata_payload), encoding="utf-8")

            _runner_log(f"sample {index + 1}/{len(samples)} done returncode={completed.returncode} json_valid={agent_output.get('json_valid')} launcher_success={bool((launcher_result or {}).get('success'))} retrieval_mode={((observability.get('retrieval') or {}).get('mode'))}")
            manifest["samples"].append(
                {
                    "sample_id": sample_id,
                    "sample_dir": to_relative_posix_path(sample_dir, run_root),
                    "returncode": completed.returncode,
                    "json_valid": agent_output["json_valid"],
                    "launcher_success": bool((launcher_result or {}).get('success')),
                    "duration_ms": int(elapsed * 1000),
                    "duration_seconds": round(elapsed, 3),
                    "observed_issues": observability.get("issues", []),
                    "fatal_agent_error": observability.get("fatal_agent_error"),
                    "retrieval_mode": ((observability.get('retrieval') or {}).get('mode')),
                    "tool_trace_entries": len(tool_trace),
                }
            )

            if observability.get("fatal_agent_error"):
                manifest["aborted_early"] = True
                manifest["fatal_agent_error"] = observability.get("fatal_agent_error")
                break

        manifest["counts"] = {
            "sample_count": len(manifest["samples"]),
            "json_valid_count": sum(1 for item in manifest["samples"] if item.get("json_valid")),
            "launcher_success_count": sum(1 for item in manifest["samples"] if item.get("launcher_success")),
            "fatal_agent_error_count": sum(1 for item in manifest["samples"] if item.get("fatal_agent_error")),
        }
        manifest["timing"] = _build_timing_summary(manifest["samples"], total_wall_seconds=time.time() - run_started_at)
        manifest_path = run_root / "run_manifest.json"
        manifest_path.write_text(_json_dump(manifest), encoding="utf-8")
        return run_root
