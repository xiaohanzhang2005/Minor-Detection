#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260322_132125\baseline\run_minor-detection-v0.1.0\codex_home\.codex\skills\minor-detection-v0.1.0\scripts\run_minor_detection_pipeline.py")
PAYLOAD = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260322_132125\baseline\run_minor-detection-v0.1.0\eval-0000-social_pos_002016_gemini-3-flash-preview_2017\payload.json")
RESULT_PATH = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260322_132125\baseline\run_minor-detection-v0.1.0\eval-0000-social_pos_002016_gemini-3-flash-preview_2017\launcher_result.json")
PIPELINE_OBSERVABILITY_PATH = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260322_132125\baseline\run_minor-detection-v0.1.0\eval-0000-social_pos_002016_gemini-3-flash-preview_2017\pipeline_observability.json")
OBSERVABILITY_PREFIX = "[MINOR_PIPELINE_OBSERVABILITY]"
PIPELINE_TIMEOUT_SEC = 90

def _write_result(payload):
    RESULT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _split_stdout(stdout_text):
    json_lines = []
    observability_payload = None
    for raw_line in (stdout_text or "").splitlines():
        if raw_line.startswith(OBSERVABILITY_PREFIX + " "):
            try:
                observability_payload = json.loads(raw_line[len(OBSERVABILITY_PREFIX) + 1:].strip())
            except json.JSONDecodeError:
                observability_payload = {"parse_error": "invalid_observability_marker"}
            continue
        json_lines.append(raw_line)
    json_text = "\n".join(json_lines).strip()
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
    _write_result({
        "success": False,
        "status": "timeout",
        "pipeline_returncode": 124,
        "timed_out": True,
        "stdout_json_valid": False,
        "stdout_excerpt": "",
        "stderr_excerpt": f"launcher timeout after {PIPELINE_TIMEOUT_SEC} seconds",
    })
    sys.stderr.write(f"launcher timeout after {PIPELINE_TIMEOUT_SEC} seconds\n")
    raise SystemExit(124)

stdout_text = completed.stdout or ""
stderr_text = completed.stderr or ""
stdout_json, observability_payload = _split_stdout(stdout_text)
if observability_payload is not None:
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

_write_result({
    "success": completed.returncode == 0 and stdout_json_valid,
    "status": "ok" if completed.returncode == 0 and stdout_json_valid else "process_error",
    "pipeline_returncode": completed.returncode,
    "timed_out": False,
    "stdout_json_valid": stdout_json_valid,
    "stdout_excerpt": stdout_json[:2000],
    "stderr_excerpt": stderr_text[:2000],
})

if stdout_json:
    sys.stdout.write(stdout_json)
if stderr_text:
    sys.stderr.write(stderr_text)
raise SystemExit(0 if completed.returncode == 0 and stdout_json_valid else (completed.returncode or 1))
