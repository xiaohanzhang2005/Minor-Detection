#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260321_202330\round_01\candidate\run_minor-detection-v0.1.1-rc001\codex_home\.codex\skills\minor-detection-v0.1.1-rc001\scripts\run_minor_detection_pipeline.py")
PAYLOAD = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260321_202330\round_01\candidate\run_minor-detection-v0.1.1-rc001\eval-0001-social_pos_002016_gemini-3-flash-preview_2017\payload.json")

completed = subprocess.run(
    [sys.executable, str(PIPELINE), "--payload-file", str(PAYLOAD)],
    capture_output=True,
    text=True,
    encoding="utf-8",
    errors="replace",
    env=dict(os.environ),
)
if completed.stdout:
    sys.stdout.write(completed.stdout)
if completed.stderr:
    sys.stderr.write(completed.stderr)
raise SystemExit(completed.returncode)
