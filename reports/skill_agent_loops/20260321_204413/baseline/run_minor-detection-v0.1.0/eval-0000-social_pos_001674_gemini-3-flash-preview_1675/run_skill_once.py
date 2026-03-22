#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PIPELINE = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260321_204413\baseline\run_minor-detection-v0.1.0\codex_home\.codex\skills\minor-detection-v0.1.0\scripts\run_minor_detection_pipeline.py")
PAYLOAD = Path(r"D:\重生之我要当桌面\teenager_AI\minor-protection\reports\skill_agent_loops\20260321_204413\baseline\run_minor-detection-v0.1.0\eval-0000-social_pos_001674_gemini-3-flash-preview_1675\payload.json")

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
