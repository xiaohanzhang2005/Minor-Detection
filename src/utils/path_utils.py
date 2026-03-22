# 模块说明：
# - 把绝对路径归一化成项目内稳定相对路径。
# - 主要服务 loop 报告和 CLI JSON 输出。

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def to_relative_posix_path(path: Path | str, start: Path | str) -> str:
    target = Path(path)
    base_dir = Path(start)
    return Path(os.path.relpath(target, start=base_dir)).as_posix()


def maybe_relativize_project_path(value: Any, project_root: Path | str, start: Path | str) -> Any:
    if not isinstance(value, str):
        return value

    candidate = Path(value)
    if not candidate.is_absolute():
        return value

    root = Path(project_root)
    try:
        candidate.relative_to(root)
    except ValueError:
        return value

    return to_relative_posix_path(candidate, start)


def normalize_project_paths(payload: Any, project_root: Path | str, start: Path | str) -> Any:
    if isinstance(payload, dict):
        return {
            key: normalize_project_paths(value, project_root=project_root, start=start)
            for key, value in payload.items()
        }

    if isinstance(payload, list):
        return [
            normalize_project_paths(item, project_root=project_root, start=start)
            for item in payload
        ]

    return maybe_relativize_project_path(payload, project_root=project_root, start=start)
