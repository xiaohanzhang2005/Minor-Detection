# 模块说明：
# - 把绝对路径归一化成项目内稳定相对路径。
# - 主要服务 loop 报告和 CLI JSON 输出。

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"[A-Za-z]:[\\/][^\s\"']+")


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

    try:
        return to_relative_posix_path(candidate, start)
    except ValueError:
        # Windows tests may create temp workspaces on a different drive than the repo.
        # Fall back to a stable project-root-relative path instead of crashing.
        return to_relative_posix_path(candidate, root)


def _project_root_aliases(project_root: Path | str) -> list[str]:
    root = Path(project_root)
    aliases = {str(root), root.as_posix()}

    posix_value = root.as_posix()
    parts = posix_value.split("/")
    if len(parts) >= 4 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        remainder = "/".join(parts[3:])
        remainder_windows = remainder.replace("/", "\\")
        if remainder:
            aliases.add(f"{drive}:/{remainder}")
            aliases.add(f"{drive}:\\{remainder_windows}")
        else:
            aliases.add(f"{drive}:/")
            aliases.add(f"{drive}:\\")

    normalized = str(root).replace("\\", "/")
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] == "/":
        drive = normalized[0].lower()
        remainder = normalized[3:]
        if remainder:
            aliases.add(f"/mnt/{drive}/{remainder}")
        else:
            aliases.add(f"/mnt/{drive}")

    escaped_aliases = {alias.replace("\\", "\\\\") for alias in aliases if "\\" in alias}
    doubled_slash_aliases = {alias.replace("/", "//") for alias in aliases if "/" in alias}
    aliases.update(escaped_aliases)
    aliases.update(doubled_slash_aliases)

    return sorted({alias for alias in aliases if alias}, key=len, reverse=True)


def sanitize_project_text(value: Any, project_root: Path | str, start: Path | str) -> Any:
    if not isinstance(value, str):
        return value

    relativized = maybe_relativize_project_path(value, project_root=project_root, start=start)
    if relativized != value:
        return relativized

    try:
        root_display = to_relative_posix_path(Path(project_root), start)
    except ValueError:
        root_display = "."

    replacement = root_display or "."
    replacement_with_sep = f"{replacement}/" if replacement != "." else "./"

    sanitized = value
    replaced = False
    for alias in _project_root_aliases(project_root):
        updated = sanitized.replace(alias + "\\", replacement_with_sep)
        updated = updated.replace(alias + "/", replacement_with_sep)
        updated = updated.replace(alias, replacement)
        if updated != sanitized:
            replaced = True
            sanitized = updated
    if replaced:
        sanitized = sanitized.replace("\\", "/")
    sanitized = WINDOWS_ABSOLUTE_PATH_RE.sub(lambda match: Path(match.group(0).replace("\\", "/")).name, sanitized)
    return sanitized


def normalize_project_paths(payload: Any, project_root: Path | str, start: Path | str) -> Any:
    if isinstance(payload, Path):
        return maybe_relativize_project_path(str(payload), project_root=project_root, start=start)

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

    return sanitize_project_text(payload, project_root=project_root, start=start)
