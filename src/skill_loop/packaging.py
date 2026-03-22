# 模块说明：
# - 做 bundled skill 校验、快照安装和打包。
# - 通过 claude-skill-creator 连接 skill 校验与分发。

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable


EXCLUDE_DIRS = {"__pycache__", "node_modules"}
EXCLUDE_GLOBS = {"*.pyc"}
EXCLUDE_FILES = {".DS_Store"}
ROOT_EXCLUDE_DIRS = {"evals"}


def _should_exclude(rel_path: Path) -> bool:
    parts = rel_path.parts
    if any(part in EXCLUDE_DIRS for part in parts):
        return True
    if len(parts) > 1 and parts[1] in ROOT_EXCLUDE_DIRS:
        return True
    name = rel_path.name
    if name in EXCLUDE_FILES:
        return True
    return any(rel_path.match(f"**/{pattern}") for pattern in EXCLUDE_GLOBS)


def _ignore_snapshot_copy(directory: str, names: list[str]) -> set[str]:
    base_dir = Path(directory)
    ignored: set[str] = set()
    for name in names:
        rel_path = Path(base_dir.name) / name
        if _should_exclude(rel_path):
            ignored.add(name)
    return ignored


def _load_validate_skill(project_root: Path) -> Callable[[Path], tuple[bool, str]]:
    module_path = project_root / "claude-skill-creator" / "scripts" / "quick_validate.py"
    spec = importlib.util.spec_from_file_location("claude_skill_quick_validate", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load skill validator from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    validator = getattr(module, "validate_skill", None)
    if validator is None:
        raise AttributeError(f"validate_skill not found in {module_path}")
    return validator


def _package_script_path(project_root: Path) -> Path:
    return project_root / "claude-skill-creator" / "scripts" / "package_skill.py"


def validate_skill_source(*, project_root: Path, skill_dir: Path) -> Path:
    skill_dir = Path(skill_dir).resolve()
    if not skill_dir.exists():
        raise FileNotFoundError(f"Skill folder not found: {skill_dir}")
    if not skill_dir.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {skill_dir}")
    validate_skill = _load_validate_skill(project_root)
    valid, message = validate_skill(skill_dir)
    if not valid:
        raise RuntimeError(f"Skill validation failed for {skill_dir}: {message}")
    return skill_dir




def _snapshot_install_dirname(source_name: str) -> str:
    normalized = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in str(source_name or "").strip())
    prefix = normalized.strip("-_")[:12] or "skill"
    digest = hashlib.sha1(str(source_name or "skill").encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"

def install_skill_snapshot(*, project_root: Path, skill_dir: Path, target_dir: Path) -> Path:
    source_dir = validate_skill_source(project_root=project_root, skill_dir=skill_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    installed_dir = target_dir / _snapshot_install_dirname(source_dir.name)
    if installed_dir.exists():
        shutil.rmtree(installed_dir)
    shutil.copytree(source_dir, installed_dir, ignore=_ignore_snapshot_copy)
    return installed_dir


def package_skill_version(
    *,
    project_root: Path,
    skill_dir: Path,
    output_dir: Path,
    python_executable: str = sys.executable,
    timeout_sec: int = 120,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    creator_root = project_root / "claude-skill-creator"
    command = [
        python_executable,
        str(_package_script_path(project_root)),
        str(skill_dir),
        str(output_dir),
    ]
    env = dict(os.environ)
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    pythonpath_parts = [str(creator_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=timeout_sec,
        cwd=str(creator_root),
        env=env,
    )
    expected_path = output_dir / f"{skill_dir.name}.skill"
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise RuntimeError(f"Failed to package skill: {skill_dir}; detail={detail}")
    if not expected_path.exists():
        raise FileNotFoundError(f"Packaged skill not found: {expected_path}")
    return expected_path


def unpack_skill_package(*, skill_package_path: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(skill_package_path, "r") as zipf:
        zipf.extractall(target_dir)
        roots = sorted({Path(name).parts[0] for name in zipf.namelist() if Path(name).parts})
    if not roots:
        raise RuntimeError(f"Skill package is empty: {skill_package_path}")
    extracted_root = target_dir / roots[0]
    if not extracted_root.exists():
        raise FileNotFoundError(f"Extracted skill root not found: {extracted_root}")
    return extracted_root
