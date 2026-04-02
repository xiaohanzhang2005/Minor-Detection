from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


VALIDATION_SEED_MARKER = "Temporary Validation Seed Override"
DEFAULT_BASELINE_SEMVER = "v0.1.0"


@dataclass(frozen=True)
class ValidationSeedBundle:
    source_dir: Path
    version_name: str
    run_tag: str
    metadata_path: Path


def build_validation_seed_version_name(*, base_name: str, run_tag: str) -> Tuple[str, str]:
    normalized_base = str(base_name or "").strip()
    normalized_tag = str(run_tag or "").strip()
    if not normalized_base:
        raise ValueError("base_name must be non-empty")
    if not normalized_tag:
        raise ValueError("run_tag must be non-empty")
    return f"{normalized_base}-{DEFAULT_BASELINE_SEMVER}-{normalized_tag}", normalized_tag


def _append_override_marker(path: Path) -> None:
    if not path.exists():
        return
    original = path.read_text(encoding="utf-8")
    if VALIDATION_SEED_MARKER in original:
        return
    suffix = (
        "\n\n"
        f"<!-- {VALIDATION_SEED_MARKER} -->\n"
        "- Temporary Validation Seed Override\n"
        "- This file belongs to a disposable Mode A validation seed bundle.\n"
    )
    path.write_text(original.rstrip() + suffix + "\n", encoding="utf-8")


def create_mode_a_validation_seed(
    *,
    source_dir: Path,
    output_root: Path,
    base_name: str,
    run_tag: str,
) -> ValidationSeedBundle:
    version_name, normalized_tag = build_validation_seed_version_name(base_name=base_name, run_tag=run_tag)
    source_dir = Path(source_dir)
    output_root = Path(output_root)
    target_dir = output_root / version_name

    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir does not exist: {source_dir}")
    if target_dir.exists():
        shutil.rmtree(target_dir)

    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

    _append_override_marker(target_dir / "references" / "classifier-system.md")
    _append_override_marker(target_dir / "references" / "evidence-rules.md")

    metadata_path = target_dir / ".validation_seed.json"
    metadata_path.write_text(
        json.dumps(
            {
                "version_name": version_name,
                "base_name": str(base_name or "").strip(),
                "run_tag": normalized_tag,
                "source_dir": str(source_dir),
                "seed_source_dir": str(target_dir),
                "seed_type": "mode_a_validation",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return ValidationSeedBundle(
        source_dir=target_dir,
        version_name=version_name,
        run_tag=normalized_tag,
        metadata_path=metadata_path,
    )


def build_mode_a_validation_command(
    *,
    baseline_version: str,
    baseline_source_dir: Path,
    dataset_path: Path,
    max_samples: Optional[int] = None,
    codex_model: Optional[str] = None,
    workspace_root: Optional[Path] = None,
    execution_mode: str = "bypass",
    timeout_sec: int = 600,
) -> str:
    parts = [
        "python scripts/run_skill_iteration_loop.py",
        f"--baseline-version {baseline_version}",
        f"--baseline-source-dir {Path(baseline_source_dir)}",
        "--refresh-baseline-version",
        f"--dataset {Path(dataset_path)}",
        "--max-rounds 1",
    ]
    if max_samples is not None:
        parts.append(f"--max-samples {int(max_samples)}")
    if codex_model:
        parts.append(f"--codex-model {codex_model}")
    if workspace_root is not None:
        parts.append(f"--workspace-root {Path(workspace_root)}")
    if execution_mode:
        parts.append(f"--execution-mode {execution_mode}")
    parts.append(f"--timeout-sec {int(timeout_sec)}")
    return " ".join(parts)


def build_mode_a_validation_payload(
    *,
    source_dir: Path,
    output_root: Path,
    dataset_path: Path,
    base_name: str,
    run_tag: str,
    max_samples: Optional[int] = None,
    codex_model: Optional[str] = None,
    workspace_root: Optional[Path] = None,
    execution_mode: str = "bypass",
    timeout_sec: int = 600,
) -> Dict[str, Any]:
    seed = create_mode_a_validation_seed(
        source_dir=Path(source_dir),
        output_root=Path(output_root),
        base_name=base_name,
        run_tag=run_tag,
    )
    run_command = build_mode_a_validation_command(
        baseline_version=seed.version_name,
        baseline_source_dir=seed.source_dir,
        dataset_path=Path(dataset_path),
        max_samples=max_samples,
        codex_model=codex_model,
        workspace_root=workspace_root,
        execution_mode=execution_mode,
        timeout_sec=timeout_sec,
    )
    cleanup = {
        "cleanup_skills_command": (
            "python scripts/cleanup_skill_versions.py "
            f"--base-name {str(base_name or '').strip()} "
            "--keep-latest-stable 0 "
            f"--only-run-tag {seed.run_tag}"
        ),
        "cleanup_seed_source_command": (
            "python -c "
            f"\"import shutil; shutil.rmtree(r'{seed.source_dir}', ignore_errors=True)\""
        ),
    }
    return {
        "seed": {
            "version_name": seed.version_name,
            "source_dir": str(seed.source_dir),
            "run_tag": seed.run_tag,
            "metadata_path": str(seed.metadata_path),
        },
        "run_command": run_command,
        "cleanup": cleanup,
    }
