# 妯″潡璇存槑锛?
# - 绠＄悊 skill 鐗堟湰鍛藉悕銆佸揩鐓с€佸簱瀛樺拰娓呯悊棰勮銆?
# - 鍒犻櫎鏃х増鏈墠搴斿厛鐪嬭繖閲岀殑瑙勫垯銆?

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional


VERSION_RE = re.compile(
    r"^(?P<base>.+)-v(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:-rc(?P<rc>\d+))?(?:-(?P<run_tag>\d{8}_\d{6}))?$"
)
SNAPSHOT_MANIFEST_NAME = ".skill_snapshot_manifest.json"


def parse_version_name(version_name: str) -> Optional[dict]:
    match = VERSION_RE.match(version_name.strip())
    if not match:
        return None
    payload = match.groupdict()
    return {
        "base": payload["base"],
        "major": int(payload["major"]),
        "minor": int(payload["minor"]),
        "patch": int(payload["patch"]),
        "rc": int(payload["rc"]) if payload.get("rc") else None,
        "run_tag": payload.get("run_tag") or None,
    }


def build_stable_version_name(base_name: str, major: int, minor: int, patch: int) -> str:
    return f"{base_name}-v{major}.{minor}.{patch}"


def build_stamped_stable_version_name(stable_version_name: str, run_tag: str) -> str:
    parsed = parse_version_name(stable_version_name)
    if parsed is None or parsed["rc"] is not None:
        raise ValueError(f"Invalid stable version name: {stable_version_name}")
    run_tag = str(run_tag or "").strip()
    if not run_tag:
        raise ValueError("run_tag must be non-empty")
    semantic_stable = build_stable_version_name(parsed["base"], parsed["major"], parsed["minor"], parsed["patch"])
    return f"{semantic_stable}-{run_tag}"


def build_candidate_version_name(stable_version_name: str, rc_index: int, run_tag: Optional[str] = None) -> str:
    parsed = parse_version_name(stable_version_name)
    if parsed is None:
        raise ValueError(f"Invalid stable version name: {stable_version_name}")
    semantic_stable = build_stable_version_name(parsed["base"], parsed["major"], parsed["minor"], parsed["patch"])
    suffix = f"-{run_tag}" if str(run_tag or "").strip() else ""
    return f"{semantic_stable}-rc{rc_index:03d}{suffix}"


def next_available_candidate_version_name(
    stable_version_name: str,
    versions_root: Path,
    start_index: int = 1,
    run_tag: Optional[str] = None,
) -> str:
    rc_index = max(1, int(start_index))
    while True:
        candidate_version_name = build_candidate_version_name(stable_version_name, rc_index, run_tag=run_tag)
        if not (versions_root / candidate_version_name).exists():
            return candidate_version_name
        rc_index += 1


def next_patch_version_name(current_stable_version: str) -> str:
    parsed = parse_version_name(current_stable_version)
    if parsed is None or parsed["rc"] is not None:
        raise ValueError(f"Invalid stable version name: {current_stable_version}")
    return build_stable_version_name(
        parsed["base"],
        parsed["major"],
        parsed["minor"],
        parsed["patch"] + 1,
    )


def iter_skill_versions(versions_root: Path, *, base_name: str) -> list[dict]:
    entries: list[dict] = []
    for path in sorted(versions_root.iterdir()) if versions_root.exists() else []:
        if not path.is_dir():
            continue
        parsed = parse_version_name(path.name)
        if parsed is None or parsed["base"] != base_name:
            continue
        manifest = _load_snapshot_manifest(path)
        entries.append(
            {
                "version": path.name,
                "path": path,
                "parsed": parsed,
                "is_candidate": parsed["rc"] is not None,
                "run_tag": parsed.get("run_tag"),
                "semantic_version": build_stable_version_name(parsed["base"], parsed["major"], parsed["minor"], parsed["patch"]),
                "has_snapshot_manifest": manifest is not None,
                "mtime": path.stat().st_mtime,
            }
        )
    return sorted(
        entries,
        key=lambda item: (
            item["parsed"]["major"],
            item["parsed"]["minor"],
            item["parsed"]["patch"],
            item["parsed"].get("run_tag") or "",
            -1 if item["parsed"]["rc"] is None else item["parsed"]["rc"],
        ),
        reverse=True,
    )


def select_cleanup_targets(
    versions_root: Path,
    *,
    base_name: str,
    keep_latest_stable: int = 2,
    delete_candidates: bool = True,
    only_run_tag: Optional[str] = None,
    exclude_run_tag: Optional[str] = None,
) -> list[Path]:
    keep_latest_stable = max(0, int(keep_latest_stable))
    only_run_tag = str(only_run_tag or "").strip() or None
    exclude_run_tag = str(exclude_run_tag or "").strip() or None
    entries = _filter_inventory_entries(
        iter_skill_versions(versions_root, base_name=base_name),
        only_run_tag=only_run_tag,
        exclude_run_tag=exclude_run_tag,
    )
    stable_entries = [item for item in entries if not item["is_candidate"]]
    candidate_entries = [item for item in entries if item["is_candidate"]]

    targets: list[Path] = []
    if delete_candidates:
        targets.extend(item["path"] for item in candidate_entries)
    if len(stable_entries) > keep_latest_stable:
        targets.extend(item["path"] for item in stable_entries[keep_latest_stable:])

    filtered: list[Path] = []
    for path in sorted(dict.fromkeys(targets)):
        parsed = parse_version_name(path.name) or {}
        run_tag = parsed.get("run_tag")
        if only_run_tag is not None and run_tag != only_run_tag:
            continue
        if exclude_run_tag is not None and run_tag == exclude_run_tag:
            continue
        filtered.append(path)
    return filtered


def delete_skill_versions(paths: Iterable[Path]) -> list[Path]:
    removed: list[Path] = []
    for path in paths:
        if not path.exists():
            continue
        shutil.rmtree(path)
        removed.append(path)
    return removed


def _filter_inventory_entries(
    entries: list[dict],
    *,
    only_run_tag: Optional[str] = None,
    exclude_run_tag: Optional[str] = None,
) -> list[dict]:
    only_run_tag = str(only_run_tag or "").strip() or None
    exclude_run_tag = str(exclude_run_tag or "").strip() or None
    filtered: list[dict] = []
    for item in entries:
        run_tag = item.get("run_tag")
        if only_run_tag is not None and run_tag != only_run_tag:
            continue
        if exclude_run_tag is not None and run_tag == exclude_run_tag:
            continue
        filtered.append(item)
    return filtered


def _resolve_inventory_active_version(
    entries: list[dict],
    active_version: Optional[str],
    *,
    scope_active_version: bool = False,
) -> Optional[str]:
    active_version = str(active_version or "").strip() or None
    if not scope_active_version or active_version is None:
        return active_version
    listed_versions = {item["version"] for item in entries}
    return active_version if active_version in listed_versions else None


def build_version_inventory(
    versions_root: Path,
    *,
    base_name: str,
    active_version: Optional[str] = None,
    keep_latest_stable: int = 2,
    delete_candidates: bool = True,
    only_run_tag: Optional[str] = None,
    exclude_run_tag: Optional[str] = None,
    scope_active_version: bool = False,
) -> dict:
    entries = _filter_inventory_entries(
        iter_skill_versions(versions_root, base_name=base_name),
        only_run_tag=only_run_tag,
        exclude_run_tag=exclude_run_tag,
    )
    stable_entries = [item for item in entries if not item["is_candidate"]]
    candidate_entries = [item for item in entries if item["is_candidate"]]
    cleanup_targets = select_cleanup_targets(
        versions_root,
        base_name=base_name,
        keep_latest_stable=keep_latest_stable,
        delete_candidates=delete_candidates,
        only_run_tag=only_run_tag,
        exclude_run_tag=exclude_run_tag,
    )
    resolved_active_version = _resolve_inventory_active_version(
        entries,
        active_version,
        scope_active_version=scope_active_version,
    )
    return {
        "base_name": base_name,
        "skills_root": str(versions_root),
        "active_version": resolved_active_version,
        "source_dir": str(versions_root / base_name),
        "source_exists": (versions_root / base_name).exists(),
        "version_count": len(entries),
        "stable_count": len(stable_entries),
        "candidate_count": len(candidate_entries),
        "stable_versions": [item["version"] for item in stable_entries],
        "candidate_versions": [item["version"] for item in candidate_entries],
        "versions": [
            {
                "version": item["version"],
                "path": str(item["path"]),
                "is_candidate": item["is_candidate"],
                "run_tag": item.get("run_tag"),
                "semantic_version": item.get("semantic_version"),
                "has_snapshot_manifest": item["has_snapshot_manifest"],
            }
            for item in entries
        ],
        "cleanup_preview": {
            "keep_latest_stable": max(0, int(keep_latest_stable)),
            "delete_candidates": bool(delete_candidates),
            "only_run_tag": str(only_run_tag or "").strip() or None,
            "exclude_run_tag": str(exclude_run_tag or "").strip() or None,
            "targets": [str(path) for path in cleanup_targets],
        },
    }

def _iter_snapshot_files(root_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue
        if SNAPSHOT_MANIFEST_NAME in path.parts:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        files.append(path)
    return sorted(files)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_snapshot_manifest(root_dir: Path) -> dict:
    return {
        "files": [
            {
                "path": path.relative_to(root_dir).as_posix(),
                "sha256": _hash_file(path),
                "size": path.stat().st_size,
            }
            for path in _iter_snapshot_files(root_dir)
        ]
    }


def _manifest_path(root_dir: Path) -> Path:
    return root_dir / SNAPSHOT_MANIFEST_NAME


def _load_snapshot_manifest(root_dir: Path) -> Optional[dict]:
    manifest_path = _manifest_path(root_dir)
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _write_snapshot_manifest(root_dir: Path, manifest: dict) -> None:
    _manifest_path(root_dir).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def ensure_version_snapshot(source_dir: Path, target_dir: Path, *, refresh: bool = False) -> Path:
    if not source_dir.exists():
        raise FileNotFoundError(f"Source skill directory does not exist: {source_dir}")
    if source_dir.resolve() == target_dir.resolve():
        source_manifest = _build_snapshot_manifest(source_dir)
        target_manifest = _load_snapshot_manifest(target_dir)
        if target_manifest != source_manifest:
            _write_snapshot_manifest(target_dir, source_manifest)
        return target_dir
    source_manifest = _build_snapshot_manifest(source_dir)
    if target_dir.exists():
        target_manifest = _load_snapshot_manifest(target_dir)
        if target_manifest == source_manifest:
            return target_dir
        if not refresh:
            raise FileExistsError(
                "Version snapshot already exists but does not match the current source directory: "
                f"{target_dir}. Delete it or rerun with refresh enabled."
            )
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    _write_snapshot_manifest(target_dir, source_manifest)
    return target_dir


def publish_candidate_to_stable(candidate_dir: Path, stable_dir: Path) -> Path:
    if stable_dir.exists():
        raise FileExistsError(f"Stable version already exists: {stable_dir}")
    shutil.copytree(candidate_dir, stable_dir, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return stable_dir



