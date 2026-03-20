"""
导出正式 Skill 使用的 retrieval assets。

将当前项目中的检索索引和检索语料复制到指定 Skill 目录下的
assets/retrieval_assets/，用于后续内置 RAG。
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import RETRIEVAL_DB_DIR, RETRIEVAL_CORPUS_DIR, SKILLS_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导出 retrieval assets 到正式 Skill 目录。")
    parser.add_argument(
        "--skill-dir",
        default=str(SKILLS_DIR / "minor-detection"),
        help="目标 Skill 目录。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    skill_dir = Path(args.skill_dir)
    assets_dir = skill_dir / "assets" / "retrieval_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    index_src = RETRIEVAL_DB_DIR / "index.pkl"
    manifest_src = RETRIEVAL_DB_DIR / "manifest.json"
    corpus_src = RETRIEVAL_CORPUS_DIR / "cases_v1.jsonl"

    required = [index_src, manifest_src, corpus_src]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"缺少检索资产，无法导出: {missing}")

    shutil.copy2(index_src, assets_dir / "index.pkl")
    shutil.copy2(corpus_src, assets_dir / "corpus.jsonl")

    with open(manifest_src, "r", encoding="utf-8") as f:
        source_manifest = json.load(f)

    exported_manifest = {
        "exported_at": datetime.now().isoformat(),
        "source_manifest": source_manifest,
        "files": {
            "index": "index.pkl",
            "corpus": "corpus.jsonl",
            "manifest": "manifest.json",
        },
        "sample_count": source_manifest.get("sample_count"),
        "embedding_model": source_manifest.get("embedding_model"),
        "dataset_hash": source_manifest.get("dataset_hash"),
    }

    with open(assets_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(exported_manifest, f, ensure_ascii=False, indent=2)

    print(f"[OK] exported retrieval assets -> {assets_dir}")


if __name__ == "__main__":
    main()
