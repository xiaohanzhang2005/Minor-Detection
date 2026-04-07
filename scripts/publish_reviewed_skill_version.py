from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

from src.config import set_active_skill_version
from src.skill_loop.versioning import publish_candidate_to_stable
from src.utils.path_utils import normalize_project_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a reviewed candidate skill version to a stable version and set it active. Contract-check indicators are advisory and are not enforced here."
    )
    parser.add_argument("--base-version", required=True)
    parser.add_argument("--candidate-version", required=True)
    parser.add_argument("--stable-version", required=True)
    args = parser.parse_args()

    candidate_dir = ROOT_DIR / "skills" / args.candidate_version
    stable_dir = ROOT_DIR / "skills" / args.stable_version
    review_path = candidate_dir / "review" / f"review_decision_vs_{args.base_version}.json"
    if not review_path.exists():
        raise FileNotFoundError(f"Missing review decision file: {review_path}")
    review_payload = json.loads(review_path.read_text(encoding="utf-8"))
    if str(review_payload.get("decision", "")).strip().lower() != "approve":
        raise ValueError("Candidate review decision is not approve")

    publish_candidate_to_stable(candidate_dir, stable_dir)
    set_active_skill_version(args.stable_version)
    payload = {
        "success": True,
        "base_version": args.base_version,
        "candidate_version": args.candidate_version,
        "published_stable_version": args.stable_version,
    }
    print(json.dumps(normalize_project_paths(payload, project_root=ROOT_DIR, start=ROOT_DIR), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
