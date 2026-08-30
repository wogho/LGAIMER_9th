#!/usr/bin/env python3
"""Record a user-supplied official leaderboard result with local ZIP evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True)
    parser.add_argument("--score", required=True, type=float)
    parser.add_argument("--previous-score", required=True, type=float)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    zip_path = (ROOT / args.zip).resolve()
    actual_sha = sha256(zip_path)
    if actual_sha != args.expected_sha256:
        raise SystemExit(f"ZIP hash mismatch: {actual_sha}")
    result = {
        "evidence_class": "USER_REPORTED_OFFICIAL_LEADERBOARD_RESULT",
        "record_date": date.today().isoformat(),
        "submission_time": None,
        "submission_time_status": "NOT_SUPPLIED",
        "zip_path": str(zip_path.relative_to(ROOT)),
        "zip_sha256": actual_sha,
        "zip_size": zip_path.stat().st_size,
        "official_score": args.score,
        "previous_official_score": args.previous_score,
        "score_delta": args.score - args.previous_score,
        "target_1100_gap": 1100.0 - args.score,
        "target_1126_4544_gap": 1126.4544 - args.score,
        "score_at_least_1100": args.score >= 1100.0,
        "promotion_to_champion": args.score > args.previous_score,
        "leaderboard_derived_tuning_authorized": False,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
