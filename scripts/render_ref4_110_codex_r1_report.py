#!/usr/bin/env python3
"""Render the 110 R1 Markdown from result.json and refresh its manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-110-CODEX-R1"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    lines = [
        "# REF4-110-CODEX-R1", "",
        f"- status: `{result['status']}`",
        f"- leaf candidates: `{result['actual_leaf_count']}`",
        f"- OOF rows: `{result['oof_rows']}`",
        f"- anchor-transfer-to-109C: `{str(result['anchor_transfer_to_109c_verified']).lower()}`", "",
        "| candidate_name | candidate_status | 2022_delta | 2023_delta | 2024_delta | 2024_BSS | 2024_local_CV | time_weighted_delta | worst_season_delta | gate_result |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in result["candidates"]:
        delta = item["delta_brier_vs_strict_base"]
        metrics = item["metrics"]["2024"]
        gates = item["gate_results"]
        gate_text = ";".join(f"{key}={str(value).lower()}" for key, value in gates.items())
        lines.append(
            f"| {item['candidate_name']} | {item['candidate_status']} | "
            f"{delta['2022']:+.15g} | {delta['2023']:+.15g} | {delta['2024']:+.15g} | "
            f"{metrics['bss']:+.15g} | {metrics['local_score']:+.15g} | "
            f"{item['time_weighted_delta']:+.15g} | {item['worst_season_delta']:+.15g} | {gate_text} |"
        )
    lines.extend(["", f"- provisional winner: `{result['provisional_winner']}`", "- official score estimated: `false`", "- test/full-train/ZIP: `false/false/false`"])
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = OUT / "audit_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative = str(Path(__file__).resolve().relative_to(ROOT))
    manifest["artifacts"][relative] = {"sha256": sha256_path(Path(__file__).resolve()), "size": Path(__file__).stat().st_size}
    validator = ROOT / "scripts/verify_ref4_110_codex_r1.py"
    if validator.is_file():
        validator_relative = str(validator.relative_to(ROOT))
        manifest["artifacts"][validator_relative] = {"sha256": sha256_path(validator), "size": validator.stat().st_size}
    report_relative = str((OUT / "result.md").relative_to(ROOT))
    manifest["artifacts"][report_relative] = {"sha256": sha256_path(OUT / "result.md"), "size": (OUT / "result.md").stat().st_size}
    manifest["artifact_count"] = len(manifest["artifacts"])
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_count": len(result["candidates"]), "markdown_rows": len(result["candidates"]), "artifact_count": manifest["artifact_count"]}, indent=2))


if __name__ == "__main__":
    main()
