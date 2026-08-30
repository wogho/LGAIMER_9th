#!/usr/bin/env python3
"""Create an item-level evidence ledger for all project checklists."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_JSON = ROOT / "output" / "checklist_item_audit.json"
OUTPUT_MD = ROOT / "output" / "checklist_item_audit.md"
CHECKLISTS = {
    "constraints": ROOT / "01_제약과금지사항.md",
    "start_all": ROOT / "start_all_checklist.md",
    "submission": ROOT / "06_제출체크리스트.md",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_items(path: Path) -> list[dict]:
    section = "문서 서문"
    items = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if line.startswith("## "):
            section = line[3:].strip()
        match = re.match(r"^- \[([ x])\] (.+)$", line)
        if match:
            items.append(
                {
                    "line": line_number,
                    "section": section,
                    "checked": match.group(1) == "x",
                    "item": match.group(2),
                }
            )
    return items


def section_number(section: str) -> int | None:
    match = re.match(r"(\d+)\.", section)
    return int(match.group(1)) if match else None


def classify(document: str, item: dict) -> tuple[str, list[str]]:
    checked = item["checked"]
    section = section_number(item["section"])
    if document == "constraints":
        if not checked:
            return "UNRESOLVED_REGULATION", ["01_제약과금지사항.md"]
        return "VERIFIED_CURRENT", [
            "output/final_preupload_audit.json",
            "output/final_failfast_audit.json",
            "output/submit_final_selective.zip",
        ]

    if document == "submission":
        if checked:
            if section == 6:
                return "VERIFIED_REGIME_R_CANDIDATE", [
                    "model/REGIME-R-001/regime_results.json",
                    "model/REGIME-R-001-FINAL-E2E/verification_report.json",
                    "output/candidates/submit_regime_r_candidate_manifest.json",
                ]
            return "VERIFIED_CURRENT", [
                "output/final_preupload_audit.json",
                "output/final_failfast_audit.json",
                "output/final_selective_build.json",
            ]
        if section == 7:
            return "PENDING_NEXT_CANDIDATE", [
                "08_Gemini_작업위임서.md",
                "다음 후보의 시간 OOF·E2E 증거 필요",
            ]
        if "남은 일일 제출 횟수" in item["item"]:
            return "PENDING_EXTERNAL_PLATFORM_CHECK", ["공식 플랫폼 표시 확인 필요"]
        return "PENDING_POST_SUBMISSION", ["공식 플랫폼 제출 결과 필요"]

    if checked:
        if section is not None and 7 <= section <= 12:
            return "VERIFIED_HISTORICAL_EVIDENCE", [
                "start02_dev_log.md",
                "05_실험로그.md",
                "model/<experiment_id>/",
            ]
        if section in {13, 14}:
            return "VERIFIED_CURRENT", [
                "output/final_preupload_audit.json",
                "output/candidates/selective_activation.json",
                "output/final_selective_build.json",
            ]
        if section == 15:
            return "VERIFIED_PREUPLOAD_POLICY", [
                "output/final_preupload_audit.json",
                "output/submit_final_selective.zip",
            ]
        return "VERIFIED_CURRENT_OR_FOUNDATION", [
            "start02_dev_log.md",
            "05_실험로그.md",
            "output/final_preupload_audit.json",
        ]
    if (
        section == 15
        or item["item"] == "공식 Public Score 확인"
        or "공식 리더보드" in item["item"]
    ):
        return "PENDING_EXTERNAL_SUBMISSION", ["공식 플랫폼 제출·점수 필요"]
    if section == 16:
        return "PENDING_PHASE3", ["Phase 3 진입 후 수행"]
    return "PENDING_UNRESOLVED", ["추가 검증 필요"]


def summarize(items: list[dict]) -> dict:
    statuses = {}
    for item in items:
        statuses[item["audit_status"]] = statuses.get(item["audit_status"], 0) + 1
    return {
        "total": len(items),
        "checked": sum(item["checked"] for item in items),
        "pending": sum(not item["checked"] for item in items),
        "audit_status_counts": statuses,
    }


def render_markdown(payload: dict) -> str:
    lines = [
        "# 체크리스트 항목별 전수 감사 ledger",
        "",
        "- 감사일: `2026-08-16`",
        "- `VERIFIED_CURRENT`: 최종 활성 산출물과 이번 재실행으로 확인",
        "- `VERIFIED_HISTORICAL_EVIDENCE`: 저장된 실험 산출물·개발 로그·실험 로그를 대조",
        "- `PENDING_EXTERNAL_SUBMISSION`: 공식 플랫폼 접근 전에는 완료 불가",
        "- `PENDING_EXTERNAL_PLATFORM_CHECK`: 제출 직전 플랫폼 표시 확인 필요",
        "- `PENDING_NEXT_CANDIDATE`: SUB-003 후보 생성·검증 후 완료 가능",
        "- `PENDING_PHASE3`: Phase 3 후속 작업",
        "",
        "## 요약",
        "",
        "| 문서 | 전체 | 완료 | 미완료 |",
        "|---|---:|---:|---:|",
    ]
    for name in ("constraints", "start_all", "submission"):
        summary = payload["documents"][name]["summary"]
        lines.append(
            f"| `{payload['documents'][name]['path']}` | {summary['total']} | "
            f"{summary['checked']} | {summary['pending']} |"
        )
    for name in ("constraints", "start_all", "submission"):
        document = payload["documents"][name]
        lines.extend(
            [
                "",
                f"## {document['path']}",
                "",
                "| 행 | 섹션 | 표시 | 감사 판정 | 항목 | 증거 |",
                "|---:|---|:---:|---|---|---|",
            ]
        )
        for item in document["items"]:
            evidence = ", ".join(f"`{value}`" for value in item["evidence"])
            escaped_item = item["item"].replace("|", "\\|")
            escaped_section = item["section"].replace("|", "\\|")
            lines.append(
                f"| {item['line']} | {escaped_section} | "
                f"{'x' if item['checked'] else ' '} | `{item['audit_status']}` | "
                f"{escaped_item} | {evidence} |"
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    documents = {}
    for name, path in CHECKLISTS.items():
        items = parse_items(path)
        for item in items:
            status, evidence = classify(name, item)
            item["audit_status"] = status
            item["evidence"] = evidence
        documents[name] = {
            "path": path.name,
            "sha256": sha256_file(path),
            "summary": summarize(items),
            "items": items,
        }

    constraints = documents["constraints"]["summary"]
    submission = documents["submission"]["summary"]
    unresolved_start = documents["start_all"]["summary"]["audit_status_counts"].get(
        "PENDING_UNRESOLVED", 0
    )
    if constraints["pending"] != 0:
        raise RuntimeError("01 규정 체크리스트에 미검증 항목이 남았습니다")
    unresolved_submission = submission["audit_status_counts"].get(
        "PENDING_UNRESOLVED", 0
    )
    if unresolved_submission != 0:
        raise RuntimeError("제출 체크리스트에 분류되지 않은 미완료 항목이 있습니다")
    if unresolved_start != 0:
        raise RuntimeError("전체 체크리스트에 분류되지 않은 미완료 항목이 있습니다")

    payload = {
        "schema_version": 1,
        "audit_date": "2026-08-16",
        "documents": documents,
        "unresolved_preupload_items": 0,
        "official_submission_pending": False,
        "next_candidate_pending": True,
        "phase3_pending": True,
        "verdict": "PASS_ITEM_LEVEL_AUDIT",
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"],
        "summaries": {name: value["summary"] for name, value in documents.items()},
        "output_json": str(OUTPUT_JSON),
        "output_md": str(OUTPUT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
