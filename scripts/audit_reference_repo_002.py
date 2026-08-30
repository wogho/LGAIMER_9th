#!/usr/bin/env python3
"""File-by-file inventory and rule/provenance audit of the local reference repo."""
from __future__ import annotations

import ast
import hashlib
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "github_reference" / "1번 레포"
OUT = ROOT / "model" / "REF-REPO-AUDIT-002"
KEYWORDS = re.compile(r"998|외부|KBO|API|test\.csv|train\.csv|shift|offset|seed|license|라이선스|규정|독립|groupby|rolling|평균", re.I)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def text_summary(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return {"text_read": False}
    lines = text.splitlines()
    hits = [{"line": i + 1, "text": line[:300]} for i, line in enumerate(lines) if KEYWORDS.search(line)][:40]
    item = {"text_read": True, "lines": len(lines), "keyword_hits": hits}
    if path.suffix == ".py":
        try:
            tree = ast.parse(text)
            imports = sorted({node.names[0].name for node in ast.walk(tree) if isinstance(node, ast.Import) and node.names})
            imports += sorted({node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module})
            calls = sorted({node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id for node in ast.walk(tree) if isinstance(node, ast.Call) and (isinstance(node.func, ast.Attribute) or isinstance(node.func, ast.Name))})
            item.update({"ast_parse": True, "imports": sorted(set(imports)), "calls": calls})
        except SyntaxError as exc:
            item.update({"ast_parse": False, "syntax_error": str(exc)})
    if path.suffix == ".ipynb":
        try:
            notebook = json.loads(text)
            cells = notebook.get("cells", [])
            code = ["".join(c.get("source", [])) for c in cells if c.get("cell_type") == "code"]
            item.update({"notebook_cells": len(cells), "code_cells": len(code), "code_keyword_hits": sum(bool(KEYWORDS.search(c)) for c in code)})
        except json.JSONDecodeError as exc:
            item.update({"notebook_parse": False, "parse_error": str(exc)})
    return item


def main() -> None:
    files = []
    for path in sorted(p for p in REPO.rglob("*") if p.is_file()):
        rel = path.relative_to(REPO).as_posix()
        item = {"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path), "suffix": path.suffix.lower()}
        if path.suffix.lower() in {".py", ".md", ".txt", ".html", ".yml", ".json", ".ipynb"}:
            item.update(text_summary(path))
        if path.suffix.lower() in {".zip", ".gz"}:
            item["compressed"] = True
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                item["members"] = sorted(zf.namelist())
        files.append(item)
    result = {
        "audit_id": "REF-REPO-AUDIT-002",
        "repo": str(REPO),
        "file_count": len(files),
        "total_bytes": sum(x["bytes"] for x in files),
        "license_files": [x["path"] for x in files if Path(x["path"]).name.lower() in {"license", "license.md", "copying"}],
        "files": files,
        "provenance_policy": "reference artifacts are analyzed only; no reference code/weights are copied into submission",
        "status": "INVENTORY_COMPLETE",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "inventory.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("audit_id", "file_count", "total_bytes", "license_files", "status")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
