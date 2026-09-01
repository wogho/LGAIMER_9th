#!/usr/bin/env python3
"""Preflight for the literal 110A/110B/110C rebuild (R2)."""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-110-ORIGINAL-R2"
TRAIN = ROOT / "data/train.csv"
BASE = ROOT / "model/REF4-OOF-DIAG-034A"
ROLLBACK = ROOT / "output/submit_ref4_super_ensemble_109C.zip"
LB = ROOT / "output/lb_record_109.json"
TRAIN_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
ROLLBACK_SHA = "03e874949ae7172af0dab16d9c3f52de94d5ac9256e571ddedd377b785d9634f"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    failures: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            failures.append(name)

    required = [
        TRAIN, ROOT / "data/trackman_history.csv", ROLLBACK, LB,
        ROOT / "start02_dev_log.md", ROOT / "start03_reference.md",
        ROOT / "01_제약과금지사항.md", ROOT / "08_Gemini_작업위임서.md",
        BASE / "diagnostic_rows.csv", BASE / "audit_manifest.json",
        BASE / "validation_report.json", BASE / "audit_attestation.json",
        ROOT / "scripts/run_ref4_exact_oof_031a.py",
        ROOT / "scripts/build_ref4_deep_hierarchical_102a.py",
        ROOT / "scripts/build_ref4_deep_hierarchical_103a.py",
        ROOT / "scripts/build_ref4_super_ensemble_107a.py",
        ROOT / "scripts/build_ref4_super_ensemble_108c.py",
        ROOT / "scripts/build_ref4_super_ensemble_109c.py",
    ]
    manifests = [
        ROOT / "model/REF4-DEEP-HIERARCHICAL-103A/production_package/model/manifest.json",
        ROOT / "model/REF4-SUPER-ENSEMBLE-107A/production_package/model/manifest.json",
        ROOT / "model/REF4-SUPER-ENSEMBLE-108C/production_package/model/manifest.json",
        ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package/model/manifest.json",
    ]
    required.extend(manifests)
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())
    if failures:
        report = {"experiment_id": OUT.name, "status": "BLOCKED", "checked_count": len(checks),
                  "passed_count": sum(bool(x["passed"]) for x in checks), "mismatch_count": len(failures),
                  "failures": failures, "checks": checks}
        write_json(OUT / "preflight_report.json", report)
        raise SystemExit(1)

    check("train_sha256", sha256(TRAIN) == TRAIN_SHA, sha256(TRAIN))
    train = pd.read_csv(TRAIN, low_memory=False)
    target = pd.to_numeric(train["control_success"], errors="coerce").to_numpy(float)
    check("train_rows", len(train) == 1_475_092, len(train))
    check("train_row_id_unique", train["row_id"].is_unique, int(train["row_id"].nunique()))
    check("target_finite_binary", np.isfinite(target).all() and set(np.unique(target)) == {0.0, 1.0}, np.unique(target).tolist())
    expected_rows = {2019: 237413, 2020: 244087, 2021: 247088, 2022: 247472, 2023: 245525, 2024: 253507}
    actual_rows = train.groupby("season", sort=True).size().astype(int).to_dict()
    check("season_rows", actual_rows == expected_rows, actual_rows)

    rb_hash = sha256(ROLLBACK)
    check("rollback_sha256", rb_hash == ROLLBACK_SHA, rb_hash)
    with zipfile.ZipFile(ROLLBACK) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    check("rollback_integrity", bad is None, bad)
    check("rollback_layout", {"script.py", "requirements.txt"}.issubset(names) and any(x.startswith("model/") for x in names), len(names))
    lb = json.loads(LB.read_text(encoding="utf-8"))
    check("rollback_lb_binding", lb.get("version") == "109C" and lb.get("zip_name") == ROLLBACK.name and math.isclose(float(lb["score"]), 1120.8914462094, abs_tol=1e-12), lb)

    att = json.loads((BASE / "audit_attestation.json").read_text(encoding="utf-8"))
    check("base_oof_attested", att.get("status") == "AUDIT_VERIFIED" and att.get("mismatch_count") == 0, att)
    check("base_manifest_binding", sha256(BASE / "audit_manifest.json") == att.get("manifest_sha256"), sha256(BASE / "audit_manifest.json"))
    check("base_report_binding", sha256(BASE / "validation_report.json") == att.get("validation_report_sha256"), sha256(BASE / "validation_report.json"))
    diag = pd.read_csv(BASE / "diagnostic_rows.csv", dtype={"row_id": str})
    check("base_oof_rows_unique", len(diag) == 746_504 and diag["row_id"].is_unique, {"rows": len(diag), "unique": int(diag["row_id"].nunique())})
    check("base_oof_seasons", sorted(diag["season"].unique().tolist()) == [2022, 2023, 2024], sorted(diag["season"].unique().tolist()))
    p = diag["prediction_current"].to_numpy(float)
    check("base_oof_prediction_valid", np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), {"min": float(p.min()), "max": float(p.max())})
    expected = train.loc[train["season"].isin([2022, 2023, 2024]), ["row_id", "season", "control_success"]].copy()
    expected["row_id"] = expected["row_id"].astype(str)
    paired = diag[["row_id", "season", "target"]].merge(expected, on="row_id", how="outer", validate="one_to_one", indicator=True, suffixes=("_oof", "_raw"))
    paired_ok = (len(paired) == len(diag) and paired["_merge"].eq("both").all()
                 and paired["season_oof"].eq(paired["season_raw"]).all()
                 and np.array_equal(paired["target"].to_numpy(float), paired["control_success"].to_numpy(float)))
    check("base_oof_row_target_pairing", paired_ok, {"rows": len(paired), "merge": paired["_merge"].value_counts().to_dict()})

    versions = []
    for path, expected_version in zip(manifests, [
        "REF4-DEEP-HIERARCHICAL-103A", "REF4-SUPER-ENSEMBLE-107A",
        "REF4-SUPER-ENSEMBLE-108C", "REF4-SUPER-ENSEMBLE-109C",
    ]):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        versions.append(manifest.get("version"))
        check(f"manifest_version:{expected_version}", manifest.get("version") == expected_version, manifest.get("version"))

    exact_oof_outputs = [OUT / "p103a_oof.csv", OUT / "expert_oof.csv"]
    check("exact_oof_not_preexisting", not any(path.exists() for path in exact_oof_outputs), [str(x.relative_to(ROOT)) for x in exact_oof_outputs if x.exists()])
    free = shutil.disk_usage(ROOT).free
    check("disk_free_at_least_8gib", free >= 8 * 1024**3, free)

    candidates = {
        "110A": {"id": "REF4-TEMPORAL-CROSSFIT-MOE-110A-R2", "single_change": "three real 107A/108C/109C expert predictions plus a fixed 3-seed shallow residual router"},
        "110B": {"id": "REF4-CROSSFIT-SUPER-110B-R2", "single_change": "109C residual target uses strict p103A OOF anchor instead of full-train p103A"},
        "110C": {"id": "REF4-HYPER-RELIABILITY-GRAND-SUPER-110C-R2", "single_change": "109C plus fixed reliability shrinkage and handedness EB partial pooling"},
    }
    check("candidate_leaf_count", len(candidates) == 3, list(candidates))

    status = "AUDIT_VERIFIED" if not failures else "AUDIT_INCOMPLETE"
    contract = {
        "experiment_id": OUT.name, "status": "LOCKED_BEFORE_RESULTS",
        "official_champion": {"version": "109C", "score": float(lb["score"]), "evidence": "USER_REPORTED_OFFICIAL_LEADERBOARD_RESULT", "zip": str(ROLLBACK.relative_to(ROOT)), "sha256": rb_hash},
        "candidates": candidates, "candidate_count": len(candidates),
        "strict_protocol": {
            "expert_oof": {"2022": [2019, 2020, 2021], "2023": [2019, 2020, 2021, 2022], "2024": [2019, 2020, 2021, 2022, 2023]},
            "router": {"2023": [2022], "2024": [2022, 2023], "2025": [2022, 2023, 2024]},
            "validation_labels_used_in_fit": False,
        },
        "fixed_parameters": {
            "router_seeds": [42, 1, 2], "regular_residual_seeds": [42, 1, 2, 3, 4], "futures_residual_seeds": [42, 1, 2, 3],
            "regular_residual_weight": 0.085, "futures_residual_weight": 0.035,
            "reliability_k": 25.0, "handedness_alpha_pitcher": 100.0, "handedness_alpha_context": 300.0,
            "recent_season_weights": {"2022": 0.20, "2023": 0.30, "2024": 0.50},
        },
        "promotion_gate": {"delta_2024_max": -0.0001, "delta_2022_max": 0.00005, "time_weighted_delta_max_exclusive": 0.0, "worst_season_delta_max": 0.00005, "bootstrap_2024_ci_high_max_exclusive": 0.0, "row_independence_atol": 1e-12, "runtime_seconds_max": 600.0},
        "forbidden_inputs": ["candidate/p_103_full_train.npy", "model/REF4-110-CODEX-R1/oof_predictions.csv", "output/multiseason_benchmark_110.json"],
        "output_rule": "No test inference, full fit, or ZIP unless one leaf passes every performance and technical gate after independent audit.",
    }
    write_json(OUT / "audit_contract.json", contract)
    report = {"experiment_id": OUT.name, "status": status, "checked_count": len(checks),
              "passed_count": sum(bool(x["passed"]) for x in checks), "mismatch_count": len(failures),
              "failures": failures, "checks": checks, "candidate_count": len(candidates),
              "train_sha256": sha256(TRAIN), "rollback_sha256": rb_hash, "free_bytes": free}
    write_json(OUT / "preflight_report.json", report)
    (OUT / "preflight_report.md").write_text(
        f"# {OUT.name} preflight\n\n- status: `{status}`\n- checked: `{len(checks)}`\n- mismatches: `{len(failures)}`\n- candidates: `{len(candidates)}`\n- rollback: `{rb_hash}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "checked_count": len(checks), "passed_count": report["passed_count"], "mismatch_count": len(failures), "candidate_count": len(candidates), "free_bytes": free}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
