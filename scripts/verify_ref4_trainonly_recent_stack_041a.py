#!/usr/bin/env python3
"""Independent source recomputation and attestation for 041A."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-TRAINONLY-RECENT-STACK-041A"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_x(d: pd.DataFrame, r: dict[str, float], m: dict[str, object]) -> np.ndarray:
    f = d.game_type.eq("F").to_numpy()
    p0 = np.where(f, d.p_v2_global + r["v2_scale"] * (d.p_v2_f - d.p_v2_global), d.p_v2_global)
    p1 = np.where(f, d.p_v3_55_global + r["v355_scale"] * (d.p_v3_55_f - d.p_v3_55_global), d.p_v3_55_global)
    recent = d.p_v3_30_global + r["v330_recent_inner_scale"] * (d.p_v3_30_f_recent - d.p_v3_30_global)
    f30 = r["v330_all_weight"] * d.p_v3_30_f_all + (1 - r["v330_all_weight"]) * recent
    p2 = np.where(f, d.p_v3_30_global + r["v330_scale"] * (f30 - d.p_v3_30_global), d.p_v3_30_global)
    main = np.average(np.vstack([p0, p1, p2]), axis=0, weights=m["main_weights"])
    q = []
    for name in ("middle", "wild", "reverse"):
        g = d[f"risk_{name}_global"].to_numpy(float); z = d[f"risk_{name}_f"].to_numpy(float)
        q.append(np.where(f, g + r["subtype_scale"] * (z - g), g))
    return np.column_stack([main, *q])


def fit(x: np.ndarray, y: np.ndarray, z: np.ndarray, alpha: float) -> np.ndarray:
    s = StandardScaler(); xt = s.fit_transform(x)
    model = Ridge(alpha=alpha, fit_intercept=True); model.fit(xt, y)
    return np.clip(model.predict(s.transform(z)), 1e-5, 1 - 1e-5)


def cluster(y: np.ndarray, b: np.ndarray, c: np.ndarray, groups: np.ndarray, reps: int, seed: int) -> tuple[float, float, float, int]:
    gain = (b - y) ** 2 - (c - y) ** 2
    codes, u = pd.factorize(groups, sort=True)
    sums = np.bincount(codes, weights=gain, minlength=len(u)); counts = np.bincount(codes, minlength=len(u))
    rng = np.random.default_rng(seed); draws = np.empty(reps)
    for i in range(reps):
        pick = rng.integers(0, len(u), len(u)); draws[i] = sums[pick].sum() / counts[pick].sum()
    return float(gain.mean()), float(np.quantile(draws, .025)), float(np.quantile(draws, .975)), int(len(u))


def main() -> None:
    contract = json.loads((OUT / "audit_contract.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    regime = json.loads((ROOT / "model/REF4-CHAMPION-STACK-030/f_regime_meta.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((ROOT / "model/REF4-CHAMPION-STACK-030/manifest.json").read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []
    def check(name: str, passed: bool, actual: object, expected: object | None = None) -> None:
        checks.append({"name": name, "checked": True, "pass": bool(passed), "actual": actual, "expected": expected})

    hash_mismatches = []
    for rel, rec in manifest["artifacts"].items():
        p = ROOT / rel; actual = sha256(p) if p.is_file() else None
        if actual != rec["sha256"] or (p.stat().st_size if p.is_file() else None) != rec["size"]:
            hash_mismatches.append(rel)
    check("manifest_artifacts", not hash_mismatches, hash_mismatches, [])
    check("manifest_artifact_count", manifest["artifact_count"] == len(manifest["artifacts"]), len(manifest["artifacts"]), manifest["artifact_count"])

    frames = {int(y): pd.read_csv(ROOT / p) for y, p in contract["source_folds"].items()}
    xs = {y: build_x(d, regime, source_manifest) for y, d in frames.items()}
    saved = pd.read_csv(OUT / "oof_predictions.csv")
    all_parts = []
    alpha = float(contract["fixed_estimator"]["alpha"])
    recomputed: dict[str, object] = {}
    gate_values = {}
    for protocol in contract["temporal_protocol"]:
        fy, vy = int(protocol["fit_season"]), int(protocol["validation_season"])
        d = frames[vy]; y = d.target.to_numpy(float); b = d.prediction.to_numpy(float)
        c = fit(xs[fy], frames[fy].target.to_numpy(float), xs[vy], alpha)
        part = saved.loc[saved.season.eq(vy)]
        check(f"row_id_{vy}", part.row_id.tolist() == d.row_id.tolist(), len(part), len(d))
        prediction_diff = float(np.max(np.abs(part.candidate_prediction.to_numpy(float) - c)))
        check(f"prediction_csv_roundtrip_{vy}", prediction_diff <= 1e-15, prediction_diff, 1e-15)
        gain = float(np.mean((b-y)**2) - np.mean((c-y)**2))
        bss_b = 1 - np.mean((b-y)**2) / np.mean((y-y.mean())**2); bss_c = 1 - np.mean((c-y)**2) / np.mean((y-y.mean())**2)
        ci = cluster(y, b, c, d.pitcher_id.to_numpy(), int(contract["promotion_gate"]["cluster_bootstrap_repetitions"]), int(contract["promotion_gate"][f"cluster_bootstrap_seed_{vy}"]))
        recorded = result["folds"][str(vy)]
        check(f"gain_{vy}", abs(gain - recorded["brier_gain"]) <= 1e-15, gain, recorded["brier_gain"])
        check(f"bss_gain_{vy}", abs((bss_c-bss_b) - recorded["bss_gain"]) <= 1e-15, bss_c-bss_b, recorded["bss_gain"])
        check(f"ci_{vy}", max(abs(ci[0]-recorded["cluster_ci"]["brier_gain"]), abs(ci[1]-recorded["cluster_ci"]["ci_low"]), abs(ci[2]-recorded["cluster_ci"]["ci_high"])) <= 1e-15, list(ci[:3]), [recorded["cluster_ci"][k] for k in ("brier_gain", "ci_low", "ci_high")])
        recomputed[str(vy)] = {"gain": gain, "bss_gain": bss_c-bss_b, "ci_low": ci[1], "ci_high": ci[2], "clusters": ci[3]}
        gate_values[f"{vy}_brier_gain"] = gain >= float(contract["promotion_gate"][f"brier_gain_{vy}_min"])
        gate_values[f"{vy}_cluster_ci_low"] = ci[1] > 0
        all_parts.append((y, b, c))
    ya = np.concatenate([p[0] for p in all_parts]); ba = np.concatenate([p[1] for p in all_parts]); ca = np.concatenate([p[2] for p in all_parts])
    pooled_gain = float(np.mean((ba-ya)**2) - np.mean((ca-ya)**2))
    gate_values["pooled_brier_gain"] = pooled_gain >= float(contract["promotion_gate"]["pooled_brier_gain_min"])
    gate_values["worst_season_bss_gain"] = min(v["bss_gain"] for v in recomputed.values()) > 0
    expected_gate = {
        "2023_brier_gain": gate_values["2023_brier_gain"], "2024_brier_gain": gate_values["2024_brier_gain"],
        "pooled_brier_gain": gate_values["pooled_brier_gain"], "worst_season_bss_gain": gate_values["worst_season_bss_gain"],
        "2023_cluster_ci_low": gate_values["2023_cluster_ci_low"], "2024_cluster_ci_low": gate_values["2024_cluster_ci_low"],
    }
    check("pooled_gain", abs(pooled_gain - result["pooled"]["brier_gain"]) <= 1e-15, pooled_gain, result["pooled"]["brier_gain"])
    check("gate_fields", result["gate_checks"] == expected_gate, result["gate_checks"], expected_gate)
    check("gate_count", result["gate_checks_count"] == len(expected_gate) == manifest["gate_count"], [result["gate_checks_count"], len(expected_gate), manifest["gate_count"]])
    expected_pass = all(expected_gate.values())
    check("promotion_pass", result["promotion_pass"] is expected_pass, result["promotion_pass"], expected_pass)
    check("leaf_count", result["actual_leaf_count"] == manifest["leaf_count"] == 1, [result["actual_leaf_count"], manifest["leaf_count"]], 1)
    check("production_stack_presence", (OUT / "production_stack.json").exists() is expected_pass, (OUT / "production_stack.json").exists(), expected_pass)
    check("no_test_or_zip", result["test_read"] is False and result["test_inference_performed"] is False and result["candidate_bundle_created"] is False and result["zip_created"] is False, {k: result[k] for k in ("test_read", "test_inference_performed", "candidate_bundle_created", "zip_created")})
    md = (OUT / "result.md").read_text(encoding="utf-8")
    check("markdown_identity", result["candidate_name"] in md and result["candidate_status"] in md and f"`{str(expected_pass).lower()}`" in md, result["candidate_name"])

    failures = [c["name"] for c in checks if not c["pass"]]
    report = {"experiment_id": contract["experiment_id"], "status": "AUDIT_VERIFIED" if not failures else "AUDIT_FAIL_REPORT", "checked_count": len(checks), "pass_count": len(checks)-len(failures), "fail_count": len(failures), "mismatch_count": len(failures), "failures": failures, "checks": checks, "recomputed": recomputed, "pooled_brier_gain": pooled_gain, "expected_gate": expected_gate, "promotion_pass": expected_pass}
    report_path = OUT / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=lambda value: value.item() if hasattr(value, "item") else str(value)) + "\n", encoding="utf-8")
    attestation = {"experiment_id": contract["experiment_id"], "overall_status": report["status"], "performance_status": "PASS" if expected_pass else "FAIL", "candidate_count": 1, "leaf_count": 1, "gate_count": len(expected_gate), "checked_count": len(checks), "pass_count": len(checks)-len(failures), "fail_count": len(failures), "mismatch_count": len(failures), "audit_manifest_sha256": sha256(OUT / "audit_manifest.json"), "validation_report_sha256": sha256(report_path), "validator_sha256": sha256(Path(__file__)), "result_sha256": sha256(OUT / "result.json"), "oof_predictions_sha256": sha256(OUT / "oof_predictions.csv"), "preserved_zip_sha256": sha256(ROOT / contract["preserve_zip"]), "promotion_pass": expected_pass, "test_read": False, "test_inference_performed": False, "candidate_bundle_created": False, "zip_created": False}
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: attestation[k] for k in ("overall_status", "performance_status", "checked_count", "pass_count", "fail_count", "mismatch_count", "promotion_pass")}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
