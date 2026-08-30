#!/usr/bin/env python3
"""REF-AUX-LABEL-001: official-train-only hidden auxiliary label audit.

The reconstruction uses only cumulative ``asof_pitcher_*`` rates and their
official denominators. It never reads test.csv or any external source.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "model" / "REF-AUX-LABEL-001"
TRAIN = ROOT / "data" / "train.csv"
LABELS = ("success", "middle", "reverse", "ball", "strike", "fastball", "breaking", "offspeed")
MIX_LABELS = {"fastball", "breaking", "offspeed"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def recover(train: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    row_order = np.argsort(train["row_id"].to_numpy(), kind="stable")
    ordered = train.iloc[row_order].reset_index(drop=True)
    pid = ordered["pitcher_id"].to_numpy()
    first = np.r_[True, pid[1:] != pid[:-1]]
    last = np.r_[pid[1:] != pid[:-1], True]

    values: dict[str, np.ndarray] = {}
    valid_counts: dict[str, int] = {}
    for label in LABELS:
        denominator = "asof_pitcher_pitchmix_n" if label in MIX_LABELS else "asof_pitcher_n"
        rate = ordered[f"asof_pitcher_{label}_rate"].to_numpy(dtype=np.float64)
        denom = ordered[denominator].to_numpy(dtype=np.float64)
        usable = (~first) & np.isfinite(rate) & np.isfinite(denom) & (denom > 0)
        if not np.all(usable | first | last):
            raise AssertionError(f"{label}: non-first/non-last cumulative value is missing")
        cumulative = np.rint(rate * denom)
        cumulative[first] = 0.0
        event = np.full(len(ordered), np.nan, dtype=np.float64)
        event[:-1] = cumulative[1:] - cumulative[:-1]
        event[first | last] = np.nan
        values[label] = event
        valid_counts[label] = int(np.isfinite(event).sum())

    recovered_ordered = pd.DataFrame(values)
    recovered_ordered.insert(0, "row_id", ordered["row_id"].to_numpy())
    recovered = recovered_ordered.set_index("row_id").reindex(train["row_id"]).reset_index()
    return recovered, valid_counts


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    required = ["row_id", "pitcher_id", "control_success", "asof_pitcher_n", "asof_pitcher_pitchmix_n"]
    required += [f"asof_pitcher_{x}_rate" for x in LABELS]
    train = pd.read_csv(TRAIN, usecols=required, encoding="utf-8-sig")

    if len(train) != 1_475_092 or not train["row_id"].is_unique:
        raise AssertionError("train row count or row_id uniqueness failed")
    recovered, valid_counts = recover(train)
    valid = recovered[list(LABELS)].notna().all(axis=1)

    success_match = bool((recovered.loc[valid, "success"].to_numpy() == train.loc[valid, "control_success"].to_numpy()).all())
    binary_ok = bool(all(recovered.loc[valid, label].isin([0.0, 1.0]).all() for label in LABELS))
    mix_sum = recovered.loc[valid, ["fastball", "breaking", "offspeed"]].sum(axis=1)
    mix_ok = bool(np.allclose(mix_sum.to_numpy(), 1.0, atol=0.0, rtol=0.0))
    if not success_match or not binary_ok or not mix_ok:
        raise AssertionError(f"reconstruction validation failed: success={success_match}, binary={binary_ok}, mix={mix_ok}")

    out_path = EXP / "recovered_labels.csv.gz"
    recovered.to_csv(out_path, index=False, compression="gzip", float_format="%.0f")
    report = {
        "experiment_id": "REF-AUX-LABEL-001",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"path": str(TRAIN.relative_to(ROOT)), "sha256": sha256(TRAIN), "rows": len(train)},
        "output": {"path": str(out_path.relative_to(ROOT)), "sha256": sha256(out_path), "rows": len(recovered)},
        "input_scope": "official train.csv only; test.csv and external data not read",
        "reconstruction": {"labels": list(LABELS), "valid_rows_all_labels": int(valid.sum()), "valid_counts": valid_counts},
        "checks": {
            "train_row_count": len(train) == 1_475_092,
            "row_id_unique": bool(train["row_id"].is_unique),
            "success_matches_control_success": success_match,
            "all_recovered_labels_binary": binary_ok,
            "pitchmix_sum_exactly_one": mix_ok,
            "last_or_first_pitcher_rows_excluded": int((~valid).sum()) > 0,
        },
        "status": "PASS",
        "next_step": "REF-AUX-OFFSET-001 only after independent review",
    }
    report_path = EXP / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    attestation = {
        "attestation_id": "REF-AUX-LABEL-001-ATTESTATION",
        "source_sha256": sha256(TRAIN),
        "labels_sha256": sha256(out_path),
        "validation_report_sha256": sha256(report_path),
        "validator_sha256": sha256(Path(__file__)),
        "rows": len(train),
        "valid_rows_all_labels": int(valid.sum()),
        "success_match": success_match,
        "binary_ok": binary_ok,
        "pitchmix_ok": mix_ok,
        "status": "REF_AUX_LABEL_AUDIT_VERIFIED",
    }
    att_path = EXP / "attestation.json"
    att_path.write_text(json.dumps(attestation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(attestation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
