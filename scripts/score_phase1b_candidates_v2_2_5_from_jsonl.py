from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_feature_rows(pattern: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"no feature files matched: {pattern}")

    rows = []
    for file in files:
        for row in read_jsonl(Path(file)):
            row["_feature_file"] = file
            rows.append(row)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phase1b_smoke/classifier_smoke_model_v2_2_5.joblib")
    ap.add_argument("--feature-glob", default="data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized/*.jsonl")
    ap.add_argument("--out", default="data/retrieval/phase1b/v2_2_5/candidate_score_snapshot_v2_2_5.jsonl")
    ap.add_argument("--summary-out", default="audit/phase_1b/candidate_score_snapshot_v2_2_5_summary.json")
    ap.add_argument("--top-csv-out", default="audit/phase_1b/candidate_score_snapshot_v2_2_5_top_by_campaign.csv")
    args = ap.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    model_id = bundle.get("model_id", "classifier_smoke_v2_2_5")

    feature_rows = load_feature_rows(args.feature_glob)

    X = []
    for row in feature_rows:
        features = row.get("features", {})
        if not isinstance(features, dict):
            features = {}
        X.append([features.get(col) for col in feature_columns])

    X_np = np.array(X, dtype=float)
    scores = model.predict_proba(X_np)[:, 1]

    by_campaign: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for i, (row, score) in enumerate(zip(feature_rows, scores)):
        by_campaign[str(row.get("campaign_id", ""))].append((i, float(score)))

    rank_info: dict[int, dict[str, Any]] = {}
    for campaign_id, items in by_campaign.items():
        ranked = sorted(items, key=lambda x: x[1], reverse=True)
        n = len(ranked)
        for rank, (idx, score) in enumerate(ranked, start=1):
            # campaign 내부 descending percentile. rank 1 = 1.0
            percentile = 1.0 - ((rank - 1) / max(n, 1))
            rank_info[idx] = {
                "campaign_candidate_count": n,
                "campaign_rank_desc": rank,
                "campaign_score_percentile_desc": float(percentile),
            }

    out_rows = []
    now = utc_now()

    for idx, (row, score) in enumerate(zip(feature_rows, scores), start=1):
        campaign_id = str(row.get("campaign_id", ""))
        image_id = str(row.get("image_id", ""))
        layout_spec_id = str(row.get("layout_spec_id", "layout_default"))
        feature_snapshot_id = str(row.get("feature_snapshot_id", ""))

        info = rank_info[idx - 1]

        out_rows.append({
            "candidate_score_snapshot_id": f"score_v2_2_5_{idx:05d}",
            "score_snapshot_version": "v2_2_5",
            "created_at": now,
            "campaign_id": campaign_id,
            "image_id": image_id,
            "pair_id": row.get("pair_id"),
            "layout_spec_id": layout_spec_id,
            "feature_snapshot_id": feature_snapshot_id,
            "feature_snapshot_version": row.get("feature_snapshot_version", "v2_2_5_duplicate_canonicalized"),
            "previous_feature_snapshot_id": row.get("previous_feature_snapshot_id"),
            "canonical_image_id": row.get("canonical_image_id"),
            "duplicate_group_id": row.get("duplicate_group_id"),
            "duplicate_canonicalization_status": row.get("duplicate_canonicalization_status"),
            "model_id": model_id,
            "model_path": args.model,
            "model_family": "logistic_regression_smoke",
            "scores": {
                "diagnostic_accept_score": float(score),
                "campaign_candidate_count": int(info["campaign_candidate_count"]),
                "campaign_rank_desc": int(info["campaign_rank_desc"]),
                "campaign_score_percentile_desc": float(info["campaign_score_percentile_desc"]),
            },
            "score_status": "diagnostic_only",
            "threshold_status": "no_calibrated_threshold",
            "interpretation": "Diagnostic candidate score only. Not a calibrated accept/reject threshold.",
        })

    write_jsonl(Path(args.out), out_rows)

    summary_by_campaign = []
    top_rows = []

    for campaign_id, items in sorted(by_campaign.items()):
        vals = sorted([score for _, score in items])
        summary_by_campaign.append({
            "campaign_id": campaign_id,
            "count": len(vals),
            "min": float(vals[0]),
            "median": float(vals[len(vals) // 2]),
            "max": float(vals[-1]),
        })

        ranked = sorted(items, key=lambda x: x[1], reverse=True)[:10]
        for rank, (idx, score) in enumerate(ranked, start=1):
            r = out_rows[idx]
            top_rows.append({
                "campaign_id": campaign_id,
                "rank": rank,
                "image_id": r["image_id"],
                "duplicate_group_id": r.get("duplicate_group_id", ""),
                "canonical_image_id": r.get("canonical_image_id", ""),
                "diagnostic_accept_score": score,
                "campaign_score_percentile_desc": r["scores"]["campaign_score_percentile_desc"],
                "feature_snapshot_id": r["feature_snapshot_id"],
            })

    summary = {
        "event": "done",
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
        "model": args.model,
        "model_id": model_id,
        "feature_glob": args.feature_glob,
        "rows": len(out_rows),
        "campaign_count": len(by_campaign),
        "score_snapshot_out": args.out,
        "summary_by_campaign": summary_by_campaign,
        "feature_columns": feature_columns,
        "non_claims": [
            "diagnostic_accept_score는 calibrated probability가 아니다.",
            "campaign 간 score를 직접 비교하지 않는다.",
            "pass/fail threshold로 사용하지 않는다.",
        ],
    }

    write_json(Path(args.summary_out), summary)
    pd.DataFrame(top_rows).to_csv(args.top_csv_out, index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
