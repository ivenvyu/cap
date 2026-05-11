from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

SCORE_STATUS = "diagnostic_only"
THRESHOLD_STATUS = "no_calibrated_threshold"
DEFAULT_MODEL_ID = "classifier_smoke_v2_2_2"


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            fid = row.get("feature_snapshot_id")
            if not isinstance(fid, str) or not fid:
                raise RuntimeError(f"feature row missing feature_snapshot_id: {path}")
            if fid in seen:
                raise RuntimeError(f"duplicate feature_snapshot_id: {fid}")
            seen.add(fid)
            if not isinstance(row.get("features"), dict):
                raise RuntimeError(f"feature row missing features object: {fid}")
            rows.append(row)
    if not rows:
        raise RuntimeError("no feature rows loaded")
    return rows


def coerce_feature_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in feature_cols:
        s = df[col] if col in df.columns else pd.Series([np.nan] * len(df), index=df.index)
        if s.dtype == bool:
            out[col] = s.astype(float)
            continue
        if s.dtype == object:
            lowered = s.astype(str).str.strip().str.lower()
            bool_map = {
                "true": 1.0,
                "false": 0.0,
                "yes": 1.0,
                "no": 0.0,
                "none": np.nan,
                "null": np.nan,
                "nan": np.nan,
                "": np.nan,
            }
            mapped = lowered.map(bool_map)
            numeric = pd.to_numeric(s, errors="coerce")
            out[col] = numeric.where(numeric.notna(), mapped)
        else:
            out[col] = pd.to_numeric(s, errors="coerce")
    return out


def build_score_frame(feature_rows: list[dict[str, Any]], feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta_rows: list[dict[str, Any]] = []
    for row in feature_rows:
        features = row["features"]
        flat: dict[str, Any] = {
            "feature_snapshot_id": row["feature_snapshot_id"],
            "snapshot_version": row.get("snapshot_version"),
            "pair_id": row.get("pair_id"),
            "campaign_id": row.get("campaign_id"),
            "image_id": row.get("image_id"),
            "layout_spec_id": row.get("layout_spec_id"),
            "preview_renderer_version": row.get("preview_renderer_version"),
        }
        for col in feature_cols:
            flat[col] = features.get(col)
        meta_rows.append(flat)

    df = pd.DataFrame(meta_rows)
    required = ["feature_snapshot_id", "pair_id", "campaign_id", "image_id"]
    missing_required = [c for c in required if c not in df.columns or df[c].isna().any()]
    if missing_required:
        raise RuntimeError(f"score frame missing required metadata columns: {missing_required}")
    X = coerce_feature_frame(df, feature_cols)
    return df, X


def attach_campaign_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["campaign_rank_desc"] = (
        out.groupby("campaign_id")["diagnostic_accept_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    out["campaign_candidate_count"] = out.groupby("campaign_id")["pair_id"].transform("count").astype(int)
    out["campaign_score_percentile_desc"] = 1.0 - (
        (out["campaign_rank_desc"] - 1) / out["campaign_candidate_count"].clip(lower=1)
    )
    return out.sort_values(["campaign_id", "campaign_rank_desc", "pair_id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/phase1b_smoke/classifier_smoke_model_v2_2_2.joblib")
    ap.add_argument("--feature-snapshots-glob", default="data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl")
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--out", default="data/retrieval/phase1b/v2_2_2/candidate_score_snapshot_v2_2_2.jsonl")
    ap.add_argument("--summary-out", default="audit/phase_1b/candidate_score_snapshot_v2_2_2_summary.json")
    ap.add_argument("--top-csv-out", default="audit/phase_1b/candidate_score_snapshot_v2_2_2_top_by_campaign.csv")
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise RuntimeError(f"model file not found: {model_path}")
    bundle = joblib.load(model_path)
    pipe = bundle.get("pipeline")
    feature_cols = bundle.get("feature_columns")
    if pipe is None or not isinstance(feature_cols, list) or not feature_cols:
        raise RuntimeError(f"model bundle must contain pipeline and feature_columns: {model_path}")

    feature_paths = sorted(Path().glob(args.feature_snapshots_glob))
    if not feature_paths:
        raise RuntimeError(f"no feature snapshot files matched: {args.feature_snapshots_glob}")

    feature_rows = load_feature_rows(feature_paths)
    meta, X = build_score_frame(feature_rows, feature_cols)
    prob = pipe.predict_proba(X)[:, 1]
    meta["diagnostic_accept_score"] = prob.astype(float)
    meta = attach_campaign_ranks(meta)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for _, row in meta.iterrows():
        record = {
            "candidate_score_snapshot_id": f"score_v2_2_2__{args.model_id}__{row['feature_snapshot_id']}",
            "created_at": created_at,
            "score_snapshot_version": "v2_2_2",
            "model_id": args.model_id,
            "model_family": bundle.get("model_family", "unknown"),
            "model_path": str(model_path),
            "score_status": SCORE_STATUS,
            "threshold_status": THRESHOLD_STATUS,
            "interpretation": "Diagnostic score only. Use for ranking/audit/disagreement sampling, not calibrated pass/fail.",
            "feature_snapshot_id": row["feature_snapshot_id"],
            "feature_snapshot_version": row.get("snapshot_version"),
            "pair_id": row["pair_id"],
            "campaign_id": row["campaign_id"],
            "image_id": row["image_id"],
            "layout_spec_id": row.get("layout_spec_id"),
            "preview_renderer_version": row.get("preview_renderer_version"),
            "scores": {
                "diagnostic_accept_score": float(row["diagnostic_accept_score"]),
                "campaign_rank_desc": int(row["campaign_rank_desc"]),
                "campaign_candidate_count": int(row["campaign_candidate_count"]),
                "campaign_score_percentile_desc": float(row["campaign_score_percentile_desc"]),
            },
        }
        records.append(record)

    out_path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records), encoding="utf-8")

    top_k = max(1, int(args.top_k))
    top = meta[meta["campaign_rank_desc"] <= top_k][
        [
            "campaign_id",
            "campaign_rank_desc",
            "diagnostic_accept_score",
            "campaign_score_percentile_desc",
            "pair_id",
            "image_id",
            "layout_spec_id",
            "feature_snapshot_id",
        ]
    ].copy()
    top_csv = Path(args.top_csv_out)
    top_csv.parent.mkdir(parents=True, exist_ok=True)
    top.to_csv(top_csv, index=False)

    score_summary = meta.groupby("campaign_id")["diagnostic_accept_score"].agg(["count", "min", "median", "max"]).reset_index()
    summary = {
        "event": "done",
        "score_status": SCORE_STATUS,
        "threshold_status": THRESHOLD_STATUS,
        "model": str(model_path),
        "model_id": args.model_id,
        "feature_snapshots_glob": args.feature_snapshots_glob,
        "feature_snapshot_files": [str(p) for p in feature_paths],
        "rows": int(len(records)),
        "campaign_count": int(meta["campaign_id"].nunique()),
        "score_snapshot_out": str(out_path),
        "top_csv_out": str(top_csv),
        "feature_columns": feature_cols,
        "score_summary_by_campaign": [
            {
                "campaign_id": str(r["campaign_id"]),
                "count": int(r["count"]),
                "min": float(r["min"]),
                "median": float(r["median"]),
                "max": float(r["max"]),
            }
            for _, r in score_summary.iterrows()
        ],
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(jdump(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
