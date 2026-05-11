from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

QUEUE_VERSION = "review_queue_v2_2_2"
QUEUE_STAGE = "diagnostic_model_available"
SCORE_STATUS = "diagnostic_only"
THRESHOLD_STATUS = "no_calibrated_threshold"

DEFAULT_BUCKET_BUDGETS = {
    "model_top_diagnostic": 6,
    "clip_high_model_low": 5,
    "dinov2_high_model_low": 5,
    "model_high_clip_negative_high": 4,
    "cluster_diversity": 5,
    "layout_safe_coverage": 3,
    "random_control": 2,
}

BUCKET_NOTES = {
    "model_top_diagnostic": "Campaign-local top diagnostic model scores. Scores are not calibrated pass/fail probabilities.",
    "clip_high_model_low": "High CLIP rank but low diagnostic model rank within the same campaign; disagreement discovery bucket.",
    "dinov2_high_model_low": "High DINOv2 campaign/family visual-anchor rank but low diagnostic model rank within the same campaign; missed-positive discovery bucket.",
    "model_high_clip_negative_high": "High diagnostic model rank and high CLIP-negative rank within the same campaign; hard-negative / semantic-conflict audit bucket.",
    "cluster_diversity": "One strong candidate per DINOv2 mid cluster where possible; coverage bucket.",
    "layout_safe_coverage": "High campaign-local layout-safety rank; layout compatibility coverage bucket.",
    "random_control": "Deterministic random control bucket for blind-spot monitoring.",
    "fill_remaining_diagnostic_mixed": "Fills remaining rows after duplicate/image dedupe using a campaign-local mixed diagnostic rank. Not a final quality claim.",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return v


def load_feature_map(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            fid = row.get("feature_snapshot_id")
            if not isinstance(fid, str) or not fid:
                raise RuntimeError(f"feature row missing feature_snapshot_id: {path}")
            if fid in out:
                raise RuntimeError(f"duplicate feature_snapshot_id: {fid}")
            if not isinstance(row.get("features"), dict):
                raise RuntimeError(f"feature row missing features object: {fid}")
            out[fid] = row
    if not out:
        raise RuntimeError("no feature rows loaded")
    return out


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    return {str(r.get("image_id")): r for r in rows if r.get("image_id") is not None}


def flatten_score_row(score: dict[str, Any], feature_row: dict[str, Any], manifest: dict[str, dict[str, Any]]) -> dict[str, Any]:
    features = feature_row["features"]
    image_id = str(score.get("image_id") or feature_row.get("image_id"))
    m = manifest.get(image_id, {})
    scores = score.get("scores") or {}

    flat: dict[str, Any] = {
        "queue_stage": QUEUE_STAGE,
        "campaign_id": score.get("campaign_id") or feature_row.get("campaign_id"),
        "pair_id": score.get("pair_id") or feature_row.get("pair_id"),
        "image_id": image_id,
        "duplicate_group_id": feature_row.get("duplicate_group_id") or image_id,
        "feature_snapshot_id": score.get("feature_snapshot_id") or feature_row.get("feature_snapshot_id"),
        "candidate_score_snapshot_id": score.get("candidate_score_snapshot_id"),
        "layout_spec_id": score.get("layout_spec_id") or feature_row.get("layout_spec_id"),
        "preview_renderer_version": score.get("preview_renderer_version") or feature_row.get("preview_renderer_version"),
        "model_id": score.get("model_id"),
        "score_status": score.get("score_status", SCORE_STATUS),
        "threshold_status": score.get("threshold_status", THRESHOLD_STATUS),
        "diagnostic_model_score": finite_or_none(scores.get("diagnostic_accept_score")),
        "campaign_model_rank_desc": scores.get("campaign_rank_desc"),
        "campaign_score_percentile_desc": finite_or_none(scores.get("campaign_score_percentile_desc")),
        "campaign_candidate_count": scores.get("campaign_candidate_count"),
        "image_path": m.get("path"),
        "resolved_path": m.get("resolved_path"),
        "category": m.get("category"),
        "source_group": m.get("source_group"),
        "place_name": m.get("place_name"),
        "subject_name": m.get("subject_name"),
        "preview_path": None,
        "decision": None,
        "issue_tags": None,
        "preference_rank": None,
        "notes": None,
    }

    selected_features = [
        "clip_margin",
        "clip_positive_max_sim",
        "clip_positive_mean_sim",
        "clip_negative_max_sim",
        "clip_negative_mean_sim",
        "clip_rank_percentile",
        "dinov2_campaign_pos_nn_sim",
        "dinov2_campaign_neg_nn_sim",
        "dinov2_campaign_margin",
        "dinov2_campaign_pos_count",
        "dinov2_campaign_neg_count",
        "dinov2_family_pos_nn_sim",
        "dinov2_family_neg_nn_sim",
        "dinov2_family_margin",
        "dinov2_family_support_count",
        "dinov2_cluster_id_coarse",
        "dinov2_cluster_id_mid",
        "dinov2_cluster_id_fine",
        "dinov2_cluster_positive_rate_coarse",
        "dinov2_cluster_positive_rate_mid",
        "dinov2_cluster_positive_rate_fine",
        "dinov2_cluster_review_count_coarse",
        "dinov2_cluster_review_count_mid",
        "dinov2_cluster_review_count_fine",
        "dinov2_duplicate_group_seen",
        "required_region_safe_mean",
        "required_region_safe_min",
        "title_region_safe_score",
        "info_region_safe_score",
        "edge_density",
        "contrast",
        "brightness",
        "saturation",
        "image_category_gallery",
        "image_category_tree",
        "image_category_flower",
        "path_has_architecture",
        "path_has_garden",
    ]
    for key in selected_features:
        flat[key] = features.get(key)
    return flat


def dense_rank_desc(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    # Missing values are placed at the bottom for descending ranks.
    sentinel = -np.inf
    return numeric.fillna(sentinel).rank(method="first", ascending=False).astype(int)


def add_campaign_ranks(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["model_rank_desc"] = out.groupby("campaign_id")["diagnostic_model_score"].transform(dense_rank_desc)
    out["clip_rank_desc"] = out.groupby("campaign_id")["clip_margin"].transform(dense_rank_desc)
    out["clip_negative_rank_desc"] = out.groupby("campaign_id")["clip_negative_max_sim"].transform(dense_rank_desc)
    out["dinov2_campaign_margin_rank_desc"] = out.groupby("campaign_id")["dinov2_campaign_margin"].transform(dense_rank_desc)
    out["dinov2_family_margin_rank_desc"] = out.groupby("campaign_id")["dinov2_family_margin"].transform(dense_rank_desc)
    out["dinov2_anchor_rank_desc"] = out[["dinov2_campaign_margin_rank_desc", "dinov2_family_margin_rank_desc"]].min(axis=1).astype(int)
    out["layout_safe_rank_desc"] = out.groupby("campaign_id")["required_region_safe_min"].transform(dense_rank_desc)
    out["model_clip_disagreement_rank_gap"] = out["model_rank_desc"] - out["clip_rank_desc"]
    out["model_dinov2_disagreement_rank_gap"] = out["model_rank_desc"] - out["dinov2_anchor_rank_desc"]
    out["mixed_rank_key"] = (
        out["model_rank_desc"].astype(float)
        + out["clip_rank_desc"].astype(float)
        + out["dinov2_anchor_rank_desc"].astype(float)
        + out["layout_safe_rank_desc"].astype(float)
    )
    return out


def add_rows(
    selected: list[dict[str, Any]],
    candidates: pd.DataFrame,
    bucket_name: str,
    budget: int,
    seen_images: set[str],
    seen_dups: set[str],
    bucket_audit: list[dict[str, Any]],
) -> None:
    requested = max(0, int(budget))
    considered = int(len(candidates))
    added = 0
    skipped_duplicate_or_image = 0

    for _, row in candidates.iterrows():
        if added >= requested:
            break
        image_id = str(row.get("image_id"))
        dup = str(row.get("duplicate_group_id") or image_id)
        if image_id in seen_images or dup in seen_dups:
            skipped_duplicate_or_image += 1
            continue
        out = row.to_dict()
        out["source_bucket"] = bucket_name
        out["bucket_semantics_actual"] = BUCKET_NOTES.get(bucket_name, bucket_name)
        selected.append(out)
        seen_images.add(image_id)
        seen_dups.add(dup)
        added += 1

    bucket_audit.append(
        {
            "bucket": bucket_name,
            "requested_budget": requested,
            "candidate_rows_considered": considered,
            "rows_added": added,
            "skipped_duplicate_or_image": skipped_duplicate_or_image,
            "score_status": SCORE_STATUS,
            "threshold_status": THRESHOLD_STATUS,
            "note": BUCKET_NOTES.get(bucket_name),
        }
    )


def best_per_cluster(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rows: list[pd.Series] = []
    cluster_col = "dinov2_cluster_id_mid"
    for _, g in df.groupby(cluster_col, dropna=False):
        gg = g.sort_values(["model_rank_desc", "clip_rank_desc", "pair_id"], ascending=[True, True, True])
        rows.append(gg.iloc[0])
    if not rows:
        return df.iloc[0:0]
    return pd.DataFrame(rows).sort_values(["model_rank_desc", "clip_rank_desc", "pair_id"], ascending=[True, True, True])


def build_queue_for_campaign(cdf: pd.DataFrame, target_per_campaign: int, seed: int) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    seen_images: set[str] = set()
    seen_dups: set[str] = set()
    bucket_audit: list[dict[str, Any]] = []

    budgets = dict(DEFAULT_BUCKET_BUDGETS)
    if target_per_campaign != sum(budgets.values()):
        # Scale by order-preserving truncation/fill rather than changing the meaning of each bucket.
        # Any remainder is handled by fill_remaining_diagnostic_mixed.
        pass

    add_rows(
        selected,
        cdf.sort_values(["model_rank_desc", "pair_id"], ascending=[True, True]),
        "model_top_diagnostic",
        min(budgets["model_top_diagnostic"], target_per_campaign - len(selected)),
        seen_images,
        seen_dups,
        bucket_audit,
    )

    add_rows(
        selected,
        cdf.sort_values(["model_clip_disagreement_rank_gap", "clip_rank_desc", "pair_id"], ascending=[False, True, True]),
        "clip_high_model_low",
        min(budgets["clip_high_model_low"], target_per_campaign - len(selected)),
        seen_images,
        seen_dups,
        bucket_audit,
    )

    add_rows(
        selected,
        cdf.sort_values(["model_dinov2_disagreement_rank_gap", "dinov2_anchor_rank_desc", "pair_id"], ascending=[False, True, True]),
        "dinov2_high_model_low",
        min(budgets["dinov2_high_model_low"], target_per_campaign - len(selected)),
        seen_images,
        seen_dups,
        bucket_audit,
    )

    add_rows(
        selected,
        cdf.sort_values(["model_rank_desc", "clip_negative_rank_desc", "pair_id"], ascending=[True, True, True]),
        "model_high_clip_negative_high",
        min(budgets["model_high_clip_negative_high"], target_per_campaign - len(selected)),
        seen_images,
        seen_dups,
        bucket_audit,
    )

    add_rows(
        selected,
        best_per_cluster(cdf),
        "cluster_diversity",
        min(budgets["cluster_diversity"], target_per_campaign - len(selected)),
        seen_images,
        seen_dups,
        bucket_audit,
    )

    add_rows(
        selected,
        cdf.sort_values(["layout_safe_rank_desc", "model_rank_desc", "pair_id"], ascending=[True, True, True]),
        "layout_safe_coverage",
        min(budgets["layout_safe_coverage"], target_per_campaign - len(selected)),
        seen_images,
        seen_dups,
        bucket_audit,
    )

    rng = random.Random(seed + abs(hash(str(cdf["campaign_id"].iloc[0]))) % 1_000_000)
    random_order = list(cdf.index)
    rng.shuffle(random_order)
    add_rows(
        selected,
        cdf.loc[random_order],
        "random_control",
        min(budgets["random_control"], target_per_campaign - len(selected)),
        seen_images,
        seen_dups,
        bucket_audit,
    )

    remaining = max(0, target_per_campaign - len(selected))
    if remaining:
        add_rows(
            selected,
            cdf.sort_values(["mixed_rank_key", "pair_id"], ascending=[True, True]),
            "fill_remaining_diagnostic_mixed",
            remaining,
            seen_images,
            seen_dups,
            bucket_audit,
        )

    out = pd.DataFrame(selected)
    return out, bucket_audit


def preferred_columns(df: pd.DataFrame) -> list[str]:
    cols = [
        "queue_id",
        "queue_row_id",
        "queue_version",
        "queue_stage",
        "source_bucket",
        "bucket_semantics_actual",
        "campaign_id",
        "pair_id",
        "image_id",
        "duplicate_group_id",
        "feature_snapshot_id",
        "candidate_score_snapshot_id",
        "layout_spec_id",
        "preview_renderer_version",
        "model_id",
        "score_status",
        "threshold_status",
        "diagnostic_model_score",
        "campaign_model_rank_desc",
        "campaign_score_percentile_desc",
        "model_rank_desc",
        "clip_rank_desc",
        "dinov2_anchor_rank_desc",
        "layout_safe_rank_desc",
        "model_clip_disagreement_rank_gap",
        "model_dinov2_disagreement_rank_gap",
        "image_path",
        "resolved_path",
        "category",
        "source_group",
        "place_name",
        "subject_name",
        "clip_margin",
        "clip_positive_max_sim",
        "clip_negative_max_sim",
        "dinov2_campaign_margin",
        "dinov2_family_margin",
        "dinov2_cluster_id_coarse",
        "dinov2_cluster_id_mid",
        "dinov2_cluster_id_fine",
        "required_region_safe_mean",
        "required_region_safe_min",
        "title_region_safe_score",
        "info_region_safe_score",
        "preview_path",
        "decision",
        "issue_tags",
        "preference_rank",
        "notes",
    ]
    return [c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--score-snapshot", default="data/retrieval/phase1b/v2_2_2/candidate_score_snapshot_v2_2_2.jsonl")
    ap.add_argument("--feature-snapshots-glob", default="data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl")
    ap.add_argument("--manifest", default="data/ontology/raw_image_manifest_v2_2_1.jsonl")
    ap.add_argument("--out-dir", default="data/review/phase1b/v2_2_2/queues")
    ap.add_argument("--audit-out", default="audit/phase_1b/review_queue_v2_2_2_summary.json")
    ap.add_argument("--combined-out", default="data/review/phase1b/v2_2_2/review_queue_v2_2_2_all_campaigns.csv")
    ap.add_argument("--queue-id-prefix", default="review_queue_v2_2_2")
    ap.add_argument("--target-per-campaign", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260511)
    args = ap.parse_args()

    score_path = Path(args.score_snapshot)
    if not score_path.exists():
        raise RuntimeError(f"score snapshot not found: {score_path}")
    scores = read_jsonl(score_path)
    if not scores:
        raise RuntimeError(f"score snapshot is empty: {score_path}")

    feature_paths = sorted(Path().glob(args.feature_snapshots_glob))
    if not feature_paths:
        raise RuntimeError(f"no feature snapshot files matched: {args.feature_snapshots_glob}")
    features_by_id = load_feature_map(feature_paths)
    manifest = load_manifest(Path(args.manifest))

    flat_rows: list[dict[str, Any]] = []
    missing_feature_ids: list[str] = []
    for score in scores:
        fid = score.get("feature_snapshot_id")
        feature_row = features_by_id.get(str(fid))
        if feature_row is None:
            missing_feature_ids.append(str(fid))
            continue
        flat_rows.append(flatten_score_row(score, feature_row, manifest))
    if missing_feature_ids:
        raise RuntimeError(f"score rows reference missing feature_snapshot_id values: {missing_feature_ids[:5]}")
    if not flat_rows:
        raise RuntimeError("no rows available after score-feature join")

    df = add_campaign_ranks(pd.DataFrame(flat_rows))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = Path(args.combined_out)
    combined_path.parent.mkdir(parents=True, exist_ok=True)

    all_queues: list[pd.DataFrame] = []
    campaign_summaries: list[dict[str, Any]] = []
    global_bucket_counts: Counter[str] = Counter()

    for campaign_id, cdf in sorted(df.groupby("campaign_id"), key=lambda kv: str(kv[0])):
        queue_id = f"{args.queue_id_prefix}__{campaign_id}"
        qdf, bucket_audit = build_queue_for_campaign(cdf.copy(), args.target_per_campaign, args.seed)
        qdf = qdf.copy()
        qdf["queue_id"] = queue_id
        qdf["queue_version"] = QUEUE_VERSION
        qdf["queue_stage"] = QUEUE_STAGE
        qdf = qdf.reset_index(drop=True)
        qdf["queue_row_id"] = [f"{queue_id}_{i:04d}" for i in range(1, len(qdf) + 1)]
        qdf = qdf[preferred_columns(qdf)]

        out_path = out_dir / f"review_queue_v2_2_2__{campaign_id}.csv"
        qdf.to_csv(out_path, index=False)
        all_queues.append(qdf)
        counts = qdf["source_bucket"].value_counts().to_dict() if not qdf.empty else {}
        global_bucket_counts.update(counts)
        campaign_summaries.append(
            {
                "campaign_id": str(campaign_id),
                "queue_id": queue_id,
                "out": str(out_path),
                "candidate_rows_available": int(len(cdf)),
                "rows_selected": int(len(qdf)),
                "unique_images": int(qdf["image_id"].nunique()) if not qdf.empty else 0,
                "unique_duplicate_groups": int(qdf["duplicate_group_id"].nunique()) if not qdf.empty else 0,
                "source_bucket_counts": {str(k): int(v) for k, v in counts.items()},
                "bucket_audit": bucket_audit,
                "score_summary": {
                    "min": float(pd.to_numeric(cdf["diagnostic_model_score"], errors="coerce").min()),
                    "median": float(pd.to_numeric(cdf["diagnostic_model_score"], errors="coerce").median()),
                    "max": float(pd.to_numeric(cdf["diagnostic_model_score"], errors="coerce").max()),
                },
            }
        )

    combined = pd.concat(all_queues, ignore_index=True) if all_queues else pd.DataFrame()
    if not combined.empty:
        combined = combined[preferred_columns(combined)]
    combined.to_csv(combined_path, index=False)

    summary = {
        "event": "done",
        "queue_version": QUEUE_VERSION,
        "queue_stage": QUEUE_STAGE,
        "score_status": SCORE_STATUS,
        "threshold_status": THRESHOLD_STATUS,
        "interpretation": (
            "Review queue uses campaign-local ranks and bucket allocations. "
            "Diagnostic model scores are not calibrated pass/fail probabilities and are not compared globally across campaigns."
        ),
        "score_snapshot": str(score_path),
        "feature_snapshots_glob": args.feature_snapshots_glob,
        "feature_snapshot_files": [str(p) for p in feature_paths],
        "out_dir": str(out_dir),
        "combined_out": str(combined_path),
        "target_per_campaign": int(args.target_per_campaign),
        "campaign_count": int(df["campaign_id"].nunique()),
        "input_rows": int(len(df)),
        "selected_rows": int(len(combined)),
        "global_source_bucket_counts": {str(k): int(v) for k, v in global_bucket_counts.items()},
        "campaigns": campaign_summaries,
        "bucket_definitions": BUCKET_NOTES,
        "policy_note": (
            "Bucket budgets are review-sampling allocations. They are not aesthetic thresholds, final quality thresholds, "
            "or calibrated acceptance rules."
        ),
    }
    audit_path = Path(args.audit_out)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(jdump(summary) + "\n", encoding="utf-8")
    print(jdump(summary))


if __name__ == "__main__":
    main()
