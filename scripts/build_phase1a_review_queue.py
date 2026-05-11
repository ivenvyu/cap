from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def flatten_feature_row(row: dict[str, Any], manifest_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    f = row["features"]
    image_id = row["image_id"]
    manifest = manifest_by_id[image_id]

    return {
        "queue_stage": "cold_start",
        "campaign_id": row["campaign_id"],
        "pair_id": row["pair_id"],
        "image_id": image_id,
        "duplicate_group_id": row.get("duplicate_group_id"),
        "feature_snapshot_id": row["feature_snapshot_id"],
        "layout_spec_id": row.get("layout_spec_id"),

        "image_path": manifest["path"],
        "resolved_path": manifest["resolved_path"],
        "category": manifest.get("category"),
        "source_group": manifest.get("source_group"),
        "place_name": manifest.get("place_name"),
        "subject_name": manifest.get("subject_name"),

        "diagnostic_model_score": None,

        "clip_margin": f.get("clip_margin"),
        "clip_positive_max_sim": f.get("clip_positive_max_sim"),
        "clip_negative_max_sim": f.get("clip_negative_max_sim"),
        "clip_rank_percentile": f.get("clip_rank_percentile"),

        "required_region_safe_mean": f.get("required_region_safe_mean"),
        "required_region_safe_min": f.get("required_region_safe_min"),
        "title_region_safe_score": f.get("title_region_safe_score"),
        "info_region_safe_score": f.get("info_region_safe_score"),

        "dinov2_cluster_id_coarse": f.get("dinov2_cluster_id_coarse"),
        "dinov2_cluster_id_mid": f.get("dinov2_cluster_id_mid"),
        "dinov2_cluster_id_fine": f.get("dinov2_cluster_id_fine"),

        "image_category_gallery": f.get("image_category_gallery"),
        "image_category_tree": f.get("image_category_tree"),
        "image_category_flower": f.get("image_category_flower"),
        "image_category_course": f.get("image_category_course"),
        "path_has_architecture": f.get("path_has_architecture"),
        "path_has_garden": f.get("path_has_garden"),

        "preview_path": None,

        # human review fields
        "decision": None,
        "issue_tags": None,
        "preference_rank": None,
        "notes": None,
    }


def add_bucket(
    selected: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    bucket_name: str,
    budget: int,
    seen_duplicate_groups: set[str],
    seen_image_ids: set[str],
) -> None:
    added = 0

    for r in candidates:
        if added >= budget:
            break

        image_id = str(r["image_id"])
        dup = str(r["duplicate_group_id"])

        if image_id in seen_image_ids:
            continue
        if dup in seen_duplicate_groups:
            continue

        out = dict(r)
        out["source_bucket"] = bucket_name
        selected.append(out)

        seen_image_ids.add(image_id)
        seen_duplicate_groups.add(dup)
        added += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", default="data/ontology/raw_image_manifest_v2_2_1.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--queue-id", default="review_queue_phase1a_v1")
    ap.add_argument("--seed", type=int, default=20260510)

    # These are review budget allocations, not quality thresholds.
    ap.add_argument("--clip-budget", type=int, default=12)
    ap.add_argument("--layout-safe-budget", type=int, default=6)
    ap.add_argument("--cluster-budget", type=int, default=8)
    ap.add_argument("--uncertainty-budget", type=int, default=3)
    ap.add_argument("--random-budget", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)

    feature_rows = read_jsonl(Path(args.features))
    manifest_rows = read_jsonl(Path(args.manifest))
    manifest_by_id = {r["image_id"]: r for r in manifest_rows}

    flat = [flatten_feature_row(r, manifest_by_id) for r in feature_rows]

    selected: list[dict[str, Any]] = []
    seen_duplicate_groups: set[str] = set()
    seen_image_ids: set[str] = set()

    # Bucket 1. CLIP semantic high.
    clip_sorted = sorted(
        flat,
        key=lambda r: (r["clip_margin"], r["clip_positive_max_sim"]),
        reverse=True,
    )
    add_bucket(
        selected,
        clip_sorted,
        "clip_high_model_low",
        args.clip_budget,
        seen_duplicate_groups,
        seen_image_ids,
    )

    # Bucket 2. Layout-safe coverage.
    layout_sorted = sorted(
        flat,
        key=lambda r: (r["required_region_safe_min"], r["required_region_safe_mean"]),
        reverse=True,
    )
    add_bucket(
        selected,
        layout_sorted,
        "layout_safe_coverage",
        args.layout_safe_budget,
        seen_duplicate_groups,
        seen_image_ids,
    )

    # Bucket 3. Cluster diversity: best CLIP margin per mid cluster.
    best_by_cluster: dict[str, dict[str, Any]] = {}
    for r in flat:
        cid = str(r["dinov2_cluster_id_mid"])
        current = best_by_cluster.get(cid)
        if current is None or r["clip_margin"] > current["clip_margin"]:
            best_by_cluster[cid] = r

    cluster_candidates = sorted(
        best_by_cluster.values(),
        key=lambda r: (r["clip_margin"], r["required_region_safe_min"]),
        reverse=True,
    )
    add_bucket(
        selected,
        cluster_candidates,
        "cluster_diversity",
        args.cluster_budget,
        seen_duplicate_groups,
        seen_image_ids,
    )

    # Bucket 4. Uncertainty proxy: candidates near median CLIP margin.
    margins = sorted(r["clip_margin"] for r in flat)
    median_margin = margins[len(margins) // 2]
    uncertainty_candidates = sorted(
        flat,
        key=lambda r: abs(r["clip_margin"] - median_margin),
    )
    add_bucket(
        selected,
        uncertainty_candidates,
        "uncertainty",
        args.uncertainty_budget,
        seen_duplicate_groups,
        seen_image_ids,
    )

    # Bucket 5. Random coverage.
    random_candidates = list(flat)
    random.shuffle(random_candidates)
    add_bucket(
        selected,
        random_candidates,
        "random_coverage",
        args.random_budget,
        seen_duplicate_groups,
        seen_image_ids,
    )

    for i, r in enumerate(selected, start=1):
        r["queue_id"] = args.queue_id
        r["queue_row_id"] = f"{args.queue_id}_{i:04d}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(selected)

    preferred_cols = [
        "queue_id",
        "queue_row_id",
        "queue_stage",
        "source_bucket",
        "campaign_id",
        "pair_id",
        "image_id",
        "duplicate_group_id",
        "feature_snapshot_id",
        "layout_spec_id",
        "image_path",
        "resolved_path",
        "category",
        "source_group",
        "place_name",
        "subject_name",
        "diagnostic_model_score",
        "clip_margin",
        "clip_positive_max_sim",
        "clip_negative_max_sim",
        "clip_rank_percentile",
        "required_region_safe_mean",
        "required_region_safe_min",
        "title_region_safe_score",
        "info_region_safe_score",
        "dinov2_cluster_id_coarse",
        "dinov2_cluster_id_mid",
        "dinov2_cluster_id_fine",
        "preview_path",
        "decision",
        "issue_tags",
        "preference_rank",
        "notes",
    ]

    rest = [c for c in df.columns if c not in preferred_cols]
    df = df[preferred_cols + rest]

    df.to_csv(out_path, index=False)

    print(json.dumps({
        "event": "done",
        "out": str(out_path),
        "rows": len(df),
        "unique_duplicate_groups": int(df["duplicate_group_id"].nunique()),
        "source_bucket_counts": df["source_bucket"].value_counts().to_dict(),
        "queue_stage": "cold_start",
        "score_status": "diagnostic_only",
        "buckets_used": sorted(df["source_bucket"].unique().tolist()),
        "buckets_skipped": {
            "dinov2_anchor_high_model_low": "no_campaign_positive_anchor_in_cold_start",
            "model_high_clip_negative_high": "no_diagnostic_model_score_in_cold_start",
            "critic_high_risk_reranker_high": "critic_unavailable_in_phase1a",
            "classifier_ranker_disagreement": "ranker_unavailable_in_phase1a"
        },
        "spec_compliance_note": (
            "Phase 1a uses cold_start fallback buckets. Skipped buckets require "
            "campaign anchors, trained model scores, critic scores, or ranker scores."
        )
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
