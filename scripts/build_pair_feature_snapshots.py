from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from jsonschema import Draft202012Validator


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_by_image_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    return {str(r["image_id"]): r for r in rows}


def flag(value: bool) -> float:
    return 1.0 if value else 0.0


def safe_region_score(region_row: dict[str, Any], region_name: str) -> float | None:
    region = region_row["regions"].get(region_name)
    if region is None:
        return None
    val = region.get("region_safe_score_diagnostic")
    return None if val is None else float(val)


def region_metric(region_row: dict[str, Any], region_name: str, key: str) -> float | None:
    region = region_row["regions"].get(region_name)
    if region is None:
        return None
    val = region.get(key)
    return None if val is None else float(val)


def make_feature_snapshot_id(
    snapshot_version: str,
    campaign_id: str,
    image_id: str,
    layout_spec_id: str,
    preview_renderer_version: str | None = None,
) -> str:
    """Return an immutable feature id at the image-layout granularity.

    v2.2.1 used campaign_id + image_id only. That is unsafe once the same
    image is scored under multiple layout specs because layout-dependent
    features such as title_region_safe_score and info_region_safe_score can
    differ. v2.2.2 therefore includes layout_spec_id, and optionally the
    preview renderer version when preview-derived critic features are present.
    """
    parts = ["feat_" + snapshot_version, campaign_id, image_id, layout_spec_id]
    if preview_renderer_version:
        parts.append(preview_renderer_version)
    return "__".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--manifest", default="data/ontology/raw_image_manifest_v2_2_1.jsonl")
    ap.add_argument("--duplicates", default="data/ontology/duplicate_groups_v1.jsonl")
    ap.add_argument("--clusters", default="data/ontology/dinov2_clusters_v1.jsonl")
    ap.add_argument("--region-safety", default="data/ontology/region_safety_maps_v1.jsonl")
    ap.add_argument("--schema", default="schemas/pair_feature_snapshot.schema.json")
    ap.add_argument("--out", required=True)

    ap.add_argument("--snapshot-version", default="v2_2_2")
    ap.add_argument("--batch-id", default=None)
    ap.add_argument("--title-region", default="top_left")
    ap.add_argument("--info-region", default="bottom_left")
    ap.add_argument("--preview-renderer-version", default=None)
    args = ap.parse_args()

    campaign = read_json(Path(args.campaign))
    retrieval_rows = read_jsonl(Path(args.retrieval))

    manifest_by_id = load_by_image_id(Path(args.manifest))
    duplicate_by_id = load_by_image_id(Path(args.duplicates))
    cluster_by_id = load_by_image_id(Path(args.clusters))
    region_by_id = load_by_image_id(Path(args.region_safety))

    schema = read_json(Path(args.schema))
    validator = Draft202012Validator(schema)

    campaign_id = campaign["campaign_id"]
    campaign_version = campaign.get("campaign_version")
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    batch_id = args.batch_id or f"phase1a_pair_features__{campaign_id}"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []

    for r in retrieval_rows:
        image_id = str(r["image_id"])

        if image_id not in manifest_by_id:
            raise RuntimeError(f"image_id missing from manifest: {image_id}")
        if image_id not in duplicate_by_id:
            raise RuntimeError(f"image_id missing from duplicates: {image_id}")
        if image_id not in cluster_by_id:
            raise RuntimeError(f"image_id missing from clusters: {image_id}")
        if image_id not in region_by_id:
            raise RuntimeError(f"image_id missing from region safety: {image_id}")

        manifest = manifest_by_id[image_id]
        dup = duplicate_by_id[image_id]
        cluster = cluster_by_id[image_id]
        region = region_by_id[image_id]

        title_safe = safe_region_score(region, args.title_region)
        info_safe = safe_region_score(region, args.info_region)
        required_scores = [x for x in [title_safe, info_safe] if x is not None]

        if not required_scores:
            raise RuntimeError(f"no required region scores for image_id={image_id}")

        required_mean = sum(required_scores) / len(required_scores)
        required_min = min(required_scores)

        source_group = manifest.get("source_group")
        category = manifest.get("category")

        features = {
            # CLIP retrieval features
            "clip_positive_max_sim": r.get("clip_positive_max_sim"),
            "clip_positive_mean_sim": r.get("clip_positive_mean_sim"),
            "clip_negative_max_sim": r.get("clip_negative_max_sim"),
            "clip_negative_mean_sim": r.get("clip_negative_mean_sim"),
            "clip_margin": r.get("clip_margin"),
            "clip_rank_percentile": r.get("clip_rank_percentile"),

            # DINOv2 anchor features are missing in cold-start.
            "dinov2_campaign_pos_nn_sim": None,
            "dinov2_campaign_neg_nn_sim": None,
            "dinov2_campaign_margin": None,
            "dinov2_campaign_pos_count": 0,
            "dinov2_campaign_neg_count": 0,
            "dinov2_campaign_anchor_missing": 1.0,

            "dinov2_family_pos_nn_sim": None,
            "dinov2_family_neg_nn_sim": None,
            "dinov2_family_margin": None,
            "dinov2_family_support_count": 0,
            "dinov2_family_anchor_missing": 1.0,

            # Cluster features
            "dinov2_cluster_id_coarse": cluster.get("dinov2_cluster_id_coarse"),
            "dinov2_cluster_id_mid": cluster.get("dinov2_cluster_id_mid"),
            "dinov2_cluster_id_fine": cluster.get("dinov2_cluster_id_fine"),
            "dinov2_cluster_positive_rate_coarse": None,
            "dinov2_cluster_positive_rate_mid": None,
            "dinov2_cluster_positive_rate_fine": None,
            "dinov2_cluster_review_count_coarse": 0,
            "dinov2_cluster_review_count_mid": 0,
            "dinov2_cluster_review_count_fine": 0,
            "dinov2_duplicate_group_seen": 0.0,

            # Layout/region safety features
            "required_region_safe_mean": required_mean,
            "required_region_safe_min": required_min,
            "title_region_safe_score": title_safe,
            "info_region_safe_score": info_safe,
            "edge_density": region_metric(region, "full", "edge_density"),
            "contrast": region_metric(region, "full", "contrast_std"),
            "brightness": region_metric(region, "full", "brightness_mean"),
            "saturation": region_metric(region, "full", "saturation_mean"),

            # Critic absent in Phase 1a.
            "critic_text_region_conflict_score": None,
            "critic_low_contrast_score": None,
            "critic_too_busy_background_score": None,
            "critic_visual_hierarchy_risk_score": None,
            "critic_layout_fit_score": None,

            # Weak metadata flags
            "image_season_unknown": 1.0,
            "image_category_gallery": flag(category == "gallery"),
            "image_category_tree": flag(source_group == "tree"),
            "image_category_flower": flag(source_group == "flower"),
            "image_category_course": flag(category == "course"),
            "path_has_architecture": flag(source_group == "건축"),
            "path_has_garden": flag(source_group == "정원"),

            # Campaign metadata flags
            "campaign_is_summer": flag(campaign.get("season") == "summer"),
            "campaign_is_walking_program": flag(campaign.get("purpose_type") == "walking_program"),
            "campaign_is_garden": flag(campaign.get("space_type") == "garden"),
        }

        layout_spec_id = f"layout_{args.title_region}_{args.info_region}"
        pair_id = f"{campaign_id}__{image_id}__{layout_spec_id}"
        feature_snapshot_id = make_feature_snapshot_id(
            args.snapshot_version,
            campaign_id,
            image_id,
            layout_spec_id,
            args.preview_renderer_version,
        )

        row = {
            "feature_snapshot_id": feature_snapshot_id,
            "snapshot_version": args.snapshot_version,
            "batch_id": batch_id,
            "created_at": now,

            "pair_id": pair_id,
            "campaign_id": campaign_id,
            "image_id": image_id,
            "layout_spec_id": layout_spec_id,
            "duplicate_group_id": dup.get("duplicate_group_id"),

            "feature_status": "diagnostic_only",

            "provenance": {
                "raw_image_asset_version": manifest.get("manifest_version"),
                "campaign_version": campaign_version,
                "layout_spec_version": "phase1a_layout_regions_v1",
                "preview_renderer_version": args.preview_renderer_version,
                "clip_embedding_version": r.get("prompt_bank_version"),
                "dinov2_embedding_version": cluster.get("embedding_version"),
                "region_safety_version": region.get("region_version"),
                "critic_version": None,
                "prompt_template_bank_version": r.get("prompt_bank_version"),
                "cluster_version": cluster.get("cluster_version"),
                "duplicate_group_version": dup.get("duplicate_version"),
            },

            "features": features,
        }

        errors = sorted(validator.iter_errors(row), key=lambda e: e.path)
        if errors:
            first = errors[0]
            raise RuntimeError(
                f"schema validation failed for image_id={image_id}: "
                f"path={list(first.path)} message={first.message}"
            )

        output_rows.append(row)

    with out_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "event": "done",
        "campaign_id": campaign_id,
        "out": str(out_path),
        "rows": len(output_rows),
        "schema": args.schema,
        "feature_status": "diagnostic_only",
        "title_region": args.title_region,
        "info_region": args.info_region,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
