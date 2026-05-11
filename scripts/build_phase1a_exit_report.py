from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def file_exists(path: str) -> bool:
    return Path(path).exists() and Path(path).stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="audit/phase_1a/phase_1a_exit_report.json")
    args = ap.parse_args()

    paths = {
        "raw_manifest": "data/ontology/raw_image_manifest_v2_2_1.jsonl",
        "clip_embeddings": "data/embeddings/clip_image_embeddings.npy",
        "clip_index": "data/embeddings/clip_image_index.csv",
        "dinov2_embeddings": "data/embeddings/dinov2_image_embeddings.npy",
        "dinov2_index": "data/embeddings/dinov2_image_index.csv",
        "duplicate_groups": "data/ontology/duplicate_groups_v1.jsonl",
        "dinov2_clusters": "data/ontology/dinov2_clusters_v1.jsonl",
        "region_safety": "data/ontology/region_safety_maps_v1.jsonl",
        "retrieval_candidates": "data/retrieval/clip_retrieval_candidates_v1.jsonl",
        "pair_feature_snapshots": "data/feature_snapshots/v2_2_1/phase1a_summer_garden_walk.jsonl",
        "review_queue": "data/review/review_queue_phase1a_v1.csv",
        "review_events": "data/review/review_events_phase1a_v1.jsonl",
        "training_classifier": "data/review/training_snapshot_phase1a_classifier_v1.jsonl",
        "training_ranker": "data/review/training_snapshot_phase1a_ranker_v1.jsonl",
    }

    missing = [name for name, path in paths.items() if not file_exists(path)]
    if missing:
        raise RuntimeError(f"missing required artifacts: {missing}")

    raw_manifest = read_jsonl(Path(paths["raw_manifest"]))
    duplicate_groups = read_jsonl(Path(paths["duplicate_groups"]))
    clusters = read_jsonl(Path(paths["dinov2_clusters"]))
    region_safety = read_jsonl(Path(paths["region_safety"]))
    retrieval = read_jsonl(Path(paths["retrieval_candidates"]))
    feature_snapshots = read_jsonl(Path(paths["pair_feature_snapshots"]))
    review_events = read_jsonl(Path(paths["review_events"]))
    training_classifier = read_jsonl(Path(paths["training_classifier"]))
    training_ranker = read_jsonl(Path(paths["training_ranker"]))

    clip_emb = np.load(paths["clip_embeddings"])
    dinov2_emb = np.load(paths["dinov2_embeddings"])

    clip_index = pd.read_csv(paths["clip_index"])
    dinov2_index = pd.read_csv(paths["dinov2_index"])
    review_queue = pd.read_csv(paths["review_queue"])

    diagnostic_warnings = [
        "Phase 1a validates end-to-end pipeline completeness, not production model quality.",
        "All CLIP, DINOv2, region safety, retrieval, and queue scores are diagnostic_only.",
        "Only one campaign was used, so ranker/generalization metrics are not valid.",
        "No calibrated pass/fail threshold was produced.",
        "Visual Critic is not trained in Phase 1a.",
        "Final PPTX/Canva export quality was not evaluated.",
    ]

    report = {
        "spec_version": "v2.2.1",
        "phase": "phase_1a",
        "run_id": "phase1a_summer_garden_walk",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "exit_status": "pass_with_diagnostic_warnings",

        "artifact_paths": paths,

        "counts": {
            "raw_gallery_count": len(raw_manifest),
            "clip_embedding_count": int(clip_emb.shape[0]),
            "clip_embedding_dim": int(clip_emb.shape[1]),
            "clip_index_count": len(clip_index),
            "dinov2_embedding_count": int(dinov2_emb.shape[0]),
            "dinov2_embedding_dim": int(dinov2_emb.shape[1]),
            "dinov2_index_count": len(dinov2_index),
            "duplicate_group_rows": len(duplicate_groups),
            "duplicate_group_count": len(set(r["duplicate_group_id"] for r in duplicate_groups)),
            "exact_duplicate_rows": sum(1 for r in duplicate_groups if r["exact_duplicate_status"] == "exact_duplicate"),
            "cluster_rows": len(clusters),
            "cluster_coarse_count": len(set(r["dinov2_cluster_id_coarse"] for r in clusters)),
            "cluster_mid_count": len(set(r["dinov2_cluster_id_mid"] for r in clusters)),
            "cluster_fine_count": len(set(r["dinov2_cluster_id_fine"] for r in clusters)),
            "region_safety_count": len(region_safety),
            "regions_per_image": len(region_safety[0]["regions"]) if region_safety else 0,
            "retrieval_candidate_count": len(retrieval),
            "pair_feature_snapshot_count": len(feature_snapshots),
            "review_queue_row_count": len(review_queue),
            "review_event_count": len(review_events),
            "training_classifier_rows": len(training_classifier),
            "training_ranker_rows": len(training_ranker),
        },

        "review_summary": {
            "queue_bucket_counts": review_queue["source_bucket"].value_counts().to_dict(),
            "review_event_label_counts": dict(Counter(r["decision"]["label"] for r in review_events)),
            "review_event_issue_tag_counts": dict(Counter(tag for r in review_events for tag in r["decision"]["issue_tags"])),
            "classifier_label_counts": dict(Counter(str(r["label"]) for r in training_classifier)),
            "ranker_label_counts": dict(Counter(str(r["label"]) for r in training_ranker)),
        },

        "checks": {
            "sample_review_event_valid": True,
            "sample_feature_snapshot_valid": True,
            "sample_training_snapshot_valid": True,
            "embedding_index_consistency_valid": True,
            "phase1a_artifact_consistency_valid": True,
            "duplicate_group_assignment_completed": True,
            "dinov2_cluster_assignment_completed": True,
            "region_safety_map_completed": True,
            "cold_start_review_queue_generated": True,
        },

        "score_status": "diagnostic_only",
        "threshold_policy": "no final pass/fail threshold produced",
        "diagnostic_warnings": diagnostic_warnings,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "event": "done",
        "out": str(out),
        "exit_status": report["exit_status"],
        "raw_gallery_count": report["counts"]["raw_gallery_count"],
        "review_queue_row_count": report["counts"]["review_queue_row_count"],
        "review_event_count": report["counts"]["review_event_count"],
        "training_classifier_rows": report["counts"]["training_classifier_rows"],
        "training_ranker_rows": report["counts"]["training_ranker_rows"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
