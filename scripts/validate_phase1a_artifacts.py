from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ids(rows: list[dict]) -> list[str]:
    return [str(r["image_id"]) for r in rows]


def require_same_set(name_a: str, ids_a: set[str], name_b: str, ids_b: set[str]) -> None:
    missing_b = sorted(ids_a - ids_b)
    missing_a = sorted(ids_b - ids_a)
    if missing_b or missing_a:
        raise RuntimeError(
            f"image_id set mismatch: {name_a} vs {name_b}; "
            f"missing_in_{name_b}={missing_b[:10]}, missing_in_{name_a}={missing_a[:10]}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ontology/raw_image_manifest_v2_2_1.jsonl")
    ap.add_argument("--clip-embeddings", default="data/embeddings/clip_image_embeddings.npy")
    ap.add_argument("--clip-index", default="data/embeddings/clip_image_index.csv")
    ap.add_argument("--dinov2-embeddings", default="data/embeddings/dinov2_image_embeddings.npy")
    ap.add_argument("--dinov2-index", default="data/embeddings/dinov2_image_index.csv")
    ap.add_argument("--duplicate-groups", default="data/ontology/duplicate_groups_v1.jsonl")
    ap.add_argument("--clusters", default="data/ontology/dinov2_clusters_v1.jsonl")
    ap.add_argument("--region-safety", default="data/ontology/region_safety_maps_v1.jsonl")
    args = ap.parse_args()

    manifest = read_jsonl(Path(args.manifest))
    duplicate_rows = read_jsonl(Path(args.duplicate_groups))
    cluster_rows = read_jsonl(Path(args.clusters))
    region_rows = read_jsonl(Path(args.region_safety))

    clip_idx = pd.read_csv(args.clip_index)
    dinov2_idx = pd.read_csv(args.dinov2_index)

    clip_emb = np.load(args.clip_embeddings)
    dinov2_emb = np.load(args.dinov2_embeddings)

    print("manifest rows:", len(manifest))
    print("clip index rows:", len(clip_idx), "clip shape:", clip_emb.shape)
    print("dinov2 index rows:", len(dinov2_idx), "dinov2 shape:", dinov2_emb.shape)
    print("duplicate rows:", len(duplicate_rows))
    print("cluster rows:", len(cluster_rows))
    print("region safety rows:", len(region_rows))

    if len(manifest) != 206:
        raise RuntimeError(f"expected Phase 1a manifest rows 206, got {len(manifest)}")

    if clip_emb.shape[0] != len(clip_idx):
        raise RuntimeError("CLIP embedding rows != CLIP index rows")

    if dinov2_emb.shape[0] != len(dinov2_idx):
        raise RuntimeError("DINOv2 embedding rows != DINOv2 index rows")

    manifest_ids = set(ids(manifest))
    clip_ids = set(clip_idx["image_id"].astype(str))
    dinov2_ids = set(dinov2_idx["image_id"].astype(str))
    duplicate_ids = set(ids(duplicate_rows))
    cluster_ids = set(ids(cluster_rows))
    region_ids = set(ids(region_rows))

    for name, current_ids in [
        ("clip", clip_ids),
        ("dinov2", dinov2_ids),
        ("duplicate", duplicate_ids),
        ("cluster", cluster_ids),
        ("region_safety", region_ids),
    ]:
        require_same_set("manifest", manifest_ids, name, current_ids)

    if list(clip_idx["image_id"].astype(str)) != list(dinov2_idx["image_id"].astype(str)):
        raise RuntimeError("CLIP and DINOv2 image_id order mismatch")

    if clip_emb.shape[1] != 512:
        raise RuntimeError(f"expected CLIP dim 512, got {clip_emb.shape[1]}")

    if dinov2_emb.shape[1] != 768:
        raise RuntimeError(f"expected DINOv2 dim 768, got {dinov2_emb.shape[1]}")

    region_counts = {len(r["regions"]) for r in region_rows}
    if region_counts != {16}:
        raise RuntimeError(f"expected all region maps to have 16 regions, got {region_counts}")

    duplicate_group_ids = [r["duplicate_group_id"] for r in duplicate_rows]
    if any(x is None or x == "" for x in duplicate_group_ids):
        raise RuntimeError("empty duplicate_group_id found")

    for r in cluster_rows:
        for key in [
            "dinov2_cluster_id_coarse",
            "dinov2_cluster_id_mid",
            "dinov2_cluster_id_fine",
        ]:
            if not r.get(key):
                raise RuntimeError(f"missing {key} for image_id={r.get('image_id')}")

    for r in region_rows:
        if r.get("score_status") != "diagnostic_only":
            raise RuntimeError(f"region safety score_status must be diagnostic_only: {r.get('image_id')}")
        for region_name, region in r["regions"].items():
            val = region.get("region_safe_score_diagnostic")
            if val is None or not (0.0 <= float(val) <= 1.0):
                raise RuntimeError(
                    f"bad region_safe_score_diagnostic for {r.get('image_id')} {region_name}: {val}"
                )

    print()
    print("PHASE 1A ARTIFACT CHECKS PASSED")


if __name__ == "__main__":
    main()
