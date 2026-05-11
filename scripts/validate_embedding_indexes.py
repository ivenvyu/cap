from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-embeddings", required=True)
    ap.add_argument("--clip-index", required=True)
    ap.add_argument("--dinov2-embeddings", required=True)
    ap.add_argument("--dinov2-index", required=True)
    args = ap.parse_args()

    clip_emb = np.load(args.clip_embeddings)
    dinov2_emb = np.load(args.dinov2_embeddings)

    clip_idx = pd.read_csv(args.clip_index)
    dinov2_idx = pd.read_csv(args.dinov2_index)

    print("clip shape:", clip_emb.shape)
    print("dinov2 shape:", dinov2_emb.shape)
    print("clip index rows:", len(clip_idx))
    print("dinov2 index rows:", len(dinov2_idx))

    if clip_emb.shape[0] != len(clip_idx):
        raise RuntimeError("CLIP embedding row count != CLIP index row count")

    if dinov2_emb.shape[0] != len(dinov2_idx):
        raise RuntimeError("DINOv2 embedding row count != DINOv2 index row count")

    required_cols = [
        "embedding_row",
        "image_id",
        "path",
        "resolved_path",
        "category",
        "source_group",
        "model_name",
        "embedding_version",
        "embedding_dim",
    ]

    for col in required_cols:
        if col not in clip_idx.columns:
            raise RuntimeError(f"missing CLIP index column: {col}")
        if col not in dinov2_idx.columns:
            raise RuntimeError(f"missing DINOv2 index column: {col}")

    if clip_idx["image_id"].duplicated().any():
        raise RuntimeError("duplicate image_id in CLIP index")

    if dinov2_idx["image_id"].duplicated().any():
        raise RuntimeError("duplicate image_id in DINOv2 index")

    if list(clip_idx["image_id"]) != list(dinov2_idx["image_id"]):
        missing_in_dino = sorted(set(clip_idx["image_id"]) - set(dinov2_idx["image_id"]))
        missing_in_clip = sorted(set(dinov2_idx["image_id"]) - set(clip_idx["image_id"]))
        raise RuntimeError(
            "CLIP and DINOv2 image_id order mismatch. "
            f"missing_in_dino={missing_in_dino[:5]}, missing_in_clip={missing_in_clip[:5]}"
        )

    if list(clip_idx["path"]) != list(dinov2_idx["path"]):
        raise RuntimeError("CLIP and DINOv2 path order mismatch")

    if not np.isfinite(clip_emb).all():
        raise RuntimeError("CLIP embeddings contain non-finite values")

    if not np.isfinite(dinov2_emb).all():
        raise RuntimeError("DINOv2 embeddings contain non-finite values")

    clip_norms = np.linalg.norm(clip_emb, axis=1)
    dinov2_norms = np.linalg.norm(dinov2_emb, axis=1)

    print("clip norm min/max:", float(clip_norms.min()), float(clip_norms.max()))
    print("dinov2 norm min/max:", float(dinov2_norms.min()), float(dinov2_norms.max()))

    print("clip model:", clip_idx["model_name"].iloc[0])
    print("clip version:", clip_idx["embedding_version"].iloc[0])
    print("dinov2 model:", dinov2_idx["model_name"].iloc[0])
    print("dinov2 version:", dinov2_idx["embedding_version"].iloc[0])

    print("EMBEDDING INDEX CHECKS PASSED")


if __name__ == "__main__":
    main()
