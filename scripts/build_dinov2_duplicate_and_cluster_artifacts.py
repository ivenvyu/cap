from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def stable_id(prefix: str, value: str, n: int = 12) -> str:
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]
    return f"{prefix}_{h}"


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["image_id"]] = r
    return rows


def nearest_neighbors_cosine(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # embeddings are already L2-normalized, so dot product = cosine similarity
    sim = emb @ emb.T
    np.fill_diagonal(sim, -np.inf)

    nn_idx = np.argmax(sim, axis=1)
    nn_sim = sim[np.arange(sim.shape[0]), nn_idx]

    return nn_idx, nn_sim


def run_kmeans(emb: np.ndarray, k: int, random_state: int) -> np.ndarray:
    n = emb.shape[0]
    k_eff = min(k, n)

    if k_eff <= 1:
        return np.zeros(n, dtype=int)

    model = KMeans(
        n_clusters=k_eff,
        random_state=random_state,
        n_init=10,
        algorithm="lloyd",
    )
    return model.fit_predict(emb)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--dinov2-embeddings", required=True)
    ap.add_argument("--dinov2-index", required=True)
    ap.add_argument("--duplicate-out", required=True)
    ap.add_argument("--cluster-out", required=True)
    ap.add_argument("--duplicate-version", default="duplicate_groups_exact_sha256_v1")
    ap.add_argument("--cluster-version", default="dinov2_clusters_v1")
    ap.add_argument("--coarse-k", type=int, default=20)
    ap.add_argument("--mid-k", type=int, default=100)
    ap.add_argument("--fine-k", type=int, default=500)
    ap.add_argument("--random-state", type=int, default=20260510)
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    emb = np.load(args.dinov2_embeddings).astype("float32")
    idx = pd.read_csv(args.dinov2_index)

    if emb.shape[0] != len(idx):
        raise RuntimeError("embedding row count != index row count")

    if idx["image_id"].duplicated().any():
        raise RuntimeError("duplicate image_id in DINOv2 index")

    missing_manifest = sorted(set(idx["image_id"]) - set(manifest))
    if missing_manifest:
        raise RuntimeError(f"image ids missing from manifest: {missing_manifest[:10]}")

    duplicate_out = Path(args.duplicate_out)
    cluster_out = Path(args.cluster_out)
    duplicate_out.parent.mkdir(parents=True, exist_ok=True)
    cluster_out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Exact duplicate groups by file SHA256.
    file_hashes: dict[str, str] = {}
    hash_to_image_ids: dict[str, list[str]] = defaultdict(list)

    for _, row in idx.iterrows():
        image_id = str(row["image_id"])
        resolved_path = Path(str(row["resolved_path"]))
        digest = sha256_file(resolved_path)
        file_hashes[image_id] = digest
        hash_to_image_ids[digest].append(image_id)

    image_to_duplicate_group = {}
    for digest, image_ids in hash_to_image_ids.items():
        group_id = stable_id("dup_exact", digest)
        for image_id in image_ids:
            image_to_duplicate_group[image_id] = group_id

    duplicate_group_sizes = {
        stable_id("dup_exact", digest): len(image_ids)
        for digest, image_ids in hash_to_image_ids.items()
    }

    # 2. DINOv2 nearest-neighbor diagnostics.
    nn_idx, nn_sim = nearest_neighbors_cosine(emb)

    duplicate_rows = []
    for i, row in idx.iterrows():
        image_id = str(row["image_id"])
        nn_image_id = str(idx.iloc[int(nn_idx[i])]["image_id"])
        group_id = image_to_duplicate_group[image_id]
        group_size = duplicate_group_sizes[group_id]

        duplicate_rows.append({
            "image_id": image_id,
            "duplicate_group_id": group_id,
            "duplicate_version": args.duplicate_version,
            "duplicate_method": "exact_file_sha256_v1",
            "exact_file_sha256": file_hashes[image_id],
            "exact_duplicate_group_size": group_size,
            "exact_duplicate_status": "exact_duplicate" if group_size > 1 else "exact_singleton",

            "nearest_dinov2_neighbor_image_id": nn_image_id,
            "nearest_dinov2_cosine_sim": float(nn_sim[i]),
            "nearest_dinov2_status": "diagnostic_only",

            "path": row["path"],
            "resolved_path": row["resolved_path"],
            "category": row.get("category"),
            "source_group": row.get("source_group"),
            "place_name": None if pd.isna(row.get("place_name")) else row.get("place_name"),
            "subject_name": None if pd.isna(row.get("subject_name")) else row.get("subject_name"),
        })

    with duplicate_out.open("w", encoding="utf-8") as f:
        for r in duplicate_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    # 3. Hierarchical DINOv2 clusters.
    labels_coarse = run_kmeans(emb, args.coarse_k, args.random_state)
    labels_mid = run_kmeans(emb, args.mid_k, args.random_state)
    labels_fine = run_kmeans(emb, args.fine_k, args.random_state)

    k_coarse_eff = len(set(labels_coarse))
    k_mid_eff = len(set(labels_mid))
    k_fine_eff = len(set(labels_fine))

    coarse_counts = Counter(labels_coarse)
    mid_counts = Counter(labels_mid)
    fine_counts = Counter(labels_fine)

    cluster_rows = []
    for i, row in idx.iterrows():
        image_id = str(row["image_id"])
        c = int(labels_coarse[i])
        m = int(labels_mid[i])
        fi = int(labels_fine[i])

        cluster_rows.append({
            "image_id": image_id,
            "cluster_version": args.cluster_version,
            "cluster_method": "kmeans_on_l2_normalized_dinov2_embeddings",
            "embedding_version": row["embedding_version"],
            "model_name": row["model_name"],

            "dinov2_cluster_id_coarse": f"coarse_{k_coarse_eff}_{c:04d}",
            "dinov2_cluster_id_mid": f"mid_{k_mid_eff}_{m:04d}",
            "dinov2_cluster_id_fine": f"fine_{k_fine_eff}_{fi:04d}",

            "dinov2_cluster_size_coarse": int(coarse_counts[c]),
            "dinov2_cluster_size_mid": int(mid_counts[m]),
            "dinov2_cluster_size_fine": int(fine_counts[fi]),

            "k_requested_coarse": args.coarse_k,
            "k_requested_mid": args.mid_k,
            "k_requested_fine": args.fine_k,
            "k_effective_coarse": k_coarse_eff,
            "k_effective_mid": k_mid_eff,
            "k_effective_fine": k_fine_eff,

            "cluster_status": "diagnostic_only",
            "path": row["path"],
            "category": row.get("category"),
            "source_group": row.get("source_group"),
            "duplicate_group_id": image_to_duplicate_group[image_id],
        })

    with cluster_out.open("w", encoding="utf-8") as f:
        for r in cluster_rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")

    exact_duplicate_groups = sum(1 for v in duplicate_group_sizes.values() if v > 1)

    print(json.dumps({
        "event": "done",
        "rows": len(idx),
        "duplicate_out": str(duplicate_out),
        "cluster_out": str(cluster_out),
        "exact_duplicate_groups": exact_duplicate_groups,
        "exact_duplicate_singletons": sum(1 for v in duplicate_group_sizes.values() if v == 1),
        "k_effective_coarse": k_coarse_eff,
        "k_effective_mid": k_mid_eff,
        "k_effective_fine": k_fine_eff,
        "cluster_status": "diagnostic_only",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
