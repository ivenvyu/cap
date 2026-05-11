from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from jsonschema import Draft202012Validator

ANCHOR_FEATURE_VERSION = "dinov2_anchor_features_from_db_v1"


def jload(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    obj = json.loads(s)
    return obj if isinstance(obj, dict) else {}


def make_feature_snapshot_id(
    snapshot_version: str,
    campaign_id: str,
    image_id: str,
    layout_spec_id: str | None,
    preview_renderer_version: str | None = None,
) -> str:
    layout = layout_spec_id or "layout_null"
    parts = ["feat_" + snapshot_version, campaign_id, image_id, layout]
    if preview_renderer_version:
        parts.append(preview_renderer_version)
    return "__".join(parts)


def load_embeddings(index_path: Path, embedding_path: Path) -> dict[str, np.ndarray]:
    idx = pd.read_csv(index_path)
    emb = np.load(embedding_path)
    if len(idx) != emb.shape[0]:
        raise RuntimeError(f"embedding index length mismatch: {len(idx)} vs {emb.shape[0]}")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb = emb / norms
    out: dict[str, np.ndarray] = {}
    for _, row in idx.iterrows():
        out[str(row["image_id"])] = emb[int(row["embedding_row"])]
    return out


def nn_sim(image_id: str, anchor_ids: list[str], embeddings: dict[str, np.ndarray]) -> float | None:
    if image_id not in embeddings:
        return None
    anchor_vecs = [embeddings[i] for i in anchor_ids if i in embeddings]
    if not anchor_vecs:
        return None
    mat = np.vstack(anchor_vecs)
    sims = mat @ embeddings[image_id]
    return float(np.max(sims))


def family_match(target: dict[str, Any], source: dict[str, Any]) -> bool:
    if target.get("campaign_id") == source.get("campaign_id"):
        return False
    for key in ["purpose_type", "space_type", "season"]:
        if target.get(key) is not None and target.get(key) == source.get(key):
            return True
    target_mood = set(target.get("mood_tags") or [])
    source_mood = set(source.get("mood_tags") or [])
    return bool(target_mood & source_mood)


def select_anchor_ids(
    label_rows: list[dict[str, Any]],
    positive: bool,
    exclude_image_id: str | None,
) -> list[str]:
    out = []
    for row in label_rows:
        if exclude_image_id is not None and row["image_id"] == exclude_image_id:
            continue
        is_pos = int(row["label"]) >= 1
        if is_pos == positive:
            out.append(row["image_id"])
    return sorted(set(out))


def cluster_stats(
    cluster_labels: dict[str, list[dict[str, Any]]],
    cluster_id: str | None,
    exclude_image_id: str | None,
) -> tuple[float | None, int]:
    if cluster_id is None:
        return None, 0
    rows = [r for r in cluster_labels.get(str(cluster_id), []) if r["image_id"] != exclude_image_id]
    if not rows:
        return None, 0
    positives = sum(1 for r in rows if int(r["label"]) >= 1)
    return positives / len(rows), len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build v2.2.2 PairFeatureSnapshots with DINOv2 anchor/cluster features from DB labels.")
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--source", default="pair_features")
    ap.add_argument("--embedding-index", default="data/embeddings/dinov2_image_index.csv")
    ap.add_argument("--embeddings", default="data/embeddings/dinov2_image_embeddings.npy")
    ap.add_argument("--schema", default="schemas/pair_feature_snapshot.schema.json")
    ap.add_argument("--out-dir", default="data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor")
    ap.add_argument("--snapshot-version", default="v2_2_2")
    ap.add_argument("--batch-prefix", default="phase1b_dinov2_anchor_features")
    ap.add_argument("--include-phase1a", action="store_true", help="Also export phase1a campaign rows if present.")
    ap.add_argument("--no-leave-one-image-out", action="store_true")
    args = ap.parse_args()

    leave_one_out = not args.no_leave_one_image_out
    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    embeddings = load_embeddings(Path(args.embedding_index), Path(args.embeddings))
    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)

    campaigns: dict[str, dict[str, Any]] = {}
    for r in conn.execute("SELECT * FROM campaigns"):
        raw = jload(r["raw_json"])
        d = dict(r)
        d.update(raw)
        campaigns[str(r["campaign_id"])] = d

    # Labels are intentionally taken from classifier snapshots: reject=0, acceptable/accept/best=1.
    label_query = """
        SELECT ts.campaign_id, ts.image_id, ts.pair_id, ts.layout_spec_id, ts.label,
               pf.duplicate_group_id, pf.features_json
        FROM training_snapshots ts
        JOIN pair_features pf ON ts.feature_snapshot_id = pf.feature_snapshot_id
        WHERE ts.snapshot_kind = 'classifier'
    """
    labels_by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cluster_labels: dict[str, dict[str, list[dict[str, Any]]]] = {
        "coarse": defaultdict(list),
        "mid": defaultdict(list),
        "fine": defaultdict(list),
    }
    duplicate_labels: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in conn.execute(label_query):
        features = jload(r["features_json"])
        row = dict(r)
        row["features"] = features
        labels_by_campaign[str(r["campaign_id"])].append(row)
        for level, key in [
            ("coarse", "dinov2_cluster_id_coarse"),
            ("mid", "dinov2_cluster_id_mid"),
            ("fine", "dinov2_cluster_id_fine"),
        ]:
            cid = features.get(key)
            if cid is not None:
                cluster_labels[level][str(cid)].append(row)
        if r["duplicate_group_id"]:
            duplicate_labels[str(r["duplicate_group_id"])].append(row)

    source_rows = conn.execute(f"SELECT * FROM {args.source} ORDER BY campaign_id, image_id, layout_spec_id").fetchall()
    rows_by_campaign: dict[str, list[dict[str, Any]]] = defaultdict(list)
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    for r in source_rows:
        campaign_id = str(r["campaign_id"])
        if not args.include_phase1a and campaign_id.startswith("phase1a_"):
            continue
        image_id = str(r["image_id"])
        layout_spec_id = r["layout_spec_id"]
        base = jload(r["raw_json"])
        if not base:
            base = {
                "campaign_id": campaign_id,
                "image_id": image_id,
                "pair_id": r["pair_id"],
                "layout_spec_id": layout_spec_id,
                "duplicate_group_id": r["duplicate_group_id"],
                "features": jload(r["features_json"]),
            }
        features = dict(base.get("features") or jload(r["features_json"]))

        exclude = image_id if leave_one_out else None
        local_rows = labels_by_campaign.get(campaign_id, [])
        local_pos = select_anchor_ids(local_rows, positive=True, exclude_image_id=exclude)
        local_neg = select_anchor_ids(local_rows, positive=False, exclude_image_id=exclude)
        pos_sim = nn_sim(image_id, local_pos, embeddings)
        neg_sim = nn_sim(image_id, local_neg, embeddings)
        features["dinov2_campaign_pos_nn_sim"] = pos_sim
        features["dinov2_campaign_neg_nn_sim"] = neg_sim
        features["dinov2_campaign_margin"] = None if pos_sim is None or neg_sim is None else pos_sim - neg_sim
        features["dinov2_campaign_pos_count"] = len(local_pos)
        features["dinov2_campaign_neg_count"] = len(local_neg)
        features["dinov2_campaign_anchor_missing"] = 1.0 if not local_pos else 0.0

        target_campaign = campaigns.get(campaign_id, {"campaign_id": campaign_id})
        family_rows: list[dict[str, Any]] = []
        for other_campaign_id, label_rows in labels_by_campaign.items():
            if family_match(target_campaign, campaigns.get(other_campaign_id, {"campaign_id": other_campaign_id})):
                family_rows.extend(label_rows)
        family_pos = select_anchor_ids(family_rows, positive=True, exclude_image_id=exclude)
        family_neg = select_anchor_ids(family_rows, positive=False, exclude_image_id=exclude)
        f_pos_sim = nn_sim(image_id, family_pos, embeddings)
        f_neg_sim = nn_sim(image_id, family_neg, embeddings)
        features["dinov2_family_pos_nn_sim"] = f_pos_sim
        features["dinov2_family_neg_nn_sim"] = f_neg_sim
        features["dinov2_family_margin"] = None if f_pos_sim is None or f_neg_sim is None else f_pos_sim - f_neg_sim
        features["dinov2_family_support_count"] = len(family_pos) + len(family_neg)
        features["dinov2_family_anchor_missing"] = 1.0 if not family_pos else 0.0

        for level, key, rate_key, count_key in [
            ("coarse", "dinov2_cluster_id_coarse", "dinov2_cluster_positive_rate_coarse", "dinov2_cluster_review_count_coarse"),
            ("mid", "dinov2_cluster_id_mid", "dinov2_cluster_positive_rate_mid", "dinov2_cluster_review_count_mid"),
            ("fine", "dinov2_cluster_id_fine", "dinov2_cluster_positive_rate_fine", "dinov2_cluster_review_count_fine"),
        ]:
            rate, count = cluster_stats(cluster_labels[level], features.get(key), exclude)
            features[rate_key] = rate
            features[count_key] = count

        dup_id = r["duplicate_group_id"]
        if dup_id:
            dup_rows = [x for x in duplicate_labels.get(str(dup_id), []) if x["image_id"] != exclude]
            features["dinov2_duplicate_group_seen"] = 1.0 if dup_rows else 0.0
        else:
            features["dinov2_duplicate_group_seen"] = 0.0

        new_id = make_feature_snapshot_id(args.snapshot_version, campaign_id, image_id, layout_spec_id)
        batch_id = f"{args.batch_prefix}__{campaign_id}"
        base.update({
            "feature_snapshot_id": new_id,
            "snapshot_version": args.snapshot_version,
            "batch_id": batch_id,
            "created_at": now,
            "pair_id": r["pair_id"],
            "campaign_id": campaign_id,
            "image_id": image_id,
            "layout_spec_id": layout_spec_id,
            "duplicate_group_id": r["duplicate_group_id"],
            "feature_status": "diagnostic_only",
            "features": features,
        })
        provenance = dict(base.get("provenance") or {})
        provenance["dinov2_anchor_feature_version"] = ANCHOR_FEATURE_VERSION
        base["provenance"] = provenance

        errors = sorted(validator.iter_errors(base), key=lambda e: e.path)
        if errors:
            e = errors[0]
            raise RuntimeError(f"schema validation failed for {new_id}: path={list(e.path)} message={e.message}")
        rows_by_campaign[campaign_id].append(base)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "created_at": now,
        "snapshot_version": args.snapshot_version,
        "anchor_feature_version": ANCHOR_FEATURE_VERSION,
        "leave_one_image_out": leave_one_out,
        "out_dir": str(out_dir),
        "campaigns": {},
    }
    for campaign_id, rows in sorted(rows_by_campaign.items()):
        out = out_dir / f"pair_feature_snapshots__{campaign_id}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        summary["campaigns"][campaign_id] = {"rows": len(rows), "path": str(out)}

    summary_path = out_dir / "dinov2_anchor_feature_build_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "done", "out_dir": str(out_dir), "campaign_count": len(rows_by_campaign), "rows": sum(len(v) for v in rows_by_campaign.values())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
