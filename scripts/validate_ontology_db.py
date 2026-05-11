from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def count(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    tables = [
        "db_builds",
        "artifact_sources",
        "import_batches",
        "images",
        "campaigns",
        "image_embeddings",
        "image_duplicates",
        "image_clusters",
        "image_regions",
        "retrieval_candidates",
        "pair_features",
        "review_events",
        "training_snapshots",
        "training_sets",
        "training_set_items",
        "tag_axes",
        "tag_values",
        "cluster_label_queue",
        "visual_cues",
        "campaign_visual_cue_requirements",
        "campaign_image_cue_scores",
        "plant_entities",
        "plant_names",
        "plant_bloom_priors",
        "campaign_image_botanical_bloom_priors",
        "campaign_image_flower_season_exclusions",
        "retrieval_candidate_filter_decisions",
    ]

    counts = {t: count(conn, t) for t in tables}

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise RuntimeError(f"foreign_key_check failed: {[tuple(r) for r in fk[:10]]}")

    image_count = counts["images"]
    if image_count <= 0:
        raise RuntimeError("images must be non-empty")

    expected_region_rows = image_count * 16
    if counts["image_regions"] != expected_region_rows:
        raise RuntimeError(
            f"image_regions expected {expected_region_rows}, got {counts['image_regions']}"
        )

    expected_embeddings = image_count * 2
    if counts["image_embeddings"] != expected_embeddings:
        raise RuntimeError(
            f"image_embeddings expected {expected_embeddings}, got {counts['image_embeddings']}"
        )

    for table in ["image_duplicates", "image_clusters"]:
        if counts[table] != image_count:
            raise RuntimeError(f"{table} expected {image_count}, got {counts[table]}")

    missing_pf = conn.execute(
        """
        SELECT COUNT(*)
        FROM review_events r
        LEFT JOIN pair_features p ON r.feature_snapshot_id = p.feature_snapshot_id
        WHERE r.feature_snapshot_id IS NOT NULL
          AND p.feature_snapshot_id IS NULL
        """
    ).fetchone()[0]
    if missing_pf:
        raise RuntimeError(f"review_events with missing pair_features: {missing_pf}")

    missing_train_pf = conn.execute(
        """
        SELECT COUNT(*)
        FROM training_snapshots t
        LEFT JOIN pair_features p ON t.feature_snapshot_id = p.feature_snapshot_id
        WHERE t.feature_snapshot_id IS NOT NULL
          AND p.feature_snapshot_id IS NULL
        """
    ).fetchone()[0]
    if missing_train_pf:
        raise RuntimeError(f"training_snapshots with missing pair_features: {missing_train_pf}")

    phase1b_filtered_classifier = conn.execute(
        """
        SELECT COUNT(*)
        FROM training_set_items
        WHERE training_set_id = 'phase1b_filtered_classifier_v1'
        """
    ).fetchone()[0]

    phase1b_filtered_ranker = conn.execute(
        """
        SELECT COUNT(*)
        FROM training_set_items
        WHERE training_set_id = 'phase1b_filtered_ranker_v1'
        """
    ).fetchone()[0]

    phase1b_filtered_classifier_labels = {
        str(row["label"]): int(row["n"])
        for row in conn.execute(
            """
            SELECT t.label, COUNT(*) AS n
            FROM training_set_items i
            JOIN training_snapshots t
              ON i.training_snapshot_id = t.training_snapshot_id
            WHERE i.training_set_id = 'phase1b_filtered_classifier_v1'
            GROUP BY t.label
            ORDER BY t.label
            """
        )
    }

    phase1b_filtered_ranker_labels = {
        str(row["label"]): int(row["n"])
        for row in conn.execute(
            """
            SELECT t.label, COUNT(*) AS n
            FROM training_set_items i
            JOIN training_snapshots t
              ON i.training_snapshot_id = t.training_snapshot_id
            WHERE i.training_set_id = 'phase1b_filtered_ranker_v1'
            GROUP BY t.label
            ORDER BY t.label
            """
        )
    }

    if phase1b_filtered_classifier != 120:
        raise RuntimeError(
            f"phase1b filtered classifier rows expected 120, got {phase1b_filtered_classifier}"
        )

    if phase1b_filtered_ranker != 120:
        raise RuntimeError(
            f"phase1b filtered ranker rows expected 120, got {phase1b_filtered_ranker}"
        )

    if phase1b_filtered_classifier_labels != {"0": 70, "1": 50}:
        raise RuntimeError(
            f"phase1b filtered classifier labels mismatch: {phase1b_filtered_classifier_labels}"
        )

    if phase1b_filtered_ranker_labels != {"0": 70, "1": 24, "2": 26}:
        raise RuntimeError(
            f"phase1b filtered ranker labels mismatch: {phase1b_filtered_ranker_labels}"
        )

    result = {
        "event": "validated",
        "db": args.db,
        "counts": counts,
        "phase1b_filtered_classifier_rows": phase1b_filtered_classifier,
        "phase1b_filtered_ranker_rows": phase1b_filtered_ranker,
        "phase1b_filtered_classifier_labels": phase1b_filtered_classifier_labels,
        "phase1b_filtered_ranker_labels": phase1b_filtered_ranker_labels,
        "foreign_key_check": "ok",
        "score_status": "diagnostic_only",
        "db_role": "operational_source_of_truth",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
