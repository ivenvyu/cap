from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def cluster_column(level: str) -> str:
    mapping = {
        "coarse": "dinov2_cluster_id_coarse",
        "mid": "dinov2_cluster_id_mid",
        "fine": "dinov2_cluster_id_fine",
    }
    if level not in mapping:
        raise RuntimeError(f"unsupported cluster level: {level}")
    return mapping[level]


def assertion_id_for(image_id: str, tag_id: str, cluster_assertion_id: str) -> str:
    h = hashlib.sha1(
        f"{image_id}:{tag_id}:{cluster_assertion_id}".encode("utf-8")
    ).hexdigest()[:16]
    return f"image_tag_{h}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--cluster-level", default="coarse", choices=["coarse", "mid", "fine"])
    ap.add_argument("--label-source", default="human_cluster_label_propagation")
    ap.add_argument("--confidence-status", default="propagated_from_cluster_diagnostic")
    ap.add_argument("--reset-propagated-for-level", action="store_true", default=True)
    args = ap.parse_args()

    conn = connect(Path(args.db))

    if args.reset_propagated_for_level:
        conn.execute(
            """
            DELETE FROM image_tag_assertions
            WHERE label_source = ?
              AND derived_from LIKE ?
            """,
            (args.label_source, f"%\"cluster_level\":\"{args.cluster_level}\"%"),
        )

    assertions = conn.execute(
        """
        SELECT *
        FROM cluster_tag_assertions
        WHERE cluster_level = ?
        ORDER BY cluster_id, tag_id
        """,
        (args.cluster_level,),
    ).fetchall()

    inserted = 0
    clusters_seen = set()

    for a in assertions:
        col = cluster_column(str(a["cluster_level"]))

        members = conn.execute(
            f"""
            SELECT image_id
            FROM image_clusters
            WHERE {col} = ?
            ORDER BY image_id
            """,
            (a["cluster_id"],),
        ).fetchall()

        clusters_seen.add(str(a["cluster_id"]))

        for m in members:
            image_id = str(m["image_id"])
            tag_id = str(a["tag_id"])
            cluster_assertion_id = str(a["assertion_id"])
            image_assertion_id = assertion_id_for(image_id, tag_id, cluster_assertion_id)

            derived_from = {
                "source": "cluster_tag_assertion",
                "cluster_assertion_id": cluster_assertion_id,
                "cluster_level": str(a["cluster_level"]),
                "cluster_id": str(a["cluster_id"]),
                "cluster_version": str(a["cluster_version"]),
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO image_tag_assertions
                (assertion_id, image_id, tag_id, label_source, confidence_status,
                 derived_from, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_assertion_id,
                    image_id,
                    tag_id,
                    args.label_source,
                    args.confidence_status,
                    json.dumps(derived_from, ensure_ascii=False, sort_keys=True),
                    f"propagated from {cluster_assertion_id}",
                    utc_now(),
                ),
            )
            inserted += 1

    conn.commit()

    print(json.dumps({
        "event": "done",
        "db": args.db,
        "cluster_level": args.cluster_level,
        "cluster_tag_assertions": len(assertions),
        "clusters_with_assertions": len(clusters_seen),
        "image_tag_assertions_inserted": inserted,
        "label_source": args.label_source,
        "confidence_status": args.confidence_status,
        "score_status": "diagnostic_only",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
