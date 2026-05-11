from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--cluster-level", default="coarse", choices=["coarse", "mid", "fine"])
    ap.add_argument("--min-cluster-tag-assertions", type=int, default=1)
    ap.add_argument("--min-image-tag-assertions", type=int, default=1)
    args = ap.parse_args()

    conn = connect(Path(args.db))

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise RuntimeError(f"foreign_key_check failed: {[tuple(r) for r in fk[:10]]}")

    cluster_assertions = conn.execute(
        """
        SELECT
            a.assertion_id,
            a.cluster_level,
            a.cluster_id,
            a.tag_id,
            v.axis_id,
            v.tag_name,
            a.label_source,
            a.confidence_status
        FROM cluster_tag_assertions a
        JOIN tag_values v
          ON a.tag_id = v.tag_id
        WHERE a.cluster_level = ?
        ORDER BY a.cluster_id, a.tag_id
        """,
        (args.cluster_level,),
    ).fetchall()

    image_assertions = conn.execute(
        """
        SELECT
            a.assertion_id,
            a.image_id,
            a.tag_id,
            v.axis_id,
            v.tag_name,
            a.label_source,
            a.confidence_status,
            a.derived_from
        FROM image_tag_assertions a
        JOIN tag_values v
          ON a.tag_id = v.tag_id
        ORDER BY a.image_id, a.tag_id
        """
    ).fetchall()

    if len(cluster_assertions) < args.min_cluster_tag_assertions:
        raise RuntimeError(
            f"expected at least {args.min_cluster_tag_assertions} cluster tag assertions, "
            f"got {len(cluster_assertions)}"
        )

    if len(image_assertions) < args.min_image_tag_assertions:
        raise RuntimeError(
            f"expected at least {args.min_image_tag_assertions} image tag assertions, "
            f"got {len(image_assertions)}"
        )

    axis_cluster_counts = Counter(str(r["axis_id"]) for r in cluster_assertions)
    axis_image_counts = Counter(str(r["axis_id"]) for r in image_assertions)

    cluster_counts = Counter(str(r["cluster_id"]) for r in cluster_assertions)

    # Propagation consistency:
    # For each cluster_tag_assertion, expected propagated count equals cluster size.
    propagation_mismatches = []
    col = cluster_column(args.cluster_level)

    for a in cluster_assertions:
        cluster_size = int(
            conn.execute(
                f"SELECT COUNT(*) FROM image_clusters WHERE {col} = ?",
                (a["cluster_id"],),
            ).fetchone()[0]
        )

        propagated = int(
            conn.execute(
                """
                SELECT COUNT(*)
                FROM image_tag_assertions
                WHERE derived_from LIKE ?
                  AND tag_id = ?
                """,
                (f"%{a['assertion_id']}%", a["tag_id"]),
            ).fetchone()[0]
        )

        if propagated != cluster_size:
            propagation_mismatches.append(
                {
                    "cluster_assertion_id": a["assertion_id"],
                    "cluster_id": a["cluster_id"],
                    "tag_id": a["tag_id"],
                    "cluster_size": cluster_size,
                    "propagated": propagated,
                }
            )

    if propagation_mismatches:
        raise RuntimeError(
            "propagation mismatch: "
            + json.dumps(propagation_mismatches[:5], ensure_ascii=False)
        )

    result = {
        "event": "validated",
        "db": args.db,
        "cluster_level": args.cluster_level,
        "cluster_tag_assertions": len(cluster_assertions),
        "image_tag_assertions": len(image_assertions),
        "clusters_with_tags": len(cluster_counts),
        "cluster_assertions_by_axis": dict(sorted(axis_cluster_counts.items())),
        "image_assertions_by_axis": dict(sorted(axis_image_counts.items())),
        "foreign_key_check": "ok",
        "propagation_consistency": "ok",
        "score_status": "diagnostic_only",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
