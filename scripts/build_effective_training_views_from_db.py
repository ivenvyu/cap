from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


FILTER_RUN_ID = "flower_season_exclusion_filter_v1"
SCORE_STATUS = "diagnostic_only"


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name = ?
            """,
            (table,),
        ).fetchone()[0]
        > 0
    )


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(r["name"])
        for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def grouped_counts(
    conn: sqlite3.Connection,
    table: str,
    candidate_group_cols: list[str],
) -> list[dict[str, Any]]:
    cols = table_columns(conn, table)
    group_cols = [c for c in candidate_group_cols if c in cols]
    if not group_cols:
        return []

    group_expr = ", ".join(group_cols)

    rows = conn.execute(
        f"""
        SELECT {group_expr}, COUNT(*) AS n
        FROM {table}
        GROUP BY {group_expr}
        ORDER BY {group_expr}
        """
    ).fetchall()

    return [dict(r) for r in rows]


def require_columns(conn: sqlite3.Connection, table: str, required: set[str]) -> None:
    cols = table_columns(conn, table)
    missing = required - cols
    if missing:
        raise RuntimeError(f"{table} missing required columns: {sorted(missing)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--filter-run-id", default=FILTER_RUN_ID)
    ap.add_argument("--summary-out", default="audit/ontology/effective_training_views_v1.summary.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise RuntimeError(f"foreign_key_check failed: {[tuple(r) for r in fk[:10]]}")

    required_objects = [
        "training_set_items",
        "training_snapshots",
        "v_effective_pair_features_v1",
        "v_effective_retrieval_candidates_excluded_v1",
        "retrieval_candidate_filter_decisions",
    ]

    for obj in required_objects:
        if not table_exists(conn, obj):
            raise RuntimeError(f"missing required DB object: {obj}")

    require_columns(
        conn,
        "training_set_items",
        {"training_set_id", "training_snapshot_id", "item_order"},
    )
    require_columns(
        conn,
        "training_snapshots",
        {"training_snapshot_id", "campaign_id", "image_id", "label", "decision_label"},
    )
    require_columns(
        conn,
        "v_effective_pair_features_v1",
        {"campaign_id", "image_id"},
    )
    require_columns(
        conn,
        "v_effective_retrieval_candidates_excluded_v1",
        {"campaign_id", "image_id", "excluded_subject_name", "exclusion_reason"},
    )

    filter_rows = count(
        conn,
        f"""
        SELECT COUNT(*)
        FROM retrieval_candidate_filter_decisions
        WHERE filter_run_id = '{args.filter_run_id}'
        """,
    )
    if filter_rows == 0:
        raise RuntimeError(
            f"no filter decisions found for filter_run_id={args.filter_run_id}; "
            "run apply_flower_season_exclusions_to_retrieval_candidates.py first"
        )

    conn.execute("DROP VIEW IF EXISTS v_effective_training_set_items_v1")
    conn.execute("DROP VIEW IF EXISTS v_effective_training_snapshots_v1")
    conn.execute("DROP VIEW IF EXISTS v_training_set_items_excluded_by_flower_season_v1")
    conn.execute("DROP VIEW IF EXISTS v_training_snapshots_excluded_by_flower_season_v1")

    # training_set_items does not contain campaign_id/image_id directly.
    # It must be resolved through training_snapshots.
    conn.execute(
        """
        CREATE VIEW v_effective_training_set_items_v1 AS
        SELECT
            tsi.training_set_id,
            tsi.training_snapshot_id,
            tsi.item_order,
            ts.snapshot_kind,
            ts.snapshot_version,
            ts.campaign_id,
            ts.image_id,
            ts.pair_id,
            ts.feature_snapshot_id,
            ts.layout_spec_id,
            ts.label,
            ts.decision_label,
            ts.group_id,
            ts.label_status,
            ts.issue_tags_json,
            ts.artifact_id
        FROM training_set_items tsi
        JOIN training_snapshots ts
          ON ts.training_snapshot_id = tsi.training_snapshot_id
        WHERE EXISTS (
            SELECT 1
            FROM v_effective_pair_features_v1 pf
            WHERE pf.campaign_id = ts.campaign_id
              AND pf.image_id = ts.image_id
        )
        """
    )

    conn.execute(
        """
        CREATE VIEW v_effective_training_snapshots_v1 AS
        SELECT ts.*
        FROM training_snapshots ts
        WHERE EXISTS (
            SELECT 1
            FROM v_effective_pair_features_v1 pf
            WHERE pf.campaign_id = ts.campaign_id
              AND pf.image_id = ts.image_id
        )
        """
    )

    conn.execute(
        """
        CREATE VIEW v_training_set_items_excluded_by_flower_season_v1 AS
        SELECT
            tsi.training_set_id,
            tsi.training_snapshot_id,
            tsi.item_order,
            ts.snapshot_kind,
            ts.snapshot_version,
            ts.campaign_id,
            ts.image_id,
            ts.pair_id,
            ts.feature_snapshot_id,
            ts.layout_spec_id,
            ts.label,
            ts.decision_label,
            ts.group_id,
            ts.label_status,
            ts.issue_tags_json,
            ts.artifact_id,
            ex.excluded_subject_name,
            ex.exclusion_reason,
            ex.exclusion_source_id
        FROM training_set_items tsi
        JOIN training_snapshots ts
          ON ts.training_snapshot_id = tsi.training_snapshot_id
        JOIN v_effective_retrieval_candidates_excluded_v1 ex
          ON ex.campaign_id = ts.campaign_id
         AND ex.image_id = ts.image_id
        """
    )

    conn.execute(
        """
        CREATE VIEW v_training_snapshots_excluded_by_flower_season_v1 AS
        SELECT
            ts.*,
            ex.excluded_subject_name,
            ex.exclusion_reason,
            ex.exclusion_source_id
        FROM training_snapshots ts
        JOIN v_effective_retrieval_candidates_excluded_v1 ex
          ON ex.campaign_id = ts.campaign_id
         AND ex.image_id = ts.image_id
        """
    )

    conn.commit()

    summary = {
        "event": "done",
        "db": args.db,
        "filter_run_id": args.filter_run_id,
        "source_training_set_items": count(conn, "SELECT COUNT(*) FROM training_set_items"),
        "effective_training_set_items": count(conn, "SELECT COUNT(*) FROM v_effective_training_set_items_v1"),
        "excluded_training_set_items": count(conn, "SELECT COUNT(*) FROM v_training_set_items_excluded_by_flower_season_v1"),
        "source_training_snapshots": count(conn, "SELECT COUNT(*) FROM training_snapshots"),
        "effective_training_snapshots": count(conn, "SELECT COUNT(*) FROM v_effective_training_snapshots_v1"),
        "excluded_training_snapshots": count(conn, "SELECT COUNT(*) FROM v_training_snapshots_excluded_by_flower_season_v1"),
        "effective_training_set_items_by_group": grouped_counts(
            conn,
            "v_effective_training_set_items_v1",
            [
                "training_set_id",
                "snapshot_kind",
                "campaign_id",
                "label",
                "decision_label",
            ],
        ),
        "effective_training_snapshots_by_group": grouped_counts(
            conn,
            "v_effective_training_snapshots_v1",
            [
                "snapshot_kind",
                "snapshot_version",
                "campaign_id",
                "label",
                "decision_label",
            ],
        ),
        "excluded_training_items_by_campaign_subject": [
            dict(r)
            for r in conn.execute(
                """
                SELECT
                    campaign_id,
                    excluded_subject_name,
                    exclusion_reason,
                    COUNT(*) AS n
                FROM v_training_set_items_excluded_by_flower_season_v1
                GROUP BY campaign_id, excluded_subject_name, exclusion_reason
                ORDER BY campaign_id, excluded_subject_name
                """
            ).fetchall()
        ],
        "excluded_training_snapshots_by_campaign_subject": [
            dict(r)
            for r in conn.execute(
                """
                SELECT
                    campaign_id,
                    excluded_subject_name,
                    exclusion_reason,
                    COUNT(*) AS n
                FROM v_training_snapshots_excluded_by_flower_season_v1
                GROUP BY campaign_id, excluded_subject_name, exclusion_reason
                ORDER BY campaign_id, excluded_subject_name
                """
            ).fetchall()
        ],
        "views": [
            "v_effective_training_set_items_v1",
            "v_effective_training_snapshots_v1",
            "v_training_set_items_excluded_by_flower_season_v1",
            "v_training_snapshots_excluded_by_flower_season_v1",
        ],
        "score_status": SCORE_STATUS,
        "interpretation": (
            "Effective training views exclude campaign-image pairs removed by "
            "flower season exclusion. training_set_items are resolved through "
            "training_snapshots because they do not store campaign_id/image_id directly. "
            "This is not a calibrated model-quality threshold."
        ),
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
