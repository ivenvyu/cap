from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


FILTER_RUN_ID = "flower_season_exclusion_filter_v1"
SCORE_STATUS = "diagnostic_only"


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(r["name"])
        for r in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }


def count(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--filter-run-id", default=FILTER_RUN_ID)
    ap.add_argument("--summary-out", default="audit/ontology/effective_retrieval_views_v1.summary.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise RuntimeError(f"foreign_key_check failed: {[tuple(r) for r in fk[:10]]}")

    retrieval_cols = table_columns(conn, "retrieval_candidates")
    pair_cols = table_columns(conn, "pair_features")

    for table_name, cols in [
        ("retrieval_candidates", retrieval_cols),
        ("pair_features", pair_cols),
    ]:
        missing = {"campaign_id", "image_id"} - cols
        if missing:
            raise RuntimeError(f"{table_name} missing required columns: {sorted(missing)}")

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

    conn.execute("DROP VIEW IF EXISTS v_effective_retrieval_candidates_v1")
    conn.execute("DROP VIEW IF EXISTS v_effective_pair_features_v1")
    conn.execute("DROP VIEW IF EXISTS v_effective_retrieval_candidates_excluded_v1")

    conn.execute(
        f"""
        CREATE VIEW v_effective_retrieval_candidates_v1 AS
        SELECT rc.*
        FROM retrieval_candidates rc
        WHERE NOT EXISTS (
            SELECT 1
            FROM retrieval_candidate_filter_decisions d
            WHERE d.filter_run_id = '{args.filter_run_id}'
              AND d.campaign_id = rc.campaign_id
              AND d.image_id = rc.image_id
              AND d.retained = 0
        )
        """
    )

    conn.execute(
        f"""
        CREATE VIEW v_effective_pair_features_v1 AS
        SELECT pf.*
        FROM pair_features pf
        WHERE EXISTS (
            SELECT 1
            FROM v_effective_retrieval_candidates_v1 rc
            WHERE rc.campaign_id = pf.campaign_id
              AND rc.image_id = pf.image_id
        )
        """
    )

    conn.execute(
        f"""
        CREATE VIEW v_effective_retrieval_candidates_excluded_v1 AS
        SELECT
            rc.*,
            d.subject_name AS excluded_subject_name,
            d.exclusion_reason,
            d.exclusion_source_id
        FROM retrieval_candidates rc
        JOIN retrieval_candidate_filter_decisions d
          ON d.filter_run_id = '{args.filter_run_id}'
         AND d.campaign_id = rc.campaign_id
         AND d.image_id = rc.image_id
         AND d.retained = 0
        """
    )

    conn.commit()

    source_retrieval = count(conn, "SELECT COUNT(*) FROM retrieval_candidates")
    effective_retrieval = count(conn, "SELECT COUNT(*) FROM v_effective_retrieval_candidates_v1")
    excluded_retrieval = count(conn, "SELECT COUNT(*) FROM v_effective_retrieval_candidates_excluded_v1")

    source_pair_features = count(conn, "SELECT COUNT(*) FROM pair_features")
    effective_pair_features = count(conn, "SELECT COUNT(*) FROM v_effective_pair_features_v1")

    by_campaign = [
        dict(r)
        for r in conn.execute(
            """
            SELECT
                rc.campaign_id,
                COUNT(*) AS retained_candidates
            FROM v_effective_retrieval_candidates_v1 rc
            GROUP BY rc.campaign_id
            ORDER BY rc.campaign_id
            """
        ).fetchall()
    ]

    excluded_by_campaign = [
        dict(r)
        for r in conn.execute(
            """
            SELECT
                campaign_id,
                excluded_subject_name,
                exclusion_reason,
                COUNT(*) AS n
            FROM v_effective_retrieval_candidates_excluded_v1
            GROUP BY campaign_id, excluded_subject_name, exclusion_reason
            ORDER BY campaign_id, excluded_subject_name
            """
        ).fetchall()
    ]

    summary = {
        "event": "done",
        "db": args.db,
        "filter_run_id": args.filter_run_id,
        "source_retrieval_candidates": source_retrieval,
        "effective_retrieval_candidates": effective_retrieval,
        "excluded_retrieval_candidates": excluded_retrieval,
        "source_pair_features": source_pair_features,
        "effective_pair_features": effective_pair_features,
        "retained_by_campaign": by_campaign,
        "excluded_by_campaign_subject": excluded_by_campaign,
        "views": [
            "v_effective_retrieval_candidates_v1",
            "v_effective_pair_features_v1",
            "v_effective_retrieval_candidates_excluded_v1",
        ],
        "score_status": SCORE_STATUS,
        "interpretation": (
            "Effective retrieval views apply campaign-specific flower season exclusions. "
            "This is not a calibrated quality threshold."
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
