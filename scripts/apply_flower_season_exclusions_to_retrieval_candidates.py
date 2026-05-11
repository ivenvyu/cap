from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


FILTER_RUN_ID = "flower_season_exclusion_filter_v1"
SCORE_STATUS = "diagnostic_only"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {str(r["name"]) for r in rows}


def candidate_key(row: dict[str, Any]) -> str:
    h = hashlib.sha1(jdump(row).encode("utf-8")).hexdigest()[:20]
    return f"retrieval_candidate_{h}"


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS retrieval_candidate_filter_decisions (
            filter_run_id TEXT NOT NULL,
            candidate_key TEXT NOT NULL,
            source_table TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            image_path TEXT,
            subject_name TEXT,
            retained INTEGER NOT NULL,
            exclusion_reason TEXT,
            exclusion_source_id TEXT,
            score_status TEXT NOT NULL,
            raw_candidate_json TEXT NOT NULL,
            raw_exclusion_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(filter_run_id, candidate_key)
        );

        CREATE INDEX IF NOT EXISTS idx_retrieval_candidate_filter_decisions_campaign
            ON retrieval_candidate_filter_decisions(filter_run_id, campaign_id);

        CREATE INDEX IF NOT EXISTS idx_retrieval_candidate_filter_decisions_image
            ON retrieval_candidate_filter_decisions(filter_run_id, image_id);

        CREATE INDEX IF NOT EXISTS idx_retrieval_candidate_filter_decisions_retained
            ON retrieval_candidate_filter_decisions(filter_run_id, retained);
        """
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "campaign_id",
        "image_id",
        "image_path",
        "subject_name",
        "retained",
        "exclusion_reason",
        "exclusion_source_id",
    ]

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--source-table", default="retrieval_candidates")
    ap.add_argument("--filter-run-id", default=FILTER_RUN_ID)
    ap.add_argument("--summary-out", default="audit/ontology/retrieval_candidates_flower_season_filter_v1.summary.json")
    ap.add_argument("--review-dir", default="data/review/ontology/retrieval_candidates_flower_season_filter_v1")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    ensure_table(conn)

    cols = table_columns(conn, args.source_table)
    required = {"campaign_id", "image_id"}
    missing = required - cols
    if missing:
        raise RuntimeError(f"{args.source_table} missing required columns: {sorted(missing)}")

    exclusion_rows = conn.execute(
        """
        SELECT
            campaign_id,
            image_id,
            requested_season,
            subject_name,
            plant_id,
            bloom_seasons_json,
            exclusion_reason,
            source_id,
            raw_json
        FROM campaign_image_flower_season_exclusions
        """
    ).fetchall()

    exclusions = {
        (str(r["campaign_id"]), str(r["image_id"])): dict(r)
        for r in exclusion_rows
    }

    image_meta = {
        str(r["image_id"]): dict(r)
        for r in conn.execute(
            """
            SELECT image_id, path, subject_name
            FROM images
            """
        ).fetchall()
    }

    candidates = conn.execute(
        f"""
        SELECT *
        FROM {args.source_table}
        ORDER BY campaign_id, image_id
        """
    ).fetchall()

    conn.execute(
        """
        DELETE FROM retrieval_candidate_filter_decisions
        WHERE filter_run_id = ?
        """,
        (args.filter_run_id,),
    )

    retained = 0
    excluded = 0
    retained_by_campaign: Counter[str] = Counter()
    excluded_by_campaign: Counter[str] = Counter()
    excluded_rows_out: list[dict[str, Any]] = []
    retained_rows_out: list[dict[str, Any]] = []

    now = utc_now()

    for row in candidates:
        raw_candidate = dict(row)
        campaign_id = str(raw_candidate["campaign_id"])
        image_id = str(raw_candidate["image_id"])
        key = candidate_key(raw_candidate)

        exclusion = exclusions.get((campaign_id, image_id))
        meta = image_meta.get(image_id, {})

        if exclusion:
            is_retained = 0
            excluded += 1
            excluded_by_campaign[campaign_id] += 1
            exclusion_reason = str(exclusion["exclusion_reason"])
            exclusion_source_id = str(exclusion["source_id"])
            raw_exclusion_json = str(exclusion["raw_json"])
        else:
            is_retained = 1
            retained += 1
            retained_by_campaign[campaign_id] += 1
            exclusion_reason = None
            exclusion_source_id = None
            raw_exclusion_json = None

        decision = {
            "campaign_id": campaign_id,
            "image_id": image_id,
            "image_path": meta.get("path"),
            "subject_name": meta.get("subject_name"),
            "retained": is_retained,
            "exclusion_reason": exclusion_reason,
            "exclusion_source_id": exclusion_source_id,
        }

        if is_retained:
            retained_rows_out.append(decision)
        else:
            excluded_rows_out.append(decision)

        conn.execute(
            """
            INSERT OR REPLACE INTO retrieval_candidate_filter_decisions
            (filter_run_id, candidate_key, source_table, campaign_id, image_id,
             image_path, subject_name, retained, exclusion_reason, exclusion_source_id,
             score_status, raw_candidate_json, raw_exclusion_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                args.filter_run_id,
                key,
                args.source_table,
                campaign_id,
                image_id,
                meta.get("path"),
                meta.get("subject_name"),
                is_retained,
                exclusion_reason,
                exclusion_source_id,
                SCORE_STATUS,
                jdump(raw_candidate),
                raw_exclusion_json,
                now,
            ),
        )

    # Views for downstream SQL.
    conn.execute("DROP VIEW IF EXISTS v_retrieval_candidates_after_flower_season_exclusion_v1")
    conn.execute("DROP VIEW IF EXISTS v_retrieval_candidates_excluded_by_flower_season_v1")

    conn.execute(
        f"""
        CREATE VIEW v_retrieval_candidates_after_flower_season_exclusion_v1 AS
        SELECT rc.*
        FROM {args.source_table} rc
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
        CREATE VIEW v_retrieval_candidates_excluded_by_flower_season_v1 AS
        SELECT
            rc.*,
            d.image_path AS excluded_image_path,
            d.subject_name AS excluded_subject_name,
            d.exclusion_reason,
            d.exclusion_source_id
        FROM {args.source_table} rc
        JOIN retrieval_candidate_filter_decisions d
          ON d.filter_run_id = '{args.filter_run_id}'
         AND d.campaign_id = rc.campaign_id
         AND d.image_id = rc.image_id
         AND d.retained = 0
        """
    )

    conn.commit()

    review_dir = Path(args.review_dir)
    write_csv(review_dir / "excluded.csv", excluded_rows_out)
    write_csv(review_dir / "retained.csv", retained_rows_out)

    summary = {
        "event": "done",
        "db": args.db,
        "source_table": args.source_table,
        "filter_run_id": args.filter_run_id,
        "source_candidates": len(candidates),
        "retained_candidates": retained,
        "excluded_candidates": excluded,
        "retained_by_campaign": dict(sorted(retained_by_campaign.items())),
        "excluded_by_campaign": dict(sorted(excluded_by_campaign.items())),
        "views": [
            "v_retrieval_candidates_after_flower_season_exclusion_v1",
            "v_retrieval_candidates_excluded_by_flower_season_v1",
        ],
        "review_dir": str(review_dir),
        "excluded_csv": str(review_dir / "excluded.csv"),
        "retained_csv": str(review_dir / "retained.csv"),
        "score_status": SCORE_STATUS,
        "rule": (
            "If campaign has a season and candidate image has a known flower subject "
            "whose bloom seasons do not include that season, exclude that campaign-image pair."
        ),
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jdump(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
