from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AXIS_COLUMNS = {
    "space_axis_tags": "space_axis",
    "temporal_axis_tags": "temporal_axis",
    "weather_light_axis_tags": "weather_light_axis",
    "subject_axis_tags": "subject_axis",
    "mood_axis_tags": "mood_axis",
    "usage_axis_tags": "usage_axis",
    "design_affordance_axis_tags": "design_affordance_axis",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise RuntimeError(f"missing label csv: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def split_tags(value: str | None) -> list[str]:
    if not value:
        return []

    text = value.strip()
    if not text:
        return []

    for sep in ["|", ";", "\n"]:
        text = text.replace(sep, ",")

    tags = [t.strip() for t in text.split(",") if t.strip()]
    return tags


def load_tag_lookup(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    rows = conn.execute(
        """
        SELECT axis_id, tag_id, tag_name
        FROM tag_values
        WHERE status = 'active'
        """
    ).fetchall()

    lookup: dict[str, dict[str, str]] = {}

    for r in rows:
        axis_id = str(r["axis_id"])
        lookup.setdefault(axis_id, {})
        lookup[axis_id][str(r["tag_name"])] = str(r["tag_id"])
        lookup[axis_id][str(r["tag_id"])] = str(r["tag_id"])

    return lookup


def require_queue_row(conn: sqlite3.Connection, queue_id: str) -> sqlite3.Row:
    row = conn.execute(
        """
        SELECT *
        FROM cluster_label_queue
        WHERE queue_id = ?
        """,
        (queue_id,),
    ).fetchone()

    if row is None:
        raise RuntimeError(f"label CSV references missing queue_id: {queue_id}")

    return row


def assertion_id_for(queue_id: str, tag_id: str) -> str:
    h = hashlib.sha1(f"{queue_id}:{tag_id}".encode("utf-8")).hexdigest()[:16]
    return f"cluster_tag_{h}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument(
        "--label-csv",
        default="data/review/ontology/cluster_label_queue_v1/cluster_label_queue__coarse.csv",
    )
    ap.add_argument("--annotator-id", default="human_001")
    ap.add_argument("--label-source", default="human_cluster_label")
    ap.add_argument("--confidence-status", default="diagnostic_cluster_label")
    ap.add_argument("--allow-partial", action="store_true", default=True)
    args = ap.parse_args()

    conn = connect(Path(args.db))
    rows = read_csv_rows(Path(args.label_csv))
    tag_lookup = load_tag_lookup(conn)

    inserted = 0
    skipped_pending = 0
    labeled_clusters = set()
    errors: list[str] = []

    for row_idx, row in enumerate(rows, start=1):
        queue_id = (row.get("queue_id") or "").strip()
        if not queue_id:
            errors.append(f"row {row_idx}: missing queue_id")
            continue

        queue = require_queue_row(conn, queue_id)

        confidence_status = (row.get("confidence_status") or "").strip()

        if confidence_status == "assistant_suggested_unverified":
            skipped_pending += 1
            continue

        if not confidence_status or confidence_status == "human_cluster_label_pending":
            # If all tag columns are empty, this is a normal pending row.
            if all(not split_tags(row.get(col)) for col in AXIS_COLUMNS):
                skipped_pending += 1
                continue

            # Tags exist but confidence is still pending. Use CLI default.
            confidence_status = args.confidence_status

        notes = (row.get("notes") or "").strip()

        row_has_tag = False

        for col, axis_id in AXIS_COLUMNS.items():
            raw_tags = split_tags(row.get(col))
            if not raw_tags:
                continue

            row_has_tag = True

            for raw_tag in raw_tags:
                tag_id = tag_lookup.get(axis_id, {}).get(raw_tag)
                if tag_id is None:
                    errors.append(
                        f"row {row_idx} queue_id={queue_id}: unknown tag '{raw_tag}' for axis {axis_id}"
                    )
                    continue

                assertion_id = assertion_id_for(queue_id, tag_id)

                conn.execute(
                    """
                    INSERT OR REPLACE INTO cluster_tag_assertions
                    (assertion_id, cluster_version, cluster_level, cluster_id, tag_id,
                     label_source, confidence_status, annotator_id, notes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        assertion_id,
                        queue["cluster_version"],
                        queue["cluster_level"],
                        queue["cluster_id"],
                        tag_id,
                        args.label_source,
                        confidence_status,
                        args.annotator_id,
                        notes,
                        utc_now(),
                    ),
                )
                inserted += 1
                labeled_clusters.add(queue_id)

        if not row_has_tag:
            skipped_pending += 1

    if errors:
        for e in errors[:20]:
            print("ERROR:", e)
        raise RuntimeError(f"failed with {len(errors)} tag validation errors")

    conn.commit()

    result = {
        "event": "done",
        "db": args.db,
        "label_csv": args.label_csv,
        "rows_read": len(rows),
        "cluster_tag_assertions_inserted": inserted,
        "labeled_clusters": len(labeled_clusters),
        "skipped_pending_rows": skipped_pending,
        "label_source": args.label_source,
        "confidence_status_default": args.confidence_status,
        "score_status": "diagnostic_only",
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
