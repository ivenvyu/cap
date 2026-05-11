from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
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


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name = ?
            """,
            (name,),
        ).fetchone()[0]
        > 0
    )


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fetch_rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    c = Counter(str(r.get(key)) for r in rows)
    return dict(sorted(c.items()))


def count_by_two(rows: list[dict[str, Any]], key1: str, key2: str) -> list[dict[str, Any]]:
    c: dict[tuple[str, str], int] = Counter(
        (str(r.get(key1)), str(r.get(key2)))
        for r in rows
    )

    return [
        {key1: k1, key2: k2, "n": n}
        for (k1, k2), n in sorted(c.items())
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--filter-run-id", default=FILTER_RUN_ID)
    ap.add_argument("--summary-out", default="audit/ontology/retrieval_filter_label_impact_v1.summary.json")
    ap.add_argument("--review-dir", default="data/review/ontology/retrieval_filter_label_impact_v1")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    required = [
        "retrieval_candidates",
        "v_effective_retrieval_candidates_v1",
        "v_effective_retrieval_candidates_excluded_v1",
        "training_snapshots",
    ]
    for obj in required:
        if not table_exists(conn, obj):
            raise RuntimeError(f"missing required DB object: {obj}")

    conflicts = fetch_rows(
        conn,
        """
        SELECT
            campaign_id,
            image_id,
            COUNT(DISTINCT decision_label) AS decision_labels,
            GROUP_CONCAT(DISTINCT decision_label) AS decision_label_values
        FROM training_snapshots
        WHERE snapshot_kind = 'classifier'
        GROUP BY campaign_id, image_id
        HAVING COUNT(DISTINCT decision_label) > 1
        ORDER BY campaign_id, image_id
        """
    )
    if conflicts:
        raise RuntimeError(
            "conflicting classifier labels found for campaign-image pairs; "
            f"examples={conflicts[:10]}"
        )

    conn.execute("DROP VIEW IF EXISTS v_classifier_pair_labels_v1")
    conn.execute(
        """
        CREATE TEMP VIEW v_classifier_pair_labels_v1 AS
        SELECT
            campaign_id,
            image_id,
            MIN(label) AS label,
            MIN(decision_label) AS decision_label
        FROM training_snapshots
        WHERE snapshot_kind = 'classifier'
        GROUP BY campaign_id, image_id
        """
    )

    original_labeled = fetch_rows(
        conn,
        """
        SELECT
            rc.campaign_id,
            rc.image_id,
            i.path,
            i.subject_name,
            l.label,
            l.decision_label
        FROM retrieval_candidates rc
        JOIN v_classifier_pair_labels_v1 l
          ON l.campaign_id = rc.campaign_id
         AND l.image_id = rc.image_id
        LEFT JOIN images i
          ON i.image_id = rc.image_id
        ORDER BY rc.campaign_id, rc.image_id
        """
    )

    effective_labeled = fetch_rows(
        conn,
        """
        SELECT
            rc.campaign_id,
            rc.image_id,
            i.path,
            i.subject_name,
            l.label,
            l.decision_label
        FROM v_effective_retrieval_candidates_v1 rc
        JOIN v_classifier_pair_labels_v1 l
          ON l.campaign_id = rc.campaign_id
         AND l.image_id = rc.image_id
        LEFT JOIN images i
          ON i.image_id = rc.image_id
        ORDER BY rc.campaign_id, rc.image_id
        """
    )

    excluded_labeled = fetch_rows(
        conn,
        """
        SELECT
            ex.campaign_id,
            ex.image_id,
            i.path,
            ex.excluded_subject_name AS subject_name,
            ex.exclusion_reason,
            ex.exclusion_source_id,
            l.label,
            l.decision_label
        FROM v_effective_retrieval_candidates_excluded_v1 ex
        LEFT JOIN v_classifier_pair_labels_v1 l
          ON l.campaign_id = ex.campaign_id
         AND l.image_id = ex.image_id
        LEFT JOIN images i
          ON i.image_id = ex.image_id
        ORDER BY ex.campaign_id, ex.excluded_subject_name, ex.image_id
        """
    )

    all_original_count = int(conn.execute("SELECT COUNT(*) FROM retrieval_candidates").fetchone()[0])
    all_effective_count = int(conn.execute("SELECT COUNT(*) FROM v_effective_retrieval_candidates_v1").fetchone()[0])
    all_excluded_count = int(conn.execute("SELECT COUNT(*) FROM v_effective_retrieval_candidates_excluded_v1").fetchone()[0])

    original_label_counts = count_by(original_labeled, "decision_label")
    effective_label_counts = count_by(effective_labeled, "decision_label")
    excluded_label_counts = count_by(excluded_labeled, "decision_label")

    harmful_exclusions = [
        r for r in excluded_labeled
        if r.get("decision_label") in {"accept", "acceptable"}
    ]

    unlabeled_exclusions = [
        r for r in excluded_labeled
        if r.get("decision_label") is None
    ]

    reject_exclusions = [
        r for r in excluded_labeled
        if r.get("decision_label") == "reject"
    ]

    review_dir = Path(args.review_dir)
    write_csv(review_dir / "excluded_labeled.csv", excluded_labeled)
    write_csv(review_dir / "harmful_exclusions_accept_or_acceptable.csv", harmful_exclusions)
    write_csv(review_dir / "original_labeled_candidates.csv", original_labeled)
    write_csv(review_dir / "effective_labeled_candidates.csv", effective_labeled)

    summary = {
        "event": "done",
        "db": args.db,
        "filter_run_id": args.filter_run_id,
        "score_status": SCORE_STATUS,
        "threshold_status": "diagnostic_only_not_calibrated",
        "source_candidate_counts": {
            "retrieval_candidates": all_original_count,
            "effective_retrieval_candidates": all_effective_count,
            "excluded_retrieval_candidates": all_excluded_count,
        },
        "labeled_candidate_counts": {
            "original_labeled": len(original_labeled),
            "effective_labeled": len(effective_labeled),
            "excluded_labeled": len(excluded_labeled),
        },
        "decision_label_counts": {
            "original": original_label_counts,
            "effective": effective_label_counts,
            "excluded": excluded_label_counts,
        },
        "excluded_by_campaign_subject": count_by_two(excluded_labeled, "campaign_id", "subject_name"),
        "excluded_by_campaign_decision": count_by_two(excluded_labeled, "campaign_id", "decision_label"),
        "safety_check": {
            "excluded_accept_or_acceptable": len(harmful_exclusions),
            "excluded_reject": len(reject_exclusions),
            "excluded_unlabeled": len(unlabeled_exclusions),
            "pass": len(harmful_exclusions) == 0,
        },
        "review_outputs": {
            "review_dir": str(review_dir),
            "excluded_labeled_csv": str(review_dir / "excluded_labeled.csv"),
            "harmful_exclusions_csv": str(review_dir / "harmful_exclusions_accept_or_acceptable.csv"),
        },
        "interpretation": (
            "This report checks whether flower-season exclusions removed reviewed reject candidates "
            "without removing accept/acceptable candidates. It evaluates retrieval filter precision, "
            "not classifier model quality."
        ),
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jdump(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
