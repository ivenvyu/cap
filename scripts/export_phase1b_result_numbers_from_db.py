from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Any


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> int:
    return int(conn.execute(sql, params).fetchone()[0])


def grouped_counts(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> dict[str, int]:
    return {str(r[0]): int(r[1]) for r in conn.execute(sql, params).fetchall()}


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def add(rows: list[dict[str, Any]], section: str, metric: str, value: Any) -> None:
    if isinstance(value, (dict, list)):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True)
    rows.append({"section": section, "metric": metric, "value": value})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--smoke-report", default="audit/phase_1b/phase_1b_classifier_smoke_report_db.json")
    ap.add_argument("--claim-report", default="audit/phase_1b/phase_1b_diagnostic_claim_support_report_db.json")
    ap.add_argument("--out-json", default="audit/phase_1b/phase_1b_result_numbers_db.json")
    ap.add_argument("--out-csv", default="audit/phase_1b/phase_1b_result_numbers_db.csv")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    smoke = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
    claim = json.loads(Path(args.claim_report).read_text(encoding="utf-8"))

    first_round_decisions = grouped_counts(
        conn,
        """
        SELECT decision_label, COUNT(*)
        FROM review_events
        WHERE campaign_id LIKE 'phase1b_%'
        GROUP BY decision_label
        ORDER BY decision_label
        """,
    )

    issue_counts: dict[str, int] = {}
    for row in conn.execute(
        """
        SELECT issue_tags_json
        FROM review_events
        WHERE campaign_id LIKE 'phase1b_%'
        """
    ):
        for tag in json.loads(row["issue_tags_json"] or "[]"):
            issue_counts[tag] = issue_counts.get(tag, 0) + 1

    bucket_counts = grouped_counts(
        conn,
        """
        SELECT source_bucket, COUNT(*)
        FROM review_events
        WHERE campaign_id LIKE 'phase1b_%'
        GROUP BY source_bucket
        ORDER BY source_bucket
        """,
    )

    classifier_labels = grouped_counts(
        conn,
        """
        SELECT t.label, COUNT(*)
        FROM training_set_items i
        JOIN training_snapshots t
          ON i.training_snapshot_id = t.training_snapshot_id
        WHERE i.training_set_id = 'phase1b_filtered_classifier_v1'
        GROUP BY t.label
        ORDER BY t.label
        """,
    )

    ranker_labels = grouped_counts(
        conn,
        """
        SELECT t.label, COUNT(*)
        FROM training_set_items i
        JOIN training_snapshots t
          ON i.training_snapshot_id = t.training_snapshot_id
        WHERE i.training_set_id = 'phase1b_filtered_ranker_v1'
        GROUP BY t.label
        ORDER BY t.label
        """,
    )

    campaign_counts = grouped_counts(
        conn,
        """
        SELECT campaign_id, COUNT(*)
        FROM review_events
        WHERE campaign_id LIKE 'phase1b_%'
        GROUP BY campaign_id
        ORDER BY campaign_id
        """,
    )

    numbers = {
        "metadata": {
            "phase": "phase_1b",
            "result_file": "phase_1b_result_numbers_db",
            "db_role": "operational_source_of_truth",
            "score_status": "diagnostic_only",
            "threshold_status": "no_calibrated_threshold",
            "candidate_support_explanation_status": "deferred_from_phase_1b",
            "claim_level_diagnostic_support_status": "executed",
        },
        "db_counts": {
            "images": scalar(conn, "SELECT COUNT(*) FROM images"),
            "image_embeddings": scalar(conn, "SELECT COUNT(*) FROM image_embeddings"),
            "image_duplicates": scalar(conn, "SELECT COUNT(*) FROM image_duplicates"),
            "image_clusters": scalar(conn, "SELECT COUNT(*) FROM image_clusters"),
            "image_regions": scalar(conn, "SELECT COUNT(*) FROM image_regions"),
            "campaigns": scalar(conn, "SELECT COUNT(*) FROM campaigns"),
            "retrieval_candidates": scalar(conn, "SELECT COUNT(*) FROM retrieval_candidates"),
            "pair_features": scalar(conn, "SELECT COUNT(*) FROM pair_features"),
            "review_events": scalar(conn, "SELECT COUNT(*) FROM review_events"),
            "training_snapshots": scalar(conn, "SELECT COUNT(*) FROM training_snapshots"),
            "training_sets": scalar(conn, "SELECT COUNT(*) FROM training_sets"),
            "training_set_items": scalar(conn, "SELECT COUNT(*) FROM training_set_items"),
            "tag_axes": scalar(conn, "SELECT COUNT(*) FROM tag_axes"),
            "tag_values": scalar(conn, "SELECT COUNT(*) FROM tag_values"),
        },
        "phase1b_first_round": {
            "review_events": scalar(conn, "SELECT COUNT(*) FROM review_events WHERE campaign_id LIKE 'phase1b_%'"),
            "decision_label_counts": first_round_decisions,
            "issue_tag_counts": issue_counts,
            "bucket_counts": bucket_counts,
            "campaign_counts": campaign_counts,
        },
        "phase1b_filtered_training_sets": {
            "classifier_rows": scalar(
                conn,
                "SELECT COUNT(*) FROM training_set_items WHERE training_set_id = 'phase1b_filtered_classifier_v1'",
            ),
            "classifier_label_counts": classifier_labels,
            "ranker_rows": scalar(
                conn,
                "SELECT COUNT(*) FROM training_set_items WHERE training_set_id = 'phase1b_filtered_ranker_v1'",
            ),
            "ranker_label_counts": ranker_labels,
        },
        "classifier_smoke_db": {
            "rows": smoke["dataset"]["rows"],
            "feature_count": smoke["dataset"]["feature_count"],
            "label_counts": smoke["dataset"]["label_counts"],
            "out_of_fold_accuracy": smoke["leave_one_campaign_out"]["out_of_fold_metrics"]["accuracy"],
            "out_of_fold_balanced_accuracy": smoke["leave_one_campaign_out"]["out_of_fold_metrics"]["balanced_accuracy"],
            "out_of_fold_roc_auc": smoke["leave_one_campaign_out"]["out_of_fold_metrics"]["roc_auc"],
            "out_of_fold_average_precision": smoke["leave_one_campaign_out"]["out_of_fold_metrics"]["average_precision"],
            "confusion_matrix": smoke["leave_one_campaign_out"]["out_of_fold_metrics"]["confusion_matrix"],
        },
        "diagnostic_claim_support_db": {
            "support_level": claim["metadata"]["support_level"],
            "report_status": claim["metadata"]["report_status"],
            "evaluation_counts": claim["evaluation_counts"],
            "future_observations": len(claim["future_observation_queue"]),
            "claim_evaluation_status": {
                claim_id: ev["evaluation_status"]
                for claim_id, ev in claim["claim_evaluations"].items()
            },
        },
    }

    write_json(Path(args.out_json), numbers)

    rows: list[dict[str, Any]] = []
    for section, metrics in numbers.items():
        for k, v in metrics.items():
            add(rows, section, k, v)
    write_csv(Path(args.out_csv), rows)

    print(json.dumps({
        "event": "done",
        "out_json": args.out_json,
        "out_csv": args.out_csv,
        "db_role": "operational_source_of_truth",
        "score_status": "diagnostic_only",
        "phase1b_review_events": numbers["phase1b_first_round"]["review_events"],
        "classifier_rows": numbers["phase1b_filtered_training_sets"]["classifier_rows"],
        "ranker_rows": numbers["phase1b_filtered_training_sets"]["ranker_rows"],
        "claim_support": numbers["diagnostic_claim_support_db"]["evaluation_counts"],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
