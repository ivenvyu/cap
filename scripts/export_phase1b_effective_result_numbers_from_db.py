from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


SCORE_STATUS = "diagnostic_only"
THRESHOLD_STATUS = "no_calibrated_threshold"


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


def count(conn: sqlite3.Connection, table_or_view: str) -> int:
    if not table_exists(conn, table_or_view):
        return 0
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_or_view}").fetchone()[0])


def fetch_one_value(conn: sqlite3.Connection, sql: str) -> int:
    return int(conn.execute(sql).fetchone()[0])


def fetch_rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def label_counts(conn: sqlite3.Connection, table: str, where: str = "1=1") -> dict[str, int]:
    if not table_exists(conn, table):
        return {}

    rows = fetch_rows(
        conn,
        f"""
        SELECT label, COUNT(*) AS n
        FROM {table}
        WHERE {where}
        GROUP BY label
        ORDER BY label
        """,
    )
    return {str(r["label"]): int(r["n"]) for r in rows}


def decision_counts(conn: sqlite3.Connection, table: str, where: str = "1=1") -> dict[str, int]:
    if not table_exists(conn, table):
        return {}

    rows = fetch_rows(
        conn,
        f"""
        SELECT decision_label, COUNT(*) AS n
        FROM {table}
        WHERE {where}
        GROUP BY decision_label
        ORDER BY decision_label
        """,
    )
    return {str(r["decision_label"]): int(r["n"]) for r in rows}


def read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_classifier_pair_label_view(conn: sqlite3.Connection) -> None:
    conn.execute("DROP VIEW IF EXISTS temp.v_classifier_pair_labels_effective_export")
    conn.execute(
        """
        CREATE TEMP VIEW v_classifier_pair_labels_effective_export AS
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


def retrieval_label_impact(conn: sqlite3.Connection) -> dict[str, Any]:
    build_classifier_pair_label_view(conn)

    original = fetch_rows(
        conn,
        """
        SELECT
            rc.campaign_id,
            rc.image_id,
            l.label,
            l.decision_label
        FROM retrieval_candidates rc
        JOIN v_classifier_pair_labels_effective_export l
          ON l.campaign_id = rc.campaign_id
         AND l.image_id = rc.image_id
        """
    )

    effective = fetch_rows(
        conn,
        """
        SELECT
            rc.campaign_id,
            rc.image_id,
            l.label,
            l.decision_label
        FROM v_effective_retrieval_candidates_v1 rc
        JOIN v_classifier_pair_labels_effective_export l
          ON l.campaign_id = rc.campaign_id
         AND l.image_id = rc.image_id
        """
    )

    excluded = fetch_rows(
        conn,
        """
        SELECT
            ex.campaign_id,
            ex.image_id,
            ex.excluded_subject_name,
            ex.exclusion_reason,
            l.label,
            l.decision_label
        FROM v_effective_retrieval_candidates_excluded_v1 ex
        LEFT JOIN v_classifier_pair_labels_effective_export l
          ON l.campaign_id = ex.campaign_id
         AND l.image_id = ex.image_id
        ORDER BY ex.campaign_id, ex.excluded_subject_name, ex.image_id
        """
    )

    def count_decisions(rows: list[dict[str, Any]]) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in rows:
            key = str(r.get("decision_label"))
            out[key] = out.get(key, 0) + 1
        return dict(sorted(out.items()))

    harmful = [
        r for r in excluded
        if r.get("decision_label") in {"accept", "acceptable"}
    ]
    reject = [
        r for r in excluded
        if r.get("decision_label") == "reject"
    ]
    unlabeled = [
        r for r in excluded
        if r.get("decision_label") is None
    ]

    return {
        "labeled_candidate_counts": {
            "original_labeled": len(original),
            "effective_labeled": len(effective),
            "excluded_candidates_total": len(excluded),
        },
        "decision_label_counts": {
            "original": count_decisions(original),
            "effective": count_decisions(effective),
            "excluded": count_decisions(excluded),
        },
        "safety_check": {
            "excluded_accept_or_acceptable": len(harmful),
            "excluded_reject": len(reject),
            "excluded_unlabeled": len(unlabeled),
            "pass": len(harmful) == 0,
        },
        "excluded_examples": excluded[:30],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument(
        "--query-smoke-report",
        default="audit/ontology/ontology_effective_search_index_query_smoke_v1.summary.json",
    )
    ap.add_argument(
        "--search-index-summary",
        default="audit/ontology/ontology_effective_search_index_v1.summary.json",
    )
    ap.add_argument(
        "--pipeline-summary",
        default="audit/ontology/phase1b_effective_retrieval_pipeline_run_v1.summary.json",
    )
    ap.add_argument(
        "--out",
        default="audit/phase_1b/phase_1b_effective_result_numbers_db.json",
    )
    args = ap.parse_args()

    conn = connect(Path(args.db))

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise RuntimeError(f"foreign_key_check failed: {[tuple(r) for r in fk[:10]]}")

    required_views = [
        "v_effective_retrieval_candidates_v1",
        "v_effective_retrieval_candidates_excluded_v1",
        "v_effective_pair_features_v1",
        "v_effective_training_set_items_v1",
        "v_effective_training_snapshots_v1",
    ]
    missing = [v for v in required_views if not table_exists(conn, v)]
    if missing:
        raise RuntimeError(f"missing required effective views: {missing}")

    query_smoke = read_json_if_exists(Path(args.query_smoke_report))
    search_index_summary = read_json_if_exists(Path(args.search_index_summary))
    pipeline_summary = read_json_if_exists(Path(args.pipeline_summary))

    result = {
        "metadata": {
            "report_status": "phase1b_effective_result_numbers_db_first",
            "db": args.db,
            "db_role": "operational_source_of_truth",
            "score_status": SCORE_STATUS,
            "threshold_status": THRESHOLD_STATUS,
            "interpretation": (
                "Phase 1b effective retrieval 결과 숫자를 DB와 audit 산출물에서 모은 요약이다. "
                "꽃 개화시기 제외 규칙은 classifier 성능 기준이 아니라 retrieval pre-filter 기준으로 평가한다."
            ),
        },
        "db_counts": {
            "images": count(conn, "images"),
            "campaigns": count(conn, "campaigns"),
            "retrieval_candidates": count(conn, "retrieval_candidates"),
            "pair_features": count(conn, "pair_features"),
            "review_events": count(conn, "review_events"),
            "training_snapshots": count(conn, "training_snapshots"),
            "training_sets": count(conn, "training_sets"),
            "training_set_items": count(conn, "training_set_items"),
            "cluster_label_queue": count(conn, "cluster_label_queue"),
            "cluster_tag_assertions": count(conn, "cluster_tag_assertions"),
            "image_tag_assertions": count(conn, "image_tag_assertions"),
            "visual_cues": count(conn, "visual_cues"),
            "campaign_visual_cue_requirements": count(conn, "campaign_visual_cue_requirements"),
            "campaign_image_cue_scores": count(conn, "campaign_image_cue_scores"),
            "plant_entities": count(conn, "plant_entities"),
            "plant_names": count(conn, "plant_names"),
            "plant_bloom_priors": count(conn, "plant_bloom_priors"),
            "campaign_image_botanical_bloom_priors": count(conn, "campaign_image_botanical_bloom_priors"),
            "campaign_image_flower_season_exclusions": count(conn, "campaign_image_flower_season_exclusions"),
            "retrieval_candidate_filter_decisions": count(conn, "retrieval_candidate_filter_decisions"),
        },
        "retrieval_effective": {
            "original_retrieval_candidates": count(conn, "retrieval_candidates"),
            "effective_retrieval_candidates": count(conn, "v_effective_retrieval_candidates_v1"),
            "excluded_retrieval_candidates": count(conn, "v_effective_retrieval_candidates_excluded_v1"),
            "original_pair_features": count(conn, "pair_features"),
            "effective_pair_features": count(conn, "v_effective_pair_features_v1"),
            "effective_by_campaign": fetch_rows(
                conn,
                """
                SELECT campaign_id, COUNT(*) AS n
                FROM v_effective_retrieval_candidates_v1
                GROUP BY campaign_id
                ORDER BY campaign_id
                """,
            ),
            "excluded_by_campaign_subject": fetch_rows(
                conn,
                """
                SELECT
                    campaign_id,
                    excluded_subject_name,
                    exclusion_reason,
                    COUNT(*) AS n
                FROM v_effective_retrieval_candidates_excluded_v1
                GROUP BY campaign_id, excluded_subject_name, exclusion_reason
                ORDER BY campaign_id, excluded_subject_name
                """,
            ),
        },
        "retrieval_label_impact": retrieval_label_impact(conn),
        "effective_training": {
            "source_training_set_items": count(conn, "training_set_items"),
            "effective_training_set_items": count(conn, "v_effective_training_set_items_v1"),
            "excluded_training_set_items": count(conn, "v_training_set_items_excluded_by_flower_season_v1"),
            "source_training_snapshots": count(conn, "training_snapshots"),
            "effective_training_snapshots": count(conn, "v_effective_training_snapshots_v1"),
            "excluded_training_snapshots": count(conn, "v_training_snapshots_excluded_by_flower_season_v1"),
            "classifier_rows": fetch_one_value(
                conn,
                """
                SELECT COUNT(*)
                FROM v_effective_training_set_items_v1
                WHERE training_set_id = 'phase1b_filtered_classifier_v1'
                """
            ),
            "classifier_label_counts": label_counts(
                conn,
                "v_effective_training_set_items_v1",
                "training_set_id = 'phase1b_filtered_classifier_v1'",
            ),
            "classifier_decision_label_counts": decision_counts(
                conn,
                "v_effective_training_set_items_v1",
                "training_set_id = 'phase1b_filtered_classifier_v1'",
            ),
            "ranker_rows": fetch_one_value(
                conn,
                """
                SELECT COUNT(*)
                FROM v_effective_training_set_items_v1
                WHERE training_set_id = 'phase1b_filtered_ranker_v1'
                """
            ),
            "ranker_label_counts": label_counts(
                conn,
                "v_effective_training_set_items_v1",
                "training_set_id = 'phase1b_filtered_ranker_v1'",
            ),
            "ranker_decision_label_counts": decision_counts(
                conn,
                "v_effective_training_set_items_v1",
                "training_set_id = 'phase1b_filtered_ranker_v1'",
            ),
        },
        "search_index": {
            "summary_present": search_index_summary is not None,
            "images_exported": None if search_index_summary is None else search_index_summary.get("images_exported"),
            "images_with_ontology_tags": None if search_index_summary is None else search_index_summary.get("images_with_ontology_tags"),
            "images_with_plant_bloom_prior": None if search_index_summary is None else search_index_summary.get("images_with_plant_bloom_prior"),
            "images_with_effective_campaigns": None if search_index_summary is None else search_index_summary.get("images_with_effective_campaigns"),
            "images_with_excluded_campaigns": None if search_index_summary is None else search_index_summary.get("images_with_excluded_campaigns"),
            "effective_campaign_counts": None if search_index_summary is None else search_index_summary.get("effective_campaign_counts"),
            "excluded_campaign_counts": None if search_index_summary is None else search_index_summary.get("excluded_campaign_counts"),
        },
        "query_smoke": {
            "summary_present": query_smoke is not None,
            "passed": None if query_smoke is None else query_smoke.get("consistency_check", {}).get("passed"),
            "conflicts": None if query_smoke is None else query_smoke.get("consistency_check", {}).get("excluded_and_effective_same_campaign_conflicts"),
            "image_count": None if query_smoke is None else query_smoke.get("image_count"),
            "images_with_effective_campaigns": None if query_smoke is None else query_smoke.get("images_with_effective_campaigns"),
            "images_with_exclusions": None if query_smoke is None else query_smoke.get("images_with_exclusions"),
            "excluded_campaign_counts": None if query_smoke is None else query_smoke.get("excluded_campaign_counts"),
        },
        "pipeline": {
            "summary_present": pipeline_summary is not None,
            "pipeline": None if pipeline_summary is None else pipeline_summary.get("pipeline"),
            "step_count": None if pipeline_summary is None else len(pipeline_summary.get("steps", [])),
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
