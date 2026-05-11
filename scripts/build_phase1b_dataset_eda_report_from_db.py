from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def counter_plain(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def describe_numeric(s: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
        }

    return {
        "count": int(len(x)),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=0)),
        "min": float(x.min()),
        "p25": float(x.quantile(0.25)),
        "median": float(x.quantile(0.50)),
        "p75": float(x.quantile(0.75)),
        "max": float(x.max()),
    }


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def phase1b_campaigns(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            campaign_id,
            campaign_family,
            purpose_type,
            space_type,
            season
        FROM campaigns
        WHERE campaign_id LIKE 'phase1b_%'
        ORDER BY campaign_id
        """
    ).fetchall()
    return [dict(r) for r in rows]


def load_phase1b_review_events(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            review_event_id,
            campaign_id,
            image_id,
            pair_id,
            feature_snapshot_id,
            decision_label,
            decision_numeric,
            issue_tags_json,
            source_bucket,
            queue_stage,
            layout_spec_id,
            preview_renderer_version
        FROM review_events
        WHERE campaign_id LIKE 'phase1b_%'
        ORDER BY campaign_id, review_event_id
        """
    ).fetchall()

    out = []
    for r in rows:
        d = dict(r)
        d["issue_tags"] = json.loads(d.pop("issue_tags_json") or "[]")
        out.append(d)

    return out


def load_training_set_rows(conn: sqlite3.Connection, training_set_id: str) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT
            i.item_order,
            t.training_snapshot_id,
            t.snapshot_kind,
            t.snapshot_version,
            t.campaign_id,
            t.image_id,
            t.pair_id,
            t.feature_snapshot_id,
            t.layout_spec_id,
            t.label,
            t.decision_label,
            t.group_id,
            t.label_status,
            t.issue_tags_json,
            p.duplicate_group_id,
            p.feature_status,
            p.features_json
        FROM training_set_items i
        JOIN training_snapshots t
          ON i.training_snapshot_id = t.training_snapshot_id
        JOIN pair_features p
          ON t.feature_snapshot_id = p.feature_snapshot_id
        WHERE i.training_set_id = ?
        ORDER BY i.item_order
        """,
        (training_set_id,),
    ).fetchall()

    flat = []
    for row in rows:
        r = dict(row)
        features = json.loads(r.pop("features_json") or "{}")
        r["issue_tags"] = json.loads(r.pop("issue_tags_json") or "[]")
        for k, v in features.items():
            r[k] = v
        flat.append(r)

    return pd.DataFrame(flat)


def load_pair_feature_rows(conn: sqlite3.Connection) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT
            feature_snapshot_id,
            campaign_id,
            image_id,
            pair_id,
            layout_spec_id,
            duplicate_group_id,
            feature_status,
            features_json
        FROM pair_features
        WHERE campaign_id LIKE 'phase1b_%'
        ORDER BY campaign_id, feature_snapshot_id
        """
    ).fetchall()

    flat = []
    for row in rows:
        r = dict(row)
        features = json.loads(r.pop("features_json") or "{}")
        for k, v in features.items():
            r[k] = v
        flat.append(r)

    return pd.DataFrame(flat)


def summarize_reviews(events: list[dict[str, Any]]) -> dict[str, Any]:
    label_counts = Counter(e["decision_label"] for e in events)
    issue_counts = Counter(tag for e in events for tag in e["issue_tags"])
    bucket_counts = Counter(e["source_bucket"] for e in events)

    campaign_label_matrix: dict[str, Counter] = defaultdict(Counter)
    campaign_issue_matrix: dict[str, Counter] = defaultdict(Counter)
    bucket_label_matrix: dict[str, Counter] = defaultdict(Counter)

    for e in events:
        campaign_label_matrix[e["campaign_id"]][e["decision_label"]] += 1
        bucket_label_matrix[e["source_bucket"]][e["decision_label"]] += 1
        for tag in e["issue_tags"]:
            campaign_issue_matrix[e["campaign_id"]][tag] += 1

    return {
        "row_count": len(events),
        "decision_label_counts": counter_plain(label_counts),
        "issue_tag_counts": counter_plain(issue_counts),
        "bucket_counts": counter_plain(bucket_counts),
        "campaign_label_matrix": {
            cid: counter_plain(c)
            for cid, c in sorted(campaign_label_matrix.items())
        },
        "campaign_issue_matrix": {
            cid: counter_plain(c)
            for cid, c in sorted(campaign_issue_matrix.items())
        },
        "bucket_label_matrix": {
            bucket: counter_plain(c)
            for bucket, c in sorted(bucket_label_matrix.items())
        },
    }


def selected_numeric_features(df: pd.DataFrame) -> list[str]:
    selected = [
        "clip_margin",
        "clip_positive_max_sim",
        "clip_negative_max_sim",
        "clip_positive_mean_sim",
        "clip_negative_mean_sim",
        "clip_rank_percentile",
        "required_region_safe_min",
        "required_region_safe_mean",
        "title_region_safe_score",
        "info_region_safe_score",
        "edge_density",
        "brightness",
        "contrast",
        "saturation",
        "path_has_architecture",
        "path_has_garden",
        "image_category_gallery",
        "image_category_flower",
        "image_category_tree",
        "campaign_is_garden",
        "campaign_is_summer",
        "campaign_is_walking_program",
    ]

    cols = []
    for c in selected:
        if c not in df.columns:
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().any():
            df[c] = converted
            cols.append(c)

    return cols


def summarize_training_set(df: pd.DataFrame) -> dict[str, Any]:
    labels = Counter(df["label"].astype(int).tolist())
    campaigns = Counter(df["campaign_id"].astype(str).tolist())
    decisions = Counter(df["decision_label"].astype(str).tolist())

    feature_cols = selected_numeric_features(df)

    feature_summary = {
        c: describe_numeric(df[c])
        for c in feature_cols
    }

    feature_by_label = {}
    for c in feature_cols:
        feature_by_label[c] = {}
        for label in sorted(df["label"].dropna().unique().tolist()):
            sub = df[df["label"] == label]
            feature_by_label[c][str(int(label))] = describe_numeric(sub[c])

    feature_by_campaign = {}
    for c in feature_cols:
        feature_by_campaign[c] = {}
        for cid in sorted(df["campaign_id"].dropna().unique().tolist()):
            sub = df[df["campaign_id"] == cid]
            feature_by_campaign[c][cid] = describe_numeric(sub[c])

    return {
        "row_count": int(len(df)),
        "label_counts": counter_plain(labels),
        "decision_label_counts": counter_plain(decisions),
        "campaign_counts": counter_plain(campaigns),
        "numeric_feature_count_selected": len(feature_cols),
        "selected_feature_columns": feature_cols,
        "feature_summary": feature_summary,
        "feature_by_label": feature_by_label,
        "feature_by_campaign": feature_by_campaign,
    }


def summarize_pair_features(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"row_count": 0}

    campaigns = Counter(df["campaign_id"].astype(str).tolist())
    duplicate_groups = int(df["duplicate_group_id"].nunique()) if "duplicate_group_id" in df.columns else None

    return {
        "row_count": int(len(df)),
        "campaign_counts": counter_plain(campaigns),
        "unique_duplicate_groups": duplicate_groups,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--classifier-training-set-id", default="phase1b_filtered_classifier_v1")
    ap.add_argument("--ranker-training-set-id", default="phase1b_filtered_ranker_v1")
    ap.add_argument("--out", default="audit/phase_1b/phase_1b_dataset_eda_report_db.json")
    ap.add_argument("--numbers-csv", default="audit/phase_1b/phase_1b_dataset_eda_numbers_db.csv")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    campaigns = phase1b_campaigns(conn)
    events = load_phase1b_review_events(conn)
    classifier_df = load_training_set_rows(conn, args.classifier_training_set_id)
    ranker_df = load_training_set_rows(conn, args.ranker_training_set_id)
    pair_df = load_pair_feature_rows(conn)

    review_summary = summarize_reviews(events)
    classifier_summary = summarize_training_set(classifier_df)
    ranker_summary = summarize_training_set(ranker_df)
    pair_summary = summarize_pair_features(pair_df)

    issue_counts = review_summary["issue_tag_counts"]
    layout_related_tags = [
        "text_region_conflict",
        "low_contrast",
        "too_busy_background",
        "visual_hierarchy_weak",
    ]
    layout_issue_total = sum(int(issue_counts.get(tag, 0)) for tag in layout_related_tags)

    report = {
        "metadata": {
            "spec_version": "v2.2.1",
            "phase": "phase_1b",
            "created_at": utc_now(),
            "report_status": "dataset_eda_db_first",
            "db_role": "operational_source_of_truth",
            "score_status": "diagnostic_only",
            "threshold_status": "no_calibrated_threshold",
            "candidate_support_explanation_status": "deferred_from_phase_1b",
        },
        "inputs": {
            "db": args.db,
            "classifier_training_set_id": args.classifier_training_set_id,
            "ranker_training_set_id": args.ranker_training_set_id,
        },
        "campaigns": campaigns,
        "phase1b_review_events": review_summary,
        "phase1b_pair_features": pair_summary,
        "filtered_classifier_training_set": classifier_summary,
        "filtered_ranker_training_set": ranker_summary,
        "eda_findings": [
            {
                "name": "reject_rate_is_high_in_first_round",
                "type": "dataset_coverage_signal",
                "evidence": review_summary["decision_label_counts"],
                "interpretation": (
                    "Phase 1b first round는 reject-heavy다. "
                    "이는 model failure가 아니라 campaign diversity와 raw pool coverage gap의 신호로 해석한다."
                ),
            },
            {
                "name": "season_mismatch_is_new_major_signal",
                "type": "ontology_axis_signal",
                "evidence": {
                    "season_mismatch": int(issue_counts.get("season_mismatch", 0)),
                    "semantic_mismatch": int(issue_counts.get("semantic_mismatch", 0)),
                },
                "interpretation": (
                    "season_mismatch가 semantic_mismatch와 비슷한 규모로 나타났다. "
                    "이는 Phase 1b에서 season axis가 실제 review 판단에 작동하기 시작했다는 신호다."
                ),
            },
            {
                "name": "layout_issues_are_under_observed",
                "type": "observability_gap",
                "evidence": {
                    "layout_related_issue_total": int(layout_issue_total),
                    "layout_related_issue_tags": {
                        tag: int(issue_counts.get(tag, 0))
                        for tag in layout_related_tags
                    },
                },
                "interpretation": (
                    "layout-related tag가 거의 없으므로 layout 문제가 없다고 결론낼 수 없다. "
                    "renderer/preview 없이 layout observability가 낮았다는 신호로 해석한다."
                ),
            },
            {
                "name": "ranker_label_boundary_unverified",
                "type": "label_policy_signal",
                "evidence": ranker_summary["label_counts"],
                "interpretation": (
                    "ranker label 1/2 경계는 아직 검증되지 않았다. "
                    "acceptable/accept 경계 안정성은 multi-annotator overlap 또는 별도 label audit이 필요하다."
                ),
            },
        ],
        "non_claims": [
            "production model quality를 주장하지 않는다.",
            "calibrated threshold를 만들지 않는다.",
            "candidate-level support explanation을 만들지 않는다.",
            "design quality를 주장하지 않는다.",
        ],
    }

    write_json(Path(args.out), report)

    number_rows = [
        {"section": "metadata", "metric": "report_status", "value": report["metadata"]["report_status"]},
        {"section": "metadata", "metric": "db_role", "value": report["metadata"]["db_role"]},
        {"section": "reviews", "metric": "phase1b_review_events", "value": review_summary["row_count"]},
        {"section": "reviews", "metric": "decision_label_counts", "value": json.dumps(review_summary["decision_label_counts"], ensure_ascii=False, sort_keys=True)},
        {"section": "reviews", "metric": "issue_tag_counts", "value": json.dumps(review_summary["issue_tag_counts"], ensure_ascii=False, sort_keys=True)},
        {"section": "features", "metric": "phase1b_pair_features", "value": pair_summary["row_count"]},
        {"section": "classifier", "metric": "rows", "value": classifier_summary["row_count"]},
        {"section": "classifier", "metric": "label_counts", "value": json.dumps(classifier_summary["label_counts"], ensure_ascii=False, sort_keys=True)},
        {"section": "ranker", "metric": "rows", "value": ranker_summary["row_count"]},
        {"section": "ranker", "metric": "label_counts", "value": json.dumps(ranker_summary["label_counts"], ensure_ascii=False, sort_keys=True)},
        {"section": "eda", "metric": "layout_related_issue_total", "value": layout_issue_total},
    ]

    write_csv(Path(args.numbers_csv), number_rows)

    print(json.dumps({
        "event": "done",
        "out": args.out,
        "numbers_csv": args.numbers_csv,
        "db": args.db,
        "review_events": review_summary["row_count"],
        "classifier_rows": classifier_summary["row_count"],
        "ranker_rows": ranker_summary["row_count"],
        "pair_features": pair_summary["row_count"],
        "review_decision_labels": review_summary["decision_label_counts"],
        "classifier_labels": classifier_summary["label_counts"],
        "ranker_labels": ranker_summary["label_counts"],
        "issue_tags": review_summary["issue_tag_counts"],
        "score_status": "diagnostic_only",
        "db_role": "operational_source_of_truth",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
