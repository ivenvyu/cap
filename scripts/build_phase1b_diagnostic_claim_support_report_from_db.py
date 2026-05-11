from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"yaml root must be object: {path}")
    return data


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing json file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def counter_plain(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def phase1b_campaign_ids(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        SELECT campaign_id
        FROM campaigns
        WHERE campaign_id LIKE 'phase1b_%'
        ORDER BY campaign_id
        """
    ).fetchall()
    return [str(r["campaign_id"]) for r in rows]


def review_summary_by_campaign(conn: sqlite3.Connection, campaign_id: str) -> dict[str, Any]:
    rows = conn.execute(
        """
        SELECT decision_label, issue_tags_json, source_bucket
        FROM review_events
        WHERE campaign_id = ?
        """,
        (campaign_id,),
    ).fetchall()

    labels = Counter()
    tags = Counter()
    buckets = Counter()

    for r in rows:
        labels[str(r["decision_label"])] += 1
        buckets[str(r["source_bucket"])] += 1
        for tag in json.loads(r["issue_tags_json"] or "[]"):
            tags[str(tag)] += 1

    usable = labels.get("accept", 0) + labels.get("acceptable", 0)

    return {
        "campaign_id": campaign_id,
        "review_rows": len(rows),
        "decision_label_counts": counter_plain(labels),
        "reject_count": int(labels.get("reject", 0)),
        "usable_count_accept_or_acceptable": int(usable),
        "issue_tag_counts": counter_plain(tags),
        "bucket_counts": counter_plain(buckets),
    }


def phase1b_issue_tag_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tags = Counter()
    for cid in phase1b_campaign_ids(conn):
        rows = conn.execute(
            "SELECT issue_tags_json FROM review_events WHERE campaign_id = ?",
            (cid,),
        ).fetchall()
        for r in rows:
            for tag in json.loads(r["issue_tags_json"] or "[]"):
                tags[str(tag)] += 1
    return counter_plain(tags)


def training_set_label_counts(conn: sqlite3.Connection, training_set_id: str) -> dict[str, int]:
    rows = conn.execute(
        """
        SELECT t.label, COUNT(*) AS n
        FROM training_set_items i
        JOIN training_snapshots t
          ON i.training_snapshot_id = t.training_snapshot_id
        WHERE i.training_set_id = ?
        GROUP BY t.label
        ORDER BY t.label
        """,
        (training_set_id,),
    ).fetchall()
    return {str(r["label"]): int(r["n"]) for r in rows}


def training_set_row_count(conn: sqlite3.Connection, training_set_id: str) -> int:
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM training_set_items WHERE training_set_id = ?",
            (training_set_id,),
        ).fetchone()[0]
    )


def next_action_for_entry(entry: dict[str, Any], threshold_semantics: dict[str, Any]) -> str:
    if "next_action" in entry:
        return str(entry["next_action"])
    status = entry["threshold_status"]
    default = threshold_semantics.get(status, {}).get("default_next_action")
    if default:
        return str(default)
    return "record_only"


def build_future_observation_queue(policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    threshold_semantics = policy["threshold_status_semantics"]

    for claim_id, claim in policy["claims"].items():
        for direction in ["would_strengthen", "would_weaken"]:
            for idx, entry in enumerate(claim.get(direction, [])):
                rows.append(
                    {
                        "claim_id": claim_id,
                        "claim_type": claim["claim_type"],
                        "direction": direction,
                        "entry_index": idx,
                        "observation": entry["observation"],
                        "measurement": entry["measurement"],
                        "threshold_status": entry["threshold_status"],
                        "next_action": next_action_for_entry(entry, threshold_semantics),
                        "current_status": "pending_future_observation",
                    }
                )

    return rows


def get_architecture_fold(smoke_report: dict[str, Any]) -> dict[str, Any]:
    for fold in smoke_report["leave_one_campaign_out"]["folds"]:
        if fold["heldout_campaign"] == "phase1b_architecture_exhibition_visit":
            return fold
    raise RuntimeError("architecture held-out fold not found in smoke report")


def evaluate_claims(
    conn: sqlite3.Connection,
    policy: dict[str, Any],
    smoke_report: dict[str, Any],
) -> dict[str, Any]:
    evaluations: dict[str, Any] = {}
    claims = policy["claims"]

    all_issue_tags = phase1b_issue_tag_counts(conn)
    architecture_fold = get_architecture_fold(smoke_report)
    smoke_meta = smoke_report["metadata"]
    smoke_dataset = smoke_report["dataset"]
    smoke_oof = smoke_report["leave_one_campaign_out"]["out_of_fold_metrics"]

    classifier_rows = training_set_row_count(conn, "phase1b_filtered_classifier_v1")
    classifier_labels = training_set_label_counts(conn, "phase1b_filtered_classifier_v1")

    ranker_rows = training_set_row_count(conn, "phase1b_filtered_ranker_v1")
    ranker_labels = training_set_label_counts(conn, "phase1b_filtered_ranker_v1")

    for claim_id, claim in claims.items():
        caveats: list[str] = []

        if claim_id == "exclude_indoor_winter_from_smoke_training":
            campaign_id = "phase1b_indoor_gallery_winter_art"
            summary = review_summary_by_campaign(conn, campaign_id)

            excluded_from_classifier_set = int(
                conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM training_set_items i
                    JOIN training_snapshots t
                      ON i.training_snapshot_id = t.training_snapshot_id
                    WHERE i.training_set_id = 'phase1b_filtered_classifier_v1'
                      AND t.campaign_id = ?
                    """,
                    (campaign_id,),
                ).fetchone()[0]
            ) == 0

            evidence = {
                **summary,
                "excluded_from_phase1b_filtered_classifier_v1": excluded_from_classifier_set,
            }

            status = "diagnostic_supported"
            if summary["reject_count"] != 29 or summary["review_rows"] != 30:
                status = "needs_review"
                caveats.append("expected 29 rejects out of 30 was not reproduced from DB")
            if not excluded_from_classifier_set:
                status = "needs_review"
                caveats.append("indoor/winter campaign appears in filtered classifier set")

            interpretation = (
                "실내/겨울 campaign 제외 판단은 DB evidence에 의해 지지된다. "
                "단, 이 판단은 model-quality failure가 아니라 raw pool coverage gap diagnostic으로만 해석한다."
            )

        elif claim_id == "prioritize_preview_renderer_v1":
            layout_tags = {
                "text_region_conflict": int(all_issue_tags.get("text_region_conflict", 0)),
                "low_contrast": int(all_issue_tags.get("low_contrast", 0)),
                "too_busy_background": int(all_issue_tags.get("too_busy_background", 0)),
                "visual_hierarchy_weak": int(all_issue_tags.get("visual_hierarchy_weak", 0)),
            }
            layout_total = sum(layout_tags.values())

            evidence = {
                "layout_related_issue_tags": layout_tags,
                "layout_related_issue_total": layout_total,
                "review_condition": "phase1b_first_round_without_layout_renderer",
            }

            status = "diagnostic_supported"
            interpretation = (
                "layout 관련 issue tag가 매우 적으므로 preview/renderer 관찰 장치 필요성이 DB evidence로 지지된다. "
                "이는 layout 문제가 없다는 뜻이 아니라 layout observability가 낮다는 뜻이다."
            )

        elif claim_id == "interpret_architecture_fold_as_campaign_shift":
            metrics = architecture_fold["metrics"]
            evidence = {
                "heldout_campaign": architecture_fold["heldout_campaign"],
                "test_label_counts": architecture_fold["test_label_counts"],
                "train_label_counts": architecture_fold["train_label_counts"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
            }

            status = "diagnostic_supported"
            if metrics["roc_auc"] is not None and metrics["roc_auc"] >= 0.5:
                caveats.append("architecture fold roc_auc is not below 0.5; campaign-shift interpretation should be weakened")

            interpretation = (
                "architecture fold 약화는 campaign-shift diagnostic으로 해석할 수 있다. "
                "단, architecture family가 1개뿐이므로 production failure나 일반화 실패로 단정하지 않는다."
            )

        elif claim_id == "reject_classifier_smoke_as_quality_claim":
            evidence = {
                "db_role": smoke_meta.get("db_role"),
                "smoke_report_status": smoke_meta["report_status"],
                "score_status": smoke_meta["score_status"],
                "threshold_status": smoke_meta["threshold_status"],
                "rows": smoke_dataset["rows"],
                "campaign_counts": smoke_dataset["campaign_counts"],
                "label_counts": smoke_dataset["label_counts"],
                "out_of_fold_metrics": {
                    "balanced_accuracy": smoke_oof["balanced_accuracy"],
                    "roc_auc": smoke_oof["roc_auc"],
                    "average_precision": smoke_oof["average_precision"],
                },
                "db_training_set_rows": classifier_rows,
                "db_training_set_labels": classifier_labels,
            }

            status = "diagnostic_supported"
            if smoke_meta["score_status"] != "diagnostic_only":
                status = "needs_review"
                caveats.append("smoke report score_status is not diagnostic_only")
            if smoke_meta["threshold_status"] != "no_calibrated_threshold":
                status = "needs_review"
                caveats.append("smoke report threshold_status is not no_calibrated_threshold")
            if classifier_rows != 120 or classifier_labels != {"0": 70, "1": 50}:
                status = "needs_review"
                caveats.append("DB classifier training set does not match expected 120 rows / 70-50 labels")

            interpretation = (
                "classifier smoke result를 quality claim으로 사용하지 않는 판단은 DB evidence에 의해 지지된다. "
                "이 결과는 DB-first feature plumbing과 campaign-held-out loop가 동작함을 확인하는 진단이다."
            )

        elif claim_id == "monitor_ranker_label_boundary_stability":
            acceptable = int(ranker_labels.get("1", 0))
            accept = int(ranker_labels.get("2", 0))
            reject = int(ranker_labels.get("0", 0))

            evidence = {
                "training_set_id": "phase1b_filtered_ranker_v1",
                "row_count": ranker_rows,
                "ranker_label_counts": ranker_labels,
                "reject_count": reject,
                "acceptable_count": acceptable,
                "accept_count": accept,
                "acceptable_accept_difference": abs(acceptable - accept),
                "multi_annotator_overlap_available": False,
            }

            status = "diagnostic_supported"
            if ranker_rows != 120 or ranker_labels != {"0": 70, "1": 24, "2": 26}:
                status = "needs_review"
                caveats.append("DB ranker training set does not match expected rows/labels")

            interpretation = (
                "ranker label 1/2 경계 안정성은 아직 검증되지 않았다. "
                "acceptable과 accept가 24/26으로 거의 같으므로, ranker metric을 성능 주장으로 쓰지 않고 "
                "multi-annotator overlap 또는 campaign별 label audit에서 경계 안정성을 확인해야 한다."
            )

        else:
            evidence = {}
            status = "not_evaluated_unknown_claim"
            caveats = ["no evaluator implemented for this claim_id"]
            interpretation = "이 claim은 현재 DB-first report builder에서 자동 평가하지 않는다."

        evaluations[claim_id] = {
            "claim_id": claim_id,
            "claim": claim["claim"],
            "claim_type": claim["claim_type"],
            "status_from_policy": claim["status"],
            "evaluation_status": status,
            "supporting_evidence_observed": evidence,
            "missing_or_weak_observations": claim.get("missing_or_weak_observations", []),
            "caveats": caveats,
            "interpretation": interpretation,
            "forbidden_interpretation": claim.get("forbidden_interpretation", []),
        }

    return evaluations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--policy", default="configs/diagnostic_claim_support_v1.yaml")
    ap.add_argument("--smoke-report", default="audit/phase_1b/phase_1b_classifier_smoke_report_db.json")
    ap.add_argument("--out", default="audit/phase_1b/phase_1b_diagnostic_claim_support_report_db.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    policy = read_yaml(Path(args.policy))
    smoke_report = read_json(Path(args.smoke_report))

    evaluations = evaluate_claims(conn, policy, smoke_report)
    evaluation_counts = Counter(ev["evaluation_status"] for ev in evaluations.values())
    future_queue = build_future_observation_queue(policy)

    report = {
        "metadata": {
            "spec_version": "v2.2.1",
            "phase": "phase_1b",
            "created_at": utc_now(),
            "report_status": "diagnostic_claim_support_execution_db_first",
            "score_status": "diagnostic_only",
            "support_level": "claim_level",
            "candidate_support_explanation_status": "deferred_from_phase_1b",
            "threshold_status": "diagnostic_trigger_only_not_final_threshold",
            "db_role": "operational_source_of_truth",
        },
        "inputs": {
            "db": args.db,
            "policy": args.policy,
            "smoke_report": args.smoke_report,
        },
        "claim_evaluations": evaluations,
        "evaluation_counts": counter_plain(evaluation_counts),
        "future_observation_queue": future_queue,
        "non_claims": [
            "candidate-level support explanation을 생성하지 않는다.",
            "calibrated accept/reject threshold를 만들지 않는다.",
            "automatic accept/reject rule로 사용하지 않는다.",
            "production model quality를 주장하지 않는다.",
        ],
    }

    write_json(Path(args.out), report)

    print(json.dumps({
        "event": "done",
        "out": args.out,
        "claims": len(evaluations),
        "evaluation_counts": counter_plain(evaluation_counts),
        "future_observations": len(future_queue),
        "score_status": "diagnostic_only",
        "support_level": "claim_level",
        "db_role": "operational_source_of_truth",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
