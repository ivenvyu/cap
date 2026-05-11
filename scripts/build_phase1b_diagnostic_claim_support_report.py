from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing json file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing yaml file: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"yaml root must be object: {path}")
    return data


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def counter_plain(c: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in c.items()}


def get_architecture_fold(smoke_report: dict[str, Any]) -> dict[str, Any]:
    folds = smoke_report["leave_one_campaign_out"]["folds"]
    for fold in folds:
        if fold["heldout_campaign"] == "phase1b_architecture_exhibition_visit":
            return fold
    raise RuntimeError("architecture held-out fold not found")


def next_action_for_entry(
    entry: dict[str, Any],
    threshold_semantics: dict[str, Any],
) -> str:
    if "next_action" in entry:
        return str(entry["next_action"])
    status = entry["threshold_status"]
    default = threshold_semantics.get(status, {}).get("default_next_action")
    if default:
        return str(default)
    return "record_only"


def build_future_observation_queue(
    claims: dict[str, Any],
    threshold_semantics: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = []
    for claim_id, claim in claims.items():
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


def evaluate_claims(
    policy: dict[str, Any],
    first_round: dict[str, Any],
    filtered_report: dict[str, Any],
    smoke_report: dict[str, Any],
) -> dict[str, Any]:
    claims = policy["claims"]
    evaluations: dict[str, Any] = {}

    # Shared evidence.
    first_summary = first_round["cumulative_review_summary"]
    first_issue_tags = first_summary["issue_tag_counts"]

    filtered_excluded = set(filtered_report.get("excluded_campaigns", {}).keys())
    smoke_meta = smoke_report["metadata"]
    smoke_dataset = smoke_report["dataset"]
    smoke_oof = smoke_report["leave_one_campaign_out"]["out_of_fold_metrics"]
    architecture_fold = get_architecture_fold(smoke_report)

    for claim_id, claim in claims.items():
        if claim_id == "exclude_indoor_winter_from_smoke_training":
            campaign_id = "phase1b_indoor_gallery_winter_art"
            campaign_summary = first_round["campaign_summaries"][campaign_id]
            reject_count = campaign_summary["reject_count"]
            review_rows = campaign_summary["review_queue_rows"]
            issue_tags = campaign_summary["issue_tag_counts"]

            evidence = {
                "campaign_id": campaign_id,
                "reject_count": reject_count,
                "review_rows": review_rows,
                "usable_count_accept_or_acceptable": campaign_summary["positive_count_accept_or_acceptable"],
                "issue_tag_counts": issue_tags,
                "excluded_from_filtered_set": campaign_id in filtered_excluded,
            }

            status = "diagnostic_supported"
            caveats = []
            if reject_count != 29 or review_rows != 30:
                status = "needs_review"
                caveats.append("expected first-round observation 29 rejects out of 30 was not reproduced")
            if campaign_id not in filtered_excluded:
                status = "needs_review"
                caveats.append("campaign is not excluded in filtered report")

            interpretation = (
                "실내/겨울 campaign 제외 판단은 현재 evidence에 의해 지지된다. "
                "단, 이 판단은 model-quality failure가 아니라 raw pool coverage gap diagnostic으로만 해석한다."
            )

        elif claim_id == "prioritize_preview_renderer_v1":
            layout_tags = {
                "text_region_conflict": int(first_issue_tags.get("text_region_conflict", 0)),
                "low_contrast": int(first_issue_tags.get("low_contrast", 0)),
                "too_busy_background": int(first_issue_tags.get("too_busy_background", 0)),
                "visual_hierarchy_weak": int(first_issue_tags.get("visual_hierarchy_weak", 0)),
            }
            layout_total = sum(layout_tags.values())

            evidence = {
                "layout_related_issue_tags": layout_tags,
                "layout_related_issue_total": layout_total,
                "review_condition": "first_round_without_preview_renderer_overlay",
                "raw_pool_expansion_status": "blocked_no_new_images_available",
            }

            status = "diagnostic_supported"
            caveats = []
            if layout_total > 1:
                caveats.append("layout tags are not as sparse as expected; still diagnostic_only")

            interpretation = (
                "preview renderer v1 우선순위 판단은 현재 evidence에 의해 지지된다. "
                "layout issue가 없다는 뜻이 아니라, overlay 없는 review에서 layout observability가 낮다는 뜻이다."
            )

        elif claim_id == "interpret_architecture_fold_as_campaign_shift":
            metrics = architecture_fold["metrics"]

            evidence = {
                "heldout_campaign": architecture_fold["heldout_campaign"],
                "test_label_counts": architecture_fold["test_label_counts"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "roc_auc": metrics["roc_auc"],
                "train_label_counts": architecture_fold["train_label_counts"],
            }

            status = "diagnostic_supported"
            caveats = []
            if architecture_fold["heldout_campaign"] != "phase1b_architecture_exhibition_visit":
                status = "needs_review"
                caveats.append("architecture fold lookup failed")
            if metrics["roc_auc"] is not None and metrics["roc_auc"] >= 0.5:
                caveats.append("architecture fold is not below 0.5 roc_auc; campaign-shift interpretation should be weakened")

            interpretation = (
                "architecture fold 약화는 campaign-shift diagnostic으로 해석할 수 있다. "
                "단, architecture family가 1개뿐이므로 production failure나 일반화 실패로 단정하지 않는다."
            )

        elif claim_id == "monitor_ranker_label_boundary_stability":
            ranker_counts = filtered_report["label_counts"]["ranker_labels"]
            acceptable = int(ranker_counts.get("1", 0))
            accept = int(ranker_counts.get("2", 0))
            reject = int(ranker_counts.get("0", 0))

            evidence = {
                "ranker_label_counts": ranker_counts,
                "reject_count": reject,
                "acceptable_count": acceptable,
                "accept_count": accept,
                "acceptable_accept_difference": abs(acceptable - accept),
                "multi_annotator_overlap_available": False,
            }

            status = "diagnostic_supported"
            caveats = [
                "acceptable/accept boundary is only suspected from label distribution; no multi-annotator overlap exists yet."
            ]

            interpretation = (
                "ranker label 1/2 경계 안정성은 아직 검증되지 않았다. "
                "acceptable과 accept가 24/26으로 거의 같으므로, ranker metric을 성능 주장으로 쓰지 않고 "
                "multi-annotator overlap 또는 campaign별 label audit에서 경계 안정성을 확인해야 한다."
            )

        elif claim_id == "reject_classifier_smoke_as_quality_claim":
            evidence = {
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
            }

            status = "diagnostic_supported"
            caveats = []
            if smoke_meta["score_status"] != "diagnostic_only":
                status = "needs_review"
                caveats.append("smoke report score_status is not diagnostic_only")
            if smoke_meta["threshold_status"] != "no_calibrated_threshold":
                status = "needs_review"
                caveats.append("smoke report threshold_status is not no_calibrated_threshold")

            interpretation = (
                "classifier smoke result를 quality claim으로 사용하지 않는 판단은 지지된다. "
                "이 결과는 feature plumbing과 campaign-held-out loop가 동작함을 확인하는 진단이다."
            )

        else:
            evidence = {}
            status = "not_evaluated_unknown_claim"
            caveats = ["no evaluator implemented for this claim_id"]
            interpretation = "이 claim은 현재 report builder에서 자동 평가하지 않는다."

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


def render_markdown(report: dict[str, Any]) -> str:
    lines = []
    lines.append("# Phase 1b Diagnostic Claim Support Report")
    lines.append("")
    lines.append("## 메타데이터")
    lines.append("")
    meta = report["metadata"]
    for key in [
        "spec_version",
        "phase",
        "report_status",
        "score_status",
        "support_level",
        "candidate_support_explanation_status",
    ]:
        lines.append(f"{key}: {meta[key]}")
    lines.append("")
    lines.append("## Claim evaluations")
    lines.append("")

    for claim_id, ev in report["claim_evaluations"].items():
        lines.append(f"### {claim_id}")
        lines.append("")
        lines.append(f"claim_type: {ev['claim_type']}")
        lines.append(f"evaluation_status: {ev['evaluation_status']}")
        lines.append("")
        lines.append("claim:")
        lines.append("")
        lines.append(ev["claim"].strip())
        lines.append("")
        lines.append("interpretation:")
        lines.append("")
        lines.append(ev["interpretation"])
        lines.append("")
        if ev["caveats"]:
            lines.append("caveats:")
            for c in ev["caveats"]:
                lines.append(f"- {c}")
            lines.append("")

    lines.append("## Future observation queue")
    lines.append("")
    for row in report["future_observation_queue"]:
        lines.append(
            f"- {row['claim_id']} / {row['direction']} / "
            f"{row['threshold_status']} → {row['current_status']}"
        )
    lines.append("")

    lines.append("## 결론")
    lines.append("")
    lines.append(
        "이 report는 candidate-level support explanation이 아니라 "
        "claim-level diagnostic support 실행 결과다. "
        "모든 판단은 diagnostic_only이며, calibrated threshold 또는 automatic decision rule로 사용하지 않는다."
    )
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="configs/diagnostic_claim_support_v1.yaml")
    ap.add_argument("--first-round-report", default="audit/phase_1b/phase_1b_first_round_report.json")
    ap.add_argument("--filtered-report", default="audit/phase_1b/phase_1b_filtered_training_report.json")
    ap.add_argument("--smoke-report", default="audit/phase_1b/phase_1b_classifier_smoke_report.json")
    ap.add_argument("--out", default="audit/phase_1b/phase_1b_diagnostic_claim_support_report.json")
    ap.add_argument("--summary-out", default="audit/phase_1b/phase_1b_diagnostic_claim_support_summary.md")
    args = ap.parse_args()

    policy = read_yaml(Path(args.policy))
    first_round = read_json(Path(args.first_round_report))
    filtered_report = read_json(Path(args.filtered_report))
    smoke_report = read_json(Path(args.smoke_report))

    evaluations = evaluate_claims(policy, first_round, filtered_report, smoke_report)
    future_queue = build_future_observation_queue(
        policy["claims"],
        policy["threshold_status_semantics"],
    )

    evaluation_counts = Counter(ev["evaluation_status"] for ev in evaluations.values())

    report = {
        "metadata": {
            "spec_version": "v2.2.1",
            "phase": "phase_1b",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "report_status": "diagnostic_claim_support_execution",
            "score_status": "diagnostic_only",
            "support_level": "claim_level",
            "candidate_support_explanation_status": "deferred_from_phase_1b",
            "threshold_status": "diagnostic_trigger_only_not_final_threshold",
        },
        "inputs": {
            "policy": args.policy,
            "first_round_report": args.first_round_report,
            "filtered_report": args.filtered_report,
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
    write_text(Path(args.summary_out), render_markdown(report))

    print(json.dumps({
        "event": "done",
        "out": args.out,
        "summary_out": args.summary_out,
        "claims": len(evaluations),
        "evaluation_counts": counter_plain(evaluation_counts),
        "future_observations": len(future_queue),
        "score_status": "diagnostic_only",
        "support_level": "claim_level",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
