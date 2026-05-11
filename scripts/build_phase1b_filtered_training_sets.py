from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def counter_plain(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default="configs/phase1b_training_filter_v1.yaml")
    ap.add_argument("--out-dir", default="data/review/phase1b/filtered")
    ap.add_argument("--report-out", default="audit/phase_1b/phase_1b_filtered_training_report.json")
    args = ap.parse_args()

    policy_path = Path(args.policy)
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    if policy is None:
        raise RuntimeError(f"empty policy file: {policy_path}")

    included = set(policy["included_campaigns"])
    excluded = set(policy["excluded_campaigns"].keys())

    source = policy["source_artifacts"]
    review_events = read_jsonl(Path(source["review_events_cumulative"]))
    classifier_rows = read_jsonl(Path(source["classifier_training_cumulative"]))
    ranker_rows = read_jsonl(Path(source["ranker_training_cumulative"]))

    def filter_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for r in rows:
            cid = r["campaign_id"]
            if cid in included:
                out.append(r)
            elif cid in excluded:
                continue
            else:
                raise RuntimeError(f"campaign_id not in included/excluded policy: {cid}")
        return out

    filtered_events = filter_rows(review_events)
    filtered_classifier = filter_rows(classifier_rows)
    filtered_ranker = filter_rows(ranker_rows)

    out_dir = Path(args.out_dir)
    event_out = out_dir / "review_events_phase1b_v1__filtered.jsonl"
    classifier_out = out_dir / "training_snapshot_phase1b_classifier_v1__filtered.jsonl"
    ranker_out = out_dir / "training_snapshot_phase1b_ranker_v1__filtered.jsonl"

    write_jsonl(event_out, filtered_events)
    write_jsonl(classifier_out, filtered_classifier)
    write_jsonl(ranker_out, filtered_ranker)

    event_ids = [r["review_event_id"] for r in filtered_events]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError("duplicate review_event_id in filtered events")

    classifier_ids = [r["training_snapshot_id"] for r in filtered_classifier]
    ranker_ids = [r["training_snapshot_id"] for r in filtered_ranker]
    if len(classifier_ids) != len(set(classifier_ids)):
        raise RuntimeError("duplicate classifier training_snapshot_id")
    if len(ranker_ids) != len(set(ranker_ids)):
        raise RuntimeError("duplicate ranker training_snapshot_id")

    expected_counts = policy["expected_filtered_counts"]
    if len(filtered_events) != expected_counts["review_events"]:
        raise RuntimeError(f"expected {expected_counts['review_events']} events, got {len(filtered_events)}")
    if len(filtered_classifier) != expected_counts["classifier_training_rows"]:
        raise RuntimeError(f"expected {expected_counts['classifier_training_rows']} classifier rows, got {len(filtered_classifier)}")
    if len(filtered_ranker) != expected_counts["ranker_training_rows"]:
        raise RuntimeError(f"expected {expected_counts['ranker_training_rows']} ranker rows, got {len(filtered_ranker)}")

    event_label_counts = Counter(r["decision"]["label"] for r in filtered_events)
    classifier_label_counts = Counter(str(r["label"]) for r in filtered_classifier)
    ranker_label_counts = Counter(str(r["label"]) for r in filtered_ranker)
    campaign_counts = Counter(r["campaign_id"] for r in filtered_events)
    issue_tag_counts = Counter(tag for r in filtered_events for tag in r["decision"]["issue_tags"])

    expected_labels = policy["expected_filtered_label_counts"]

    if counter_plain(event_label_counts) != expected_labels["review_event_decision_labels"]:
        raise RuntimeError(
            f"event label counts mismatch: got={counter_plain(event_label_counts)} "
            f"expected={expected_labels['review_event_decision_labels']}"
        )

    if counter_plain(classifier_label_counts) != expected_labels["classifier_labels"]:
        raise RuntimeError(
            f"classifier label counts mismatch: got={counter_plain(classifier_label_counts)} "
            f"expected={expected_labels['classifier_labels']}"
        )

    if counter_plain(ranker_label_counts) != expected_labels["ranker_labels"]:
        raise RuntimeError(
            f"ranker label counts mismatch: got={counter_plain(ranker_label_counts)} "
            f"expected={expected_labels['ranker_labels']}"
        )

    report = {
        "metadata": {
            "spec_version": "v2.2.1",
            "phase": "phase_1b",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "score_status": "diagnostic_only",
            "filter_policy": str(policy_path),
            "report_status": "filtered_training_set_audit",
        },
        "included_campaigns": sorted(included),
        "excluded_campaigns": {
            cid: policy["excluded_campaigns"][cid]
            for cid in sorted(excluded)
        },
        "outputs": {
            "review_events_filtered": str(event_out),
            "classifier_training_filtered": str(classifier_out),
            "ranker_training_filtered": str(ranker_out),
        },
        "counts": {
            "review_events_filtered": len(filtered_events),
            "classifier_training_filtered": len(filtered_classifier),
            "ranker_training_filtered": len(filtered_ranker),
        },
        "label_counts": {
            "review_event_decision_labels": counter_plain(event_label_counts),
            "classifier_labels": counter_plain(classifier_label_counts),
            "ranker_labels": counter_plain(ranker_label_counts),
        },
        "campaign_counts": counter_plain(campaign_counts),
        "issue_tag_counts": counter_plain(issue_tag_counts),
        "interpretation": {
            "excluded_campaign_treatment": (
                "phase1b_indoor_gallery_winter_art는 coverage-gap diagnostic evidence로 보존하되, "
                "classifier/ranker smoke training 및 evaluation claim에서는 제외한다."
            ),
            "training_claim": (
                "이 filtered set은 feature plumbing 및 classifier smoke training 확인용으로만 사용할 수 있다. "
                "production quality 또는 calibrated threshold claim은 하지 않는다."
            ),
        },
    }

    report_path = Path(args.report_out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "event": "done",
        "policy": str(policy_path),
        "review_events_filtered": len(filtered_events),
        "classifier_training_filtered": len(filtered_classifier),
        "ranker_training_filtered": len(filtered_ranker),
        "event_label_counts": counter_plain(event_label_counts),
        "classifier_label_counts": counter_plain(classifier_label_counts),
        "ranker_label_counts": counter_plain(ranker_label_counts),
        "issue_tag_counts": counter_plain(issue_tag_counts),
        "report": str(report_path),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
