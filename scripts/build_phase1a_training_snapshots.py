from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from collections import Counter

from jsonschema import Draft202012Validator


CLASSIFIER_LABEL = {
    "reject": 0,
    "acceptable": 1,
    "accept": 1,
    "best": 1,
}

RANKER_LABEL = {
    "reject": 0,
    "acceptable": 1,
    "accept": 2,
    "best": 3,
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def make_aggregation_policy() -> dict[str, Any]:
    return {
        "policy_name": "training_snapshot_aggregation_v1",
        "primary_key": [
            "pair_id",
            "layout_spec_id",
            "preview_renderer_version",
        ],
        "same_annotator_overwrite": "most_recent",
        "multi_annotator_aggregation": "not_applicable",
        "tie_break": None,
        "log_disagreement": True,
    }


def make_classifier_label_policy() -> dict[str, Any]:
    return {
        "policy_name": "classifier_binary_v1",
        "label_mapping": {
            "reject": 0,
            "acceptable": 1,
            "accept": 1,
            "best": 1,
        },
    }


def make_ranker_label_policy() -> dict[str, Any]:
    return {
        "policy_name": "ranker_relevance_v1",
        "label_mapping": {
            "reject": 0,
            "acceptable": 1,
            "accept": 2,
            "best": 3,
        },
    }


def make_snapshot_row(
    *,
    event: dict[str, Any],
    snapshot_kind: str,
    snapshot_version: str,
    created_at: str,
) -> dict[str, Any]:
    decision_label = event["decision"]["label"]

    if snapshot_kind == "classifier":
        label = CLASSIFIER_LABEL[decision_label]
        relevance_grade = None
        label_policy = make_classifier_label_policy()
        group_id = None
    elif snapshot_kind == "ranker":
        label = RANKER_LABEL[decision_label]
        relevance_grade = label
        label_policy = make_ranker_label_policy()
        group_id = event["campaign_id"]
    else:
        raise ValueError(f"unsupported snapshot_kind: {snapshot_kind}")

    context = event["review_context"]

    row_hash = stable_hash(
        "|".join([
            snapshot_version,
            snapshot_kind,
            event["review_event_id"],
            event["pair_id"],
            event["feature_snapshot_id"],
        ])
    )

    return {
        "training_snapshot_id": f"train_snap_{snapshot_kind}_{row_hash}",
        "snapshot_version": snapshot_version,
        "created_at": created_at,
        "snapshot_kind": snapshot_kind,

        "label_policy": label_policy,
        "aggregation_policy": make_aggregation_policy(),

        "source_review_event_ids": [
            event["review_event_id"],
        ],

        "pair_id": event["pair_id"],
        "campaign_id": event["campaign_id"],
        "image_id": event["image_id"],
        "layout_spec_id": context.get("layout_spec_id"),
        "preview_renderer_version": context.get("preview_renderer_version"),
        "duplicate_group_id": event.get("duplicate_group_id"),

        "feature_snapshot_id": event["feature_snapshot_id"],

        "label": label,
        "label_status": "human_reviewed",
        "decision_label": decision_label,
        "issue_tags": event["decision"].get("issue_tags", []),
        "preference_rank": event["decision"].get("preference_rank"),

        "group_id": group_id,
        "relevance_grade": relevance_grade,

        "disagreement": {
            "has_disagreement": False,
            "disagreement_type": None,
            "excluded_from_critic": False,
        },

        "audit": {
            "notes": event["decision"].get("notes", ""),
            "source_bucket": context.get("source_bucket"),
            "queue_stage": context.get("queue_stage"),
        },
    }


def validate_feature_snapshot_links(
    events: list[dict[str, Any]],
    feature_snapshots: list[dict[str, Any]],
) -> None:
    feature_ids = {r["feature_snapshot_id"] for r in feature_snapshots}

    missing = [
        e["feature_snapshot_id"]
        for e in events
        if e["feature_snapshot_id"] not in feature_ids
    ]

    if missing:
        raise RuntimeError(f"missing feature_snapshot_id links: {missing[:10]}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-events", required=True)
    ap.add_argument("--feature-snapshots", required=True)
    ap.add_argument("--schema", default="schemas/training_snapshot.schema.json")
    ap.add_argument("--classifier-out", required=True)
    ap.add_argument("--ranker-out", required=True)
    ap.add_argument("--snapshot-version-prefix", default="train_labels_phase1a_v1")
    args = ap.parse_args()

    events = read_jsonl(Path(args.review_events))
    feature_snapshots = read_jsonl(Path(args.feature_snapshots))

    if not events:
        raise RuntimeError("empty review events")

    validate_feature_snapshot_links(events, feature_snapshots)

    schema = read_json(Path(args.schema))
    validator = Draft202012Validator(schema)

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    classifier_rows = [
        make_snapshot_row(
            event=e,
            snapshot_kind="classifier",
            snapshot_version=f"{args.snapshot_version_prefix}_classifier",
            created_at=created_at,
        )
        for e in events
    ]

    ranker_rows = [
        make_snapshot_row(
            event=e,
            snapshot_kind="ranker",
            snapshot_version=f"{args.snapshot_version_prefix}_ranker",
            created_at=created_at,
        )
        for e in events
    ]

    for rows, kind in [
        (classifier_rows, "classifier"),
        (ranker_rows, "ranker"),
    ]:
        for row in rows:
            errors = sorted(validator.iter_errors(row), key=lambda e: e.path)
            if errors:
                first = errors[0]
                raise RuntimeError(
                    f"{kind} TrainingSnapshot schema validation failed: "
                    f"id={row.get('training_snapshot_id')} "
                    f"path={list(first.path)} message={first.message}"
                )

    write_jsonl(Path(args.classifier_out), classifier_rows)
    write_jsonl(Path(args.ranker_out), ranker_rows)

    print(json.dumps({
        "event": "done",
        "review_events": args.review_events,
        "classifier_out": args.classifier_out,
        "ranker_out": args.ranker_out,
        "classifier_rows": len(classifier_rows),
        "ranker_rows": len(ranker_rows),
        "classifier_labels": dict(Counter(r["label"] for r in classifier_rows)),
        "ranker_labels": dict(Counter(r["label"] for r in ranker_rows)),
        "decision_labels": dict(Counter(r["decision_label"] for r in classifier_rows)),
        "label_status": dict(Counter(r["label_status"] for r in classifier_rows)),
        "schema": args.schema,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
