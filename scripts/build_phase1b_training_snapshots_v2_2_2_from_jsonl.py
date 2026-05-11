from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

SUPPORTED_LABELS = set(CLASSIFIER_LABEL) | set(RANKER_LABEL)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def stable_hash(value: str, n: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:n]


def load_feature_snapshots(paths: list[Path]) -> dict[str, dict[str, Any]]:
    by_pair: dict[str, dict[str, Any]] = {}
    duplicate_pairs: list[str] = []

    for path in paths:
        for row in read_jsonl(path):
            pair_id = row.get("pair_id")
            if not isinstance(pair_id, str) or not pair_id:
                raise RuntimeError(f"feature snapshot missing pair_id: {path}")
            if pair_id in by_pair:
                duplicate_pairs.append(pair_id)
                continue
            by_pair[pair_id] = row

    if duplicate_pairs:
        sample = ", ".join(sorted(set(duplicate_pairs))[:10])
        raise RuntimeError(f"duplicate feature snapshot pair_id values: {sample}")

    return by_pair


def make_aggregation_policy() -> dict[str, Any]:
    return {
        "policy_name": "training_snapshot_aggregation_v1",
        "primary_key": ["pair_id", "layout_spec_id", "preview_renderer_version"],
        "same_annotator_overwrite": "most_recent",
        "multi_annotator_aggregation": "not_applicable",
        "tie_break": None,
        "log_disagreement": True,
    }


def make_classifier_label_policy() -> dict[str, Any]:
    return {
        "policy_name": "classifier_binary_v1",
        "label_mapping": dict(CLASSIFIER_LABEL),
    }


def make_ranker_label_policy() -> dict[str, Any]:
    return {
        "policy_name": "ranker_relevance_v1",
        "label_mapping": dict(RANKER_LABEL),
    }


def make_snapshot_row(
    *,
    event: dict[str, Any],
    feature_snapshot: dict[str, Any],
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

    context = event.get("review_context", {})
    old_feature_snapshot_id = event.get("feature_snapshot_id")
    new_feature_snapshot_id = feature_snapshot["feature_snapshot_id"]

    row_hash = stable_hash(
        "|".join([
            snapshot_version,
            snapshot_kind,
            event["review_event_id"],
            event["pair_id"],
            new_feature_snapshot_id,
        ])
    )

    return {
        "training_snapshot_id": f"train_snap_{snapshot_kind}_{row_hash}",
        "snapshot_version": snapshot_version,
        "created_at": created_at,
        "snapshot_kind": snapshot_kind,
        "label_policy": label_policy,
        "aggregation_policy": make_aggregation_policy(),
        "source_review_event_ids": [event["review_event_id"]],
        "pair_id": event["pair_id"],
        "campaign_id": event["campaign_id"],
        "image_id": event["image_id"],
        "layout_spec_id": context.get("layout_spec_id") or feature_snapshot.get("layout_spec_id"),
        "preview_renderer_version": context.get("preview_renderer_version") or feature_snapshot.get("preview_renderer_version"),
        "duplicate_group_id": event.get("duplicate_group_id") or feature_snapshot.get("duplicate_group_id"),
        "feature_snapshot_id": new_feature_snapshot_id,
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
            "relink_policy": "pair_id_exact_match_v2_2_2",
            "old_feature_snapshot_id": old_feature_snapshot_id,
            "new_feature_snapshot_id": new_feature_snapshot_id,
            "feature_snapshot_version": feature_snapshot.get("snapshot_version"),
            "feature_status": feature_snapshot.get("feature_status"),
        },
    }


def validate_rows(rows: list[dict[str, Any]], schema_path: Path, kind: str) -> None:
    schema = read_json(schema_path)
    validator = Draft202012Validator(schema)
    for row in rows:
        errors = sorted(validator.iter_errors(row), key=lambda e: e.path)
        if errors:
            first = errors[0]
            raise RuntimeError(
                f"{kind} TrainingSnapshot schema validation failed: "
                f"id={row.get('training_snapshot_id')} "
                f"path={list(first.path)} message={first.message}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--review-events",
        default="data/review/phase1b/filtered/review_events_phase1b_v1__filtered.jsonl",
    )
    ap.add_argument(
        "--feature-snapshots-glob",
        default="data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl",
    )
    ap.add_argument("--schema", default="schemas/training_snapshot.schema.json")
    ap.add_argument("--out-dir", default="data/review/phase1b/v2_2_2")
    ap.add_argument("--snapshot-version-prefix", default="train_labels_phase1b_v2_2_2")
    args = ap.parse_args()

    review_path = Path(args.review_events)
    if not review_path.exists():
        raise RuntimeError(f"missing review events file: {review_path}")

    feature_paths = sorted(Path().glob(args.feature_snapshots_glob))
    if not feature_paths:
        raise RuntimeError(f"no feature snapshot files matched: {args.feature_snapshots_glob}")

    events_all = read_jsonl(review_path)
    unsupported = [e for e in events_all if e.get("decision", {}).get("label") not in SUPPORTED_LABELS]
    events = [e for e in events_all if e.get("decision", {}).get("label") in SUPPORTED_LABELS]

    features_by_pair = load_feature_snapshots(feature_paths)
    missing_pair_ids = [e["pair_id"] for e in events if e["pair_id"] not in features_by_pair]
    if missing_pair_ids:
        sample = ", ".join(missing_pair_ids[:10])
        raise RuntimeError(f"missing v2.2.2 feature snapshots for review pair_id values: {sample}")

    created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    classifier_rows = [
        make_snapshot_row(
            event=e,
            feature_snapshot=features_by_pair[e["pair_id"]],
            snapshot_kind="classifier",
            snapshot_version=f"{args.snapshot_version_prefix}_classifier",
            created_at=created_at,
        )
        for e in events
    ]
    ranker_rows = [
        make_snapshot_row(
            event=e,
            feature_snapshot=features_by_pair[e["pair_id"]],
            snapshot_kind="ranker",
            snapshot_version=f"{args.snapshot_version_prefix}_ranker",
            created_at=created_at,
        )
        for e in events
    ]

    validate_rows(classifier_rows, Path(args.schema), "classifier")
    validate_rows(ranker_rows, Path(args.schema), "ranker")

    out_dir = Path(args.out_dir)
    classifier_out = out_dir / "training_snapshot_phase1b_classifier_v2_2_2.jsonl"
    ranker_out = out_dir / "training_snapshot_phase1b_ranker_v2_2_2.jsonl"
    summary_out = out_dir / "training_snapshot_phase1b_v2_2_2_relink_summary.json"

    write_jsonl(classifier_out, classifier_rows)
    write_jsonl(ranker_out, ranker_rows)

    summary = {
        "event": "done",
        "review_events": str(review_path),
        "feature_snapshots_glob": args.feature_snapshots_glob,
        "feature_snapshot_files": [str(p) for p in feature_paths],
        "out_dir": str(out_dir),
        "classifier_out": str(classifier_out),
        "ranker_out": str(ranker_out),
        "classifier_rows": len(classifier_rows),
        "ranker_rows": len(ranker_rows),
        "unsupported_events_excluded": len(unsupported),
        "classifier_labels": dict(Counter(r["label"] for r in classifier_rows)),
        "ranker_labels": dict(Counter(r["label"] for r in ranker_rows)),
        "decision_labels": dict(Counter(r["decision_label"] for r in classifier_rows)),
        "campaign_counts": dict(Counter(r["campaign_id"] for r in classifier_rows)),
        "schema": args.schema,
        "relink_policy": "pair_id_exact_match_v2_2_2",
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
