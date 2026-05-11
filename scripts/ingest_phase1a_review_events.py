from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from jsonschema import Draft202012Validator


DECISION_INT_TO_LABEL = {
    0: "reject",
    1: "acceptable",
    2: "accept",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_issue_tags(path: Path) -> set[str]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        raise RuntimeError(f"empty issue tag config: {path}")

    tags = set()
    for _, obj in data["categories"].items():
        for tag in obj["tags"]:
            tags.add(tag["id"])
    return tags


def parse_issue_tags(value: Any) -> list[str]:
    if pd.isna(value):
        return []

    s = str(value).replace("\n", "").strip().strip('"').strip()
    if not s:
        return []

    return [x.strip() for x in s.split(",") if x.strip()]


def nullable_number(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value)


def nullable_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(value)


def nullable_string(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    s = str(value)
    if s.lower() == "nan":
        return None
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review-csv", required=True)
    ap.add_argument("--feature-snapshots", required=True)
    ap.add_argument("--schema", default="schemas/review_event.schema.json")
    ap.add_argument("--issue-tags-config", default="configs/issue_tags_v1.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--annotator-id", default="human_001")
    ap.add_argument("--queue-version", default="review_queue_phase1a_v1")
    args = ap.parse_args()

    df = pd.read_csv(args.review_csv)
    feature_rows = load_jsonl(Path(args.feature_snapshots))
    feature_by_id = {r["feature_snapshot_id"]: r for r in feature_rows}

    schema = load_json(Path(args.schema))
    validator = Draft202012Validator(schema)

    allowed_issue_tags = load_issue_tags(Path(args.issue_tags_config))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    events: list[dict[str, Any]] = []

    required_review_cols = [
        "queue_row_id",
        "queue_stage",
        "source_bucket",
        "campaign_id",
        "pair_id",
        "image_id",
        "duplicate_group_id",
        "feature_snapshot_id",
        "layout_spec_id",
        "decision",
        "issue_tags",
    ]

    for col in required_review_cols:
        if col not in df.columns:
            raise RuntimeError(f"missing required review csv column: {col}")

    for _, row in df.iterrows():
        feature_snapshot_id = str(row["feature_snapshot_id"])
        if feature_snapshot_id not in feature_by_id:
            raise RuntimeError(f"feature_snapshot_id not found: {feature_snapshot_id}")

        decision_int = nullable_int(row["decision"])
        if decision_int not in DECISION_INT_TO_LABEL:
            raise RuntimeError(f"bad decision value for row {row['queue_row_id']}: {row['decision']}")

        label = DECISION_INT_TO_LABEL[decision_int]
        issue_tags = parse_issue_tags(row["issue_tags"])

        for tag in issue_tags:
            if tag not in allowed_issue_tags:
                raise RuntimeError(f"invalid issue tag {tag} in row {row['queue_row_id']}")

        if label == "reject" and not issue_tags:
            raise RuntimeError(f"reject row without issue_tags: {row['queue_row_id']}")

        notes = "" if ("notes" not in df.columns or pd.isna(row.get("notes"))) else str(row.get("notes"))

        event = {
            "review_event_id": f"rev_{row['queue_row_id']}",
            "timestamp": now,
            "annotator_id": args.annotator_id,

            "pair_id": str(row["pair_id"]),
            "campaign_id": str(row["campaign_id"]),
            "image_id": str(row["image_id"]),
            "duplicate_group_id": nullable_string(row["duplicate_group_id"]),

            "review_context": {
                "queue_version": args.queue_version,
                "queue_stage": str(row["queue_stage"]),
                "source_bucket": str(row["source_bucket"]),
                "preview_renderer_version": None,
                "layout_spec_id": nullable_string(row["layout_spec_id"]),
            },

            "decision": {
                "label": label,
                "issue_tags": issue_tags,
                "preference_rank": nullable_number(row.get("preference_rank")) if "preference_rank" in df.columns else None,
                "notes": notes,
            },

            "feature_snapshot_id": feature_snapshot_id,

            "model_score_at_review": {
                "lightgbm_classifier_v0": None,
                "lightgbm_ranker_v0": None,
                "critic_v0": None,
                "clip_positive_max_sim": nullable_number(row.get("clip_positive_max_sim")),
                "clip_negative_max_sim": nullable_number(row.get("clip_negative_max_sim")),
                "clip_margin": nullable_number(row.get("clip_margin")),
                "required_region_safe_min": nullable_number(row.get("required_region_safe_min")),
                "required_region_safe_mean": nullable_number(row.get("required_region_safe_mean")),
                "dinov2_campaign_pos_nn_sim": None,
            },
        }

        errors = sorted(validator.iter_errors(event), key=lambda e: e.path)
        if errors:
            first = errors[0]
            raise RuntimeError(
                f"ReviewEvent schema validation failed for {row['queue_row_id']}: "
                f"path={list(first.path)} message={first.message}"
            )

        events.append(event)

    with out_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({
        "event": "done",
        "review_csv": args.review_csv,
        "out": str(out_path),
        "rows": len(events),
        "label_counts": pd.Series([e["decision"]["label"] for e in events]).value_counts().to_dict(),
        "schema": args.schema,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
