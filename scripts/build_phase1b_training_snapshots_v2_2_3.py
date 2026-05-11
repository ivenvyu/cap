from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing jsonl: {path}")
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"missing yaml: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"yaml root must be object: {path}")
    return data


def as_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def merge_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        as_str(row.get("pair_id")),
        as_str(row.get("layout_spec_id", "layout_default")),
        as_str(row.get("preview_renderer_version", "")),
    )


def normalize_row(
    row: dict[str, Any],
    *,
    task: str,
    source: str,
    campaign_briefs: dict[str, Any],
    priority: int,
    index: int,
) -> dict[str, Any]:
    campaign_id = as_str(row.get("campaign_id"))
    campaign_cfg = campaign_briefs.get(campaign_id, {})
    campaign_status = campaign_cfg.get("campaign_status", "unknown")
    exclude_from_model_quality_claims = bool(campaign_cfg.get("exclude_from_model_quality_claims", False))
    use_for_training = bool(campaign_cfg.get("use_for_training", True))

    out = dict(row)
    out["snapshot_version"] = "v2_2_3"
    out["task"] = task
    out["source_snapshot"] = source
    out["merge_priority"] = priority
    out["merged_at"] = utc_now()

    out["campaign_status"] = campaign_status
    out["campaign_title_ko"] = campaign_cfg.get("title_ko", "")
    out["exclude_from_model_quality_claims"] = exclude_from_model_quality_claims
    out["use_for_training"] = use_for_training

    out["score_status"] = "diagnostic_only"
    out["threshold_status"] = "no_calibrated_threshold"

    old_id = as_str(out.get("training_snapshot_id"))
    out["previous_training_snapshot_id"] = old_id
    out["training_snapshot_id"] = f"train_{task}_v2_2_3_{index:05d}"

    return out


def merge_rows(
    base_rows: list[dict[str, Any]],
    triage_rows: list[dict[str, Any]],
    *,
    task: str,
    campaign_briefs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_by_key: dict[tuple[str, str, str], str] = {}
    duplicate_events: list[dict[str, Any]] = []

    for i, row in enumerate(base_rows, start=1):
        key = merge_key(row)
        normalized = normalize_row(
            row,
            task=task,
            source="v2_2_2_base",
            campaign_briefs=campaign_briefs,
            priority=1,
            index=i,
        )
        merged[key] = normalized
        source_by_key[key] = "v2_2_2_base"

    replaced = 0
    added = 0

    for j, row in enumerate(triage_rows, start=1):
        key = merge_key(row)
        existed = key in merged
        normalized = normalize_row(
            row,
            task=task,
            source="v2_2_2_triage_25",
            campaign_briefs=campaign_briefs,
            priority=2,
            index=len(merged) + j,
        )

        if existed:
            replaced += 1
            duplicate_events.append({
                "key": list(key),
                "previous_source": source_by_key.get(key),
                "new_source": "v2_2_2_triage_25",
                "policy": "triage_label_overrides_base_label",
            })
        else:
            added += 1

        merged[key] = normalized
        source_by_key[key] = "v2_2_2_triage_25"

    rows = list(merged.values())
    rows.sort(key=lambda r: (as_str(r.get("campaign_id")), as_str(r.get("pair_id")), as_str(r.get("layout_spec_id"))))

    # 재정렬 후 id 안정화
    for idx, row in enumerate(rows, start=1):
        row["training_snapshot_id"] = f"train_{task}_v2_2_3_{idx:05d}"

    decision_counts = Counter(as_str(r.get("decision_label", "")) for r in rows)
    label_counts = Counter(as_str(r.get("label", "")) for r in rows)
    campaign_counts = Counter(as_str(r.get("campaign_id", "")) for r in rows)
    source_counts = Counter(as_str(r.get("source_snapshot", "")) for r in rows)
    campaign_status_counts = Counter(as_str(r.get("campaign_status", "")) for r in rows)

    quality_claim_eligible_rows = [
        r for r in rows
        if not bool(r.get("exclude_from_model_quality_claims", False))
    ]
    training_eligible_rows = [
        r for r in rows
        if bool(r.get("use_for_training", True))
    ]

    audit = {
        "task": task,
        "base_rows": len(base_rows),
        "triage_rows": len(triage_rows),
        "merged_rows": len(rows),
        "triage_added": added,
        "triage_replaced_existing": replaced,
        "merge_key": ["pair_id", "layout_spec_id", "preview_renderer_version"],
        "merge_policy": "triage_label_overrides_base_label_on_same_key",
        "decision_counts": dict(decision_counts),
        "label_counts": dict(label_counts),
        "campaign_counts": dict(campaign_counts),
        "source_counts": dict(source_counts),
        "campaign_status_counts": dict(campaign_status_counts),
        "quality_claim_eligible_rows": len(quality_claim_eligible_rows),
        "training_eligible_rows": len(training_eligible_rows),
        "duplicate_events": duplicate_events,
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }

    return rows, audit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-classifier", default="data/review/phase1b/v2_2_2/training_snapshot_phase1b_classifier_v2_2_2.jsonl")
    ap.add_argument("--base-ranker", default="data/review/phase1b/v2_2_2/training_snapshot_phase1b_ranker_v2_2_2.jsonl")
    ap.add_argument("--triage-classifier", default="data/review/phase1b/v2_2_2/triage/training_snapshot_classifier_v2_2_2_triage_25.jsonl")
    ap.add_argument("--triage-ranker", default="data/review/phase1b/v2_2_2/triage/training_snapshot_ranker_v2_2_2_triage_25.jsonl")
    ap.add_argument("--campaign-briefs", default="configs/campaign_review_briefs_v1.yaml")
    ap.add_argument("--out-dir", default="data/review/phase1b/v2_2_3")
    ap.add_argument("--audit-out", default="audit/phase_1b/training_snapshot_v2_2_3_merge_audit.json")
    args = ap.parse_args()

    campaign_brief_doc = read_yaml(Path(args.campaign_briefs))
    campaign_briefs = campaign_brief_doc.get("campaigns", {})
    if not isinstance(campaign_briefs, dict):
        raise RuntimeError("campaign_review_briefs_v1.yaml must contain campaigns object")

    base_classifier = read_jsonl(Path(args.base_classifier))
    base_ranker = read_jsonl(Path(args.base_ranker))
    triage_classifier = read_jsonl(Path(args.triage_classifier))
    triage_ranker = read_jsonl(Path(args.triage_ranker))

    classifier_rows, classifier_audit = merge_rows(
        base_classifier,
        triage_classifier,
        task="classifier",
        campaign_briefs=campaign_briefs,
    )
    ranker_rows, ranker_audit = merge_rows(
        base_ranker,
        triage_ranker,
        task="ranker",
        campaign_briefs=campaign_briefs,
    )

    out_dir = Path(args.out_dir)
    classifier_out = out_dir / "training_snapshot_classifier_v2_2_3.jsonl"
    ranker_out = out_dir / "training_snapshot_ranker_v2_2_3.jsonl"

    write_jsonl(classifier_out, classifier_rows)
    write_jsonl(ranker_out, ranker_rows)

    audit = {
        "event": "training_snapshot_v2_2_3_merge_done",
        "created_at": utc_now(),
        "inputs": {
            "base_classifier": args.base_classifier,
            "base_ranker": args.base_ranker,
            "triage_classifier": args.triage_classifier,
            "triage_ranker": args.triage_ranker,
            "campaign_briefs": args.campaign_briefs,
        },
        "outputs": {
            "classifier": str(classifier_out),
            "ranker": str(ranker_out),
        },
        "classifier": classifier_audit,
        "ranker": ranker_audit,
        "non_claims": [
            "calibrated accept/reject threshold를 만들지 않는다.",
            "production model quality를 주장하지 않는다.",
            "coverage_gap campaign을 일반 모델 품질 claim에 섞지 않는다.",
        ],
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }

    write_json(Path(args.audit_out), audit)

    print(json.dumps({
        "event": "done",
        "classifier_out": str(classifier_out),
        "ranker_out": str(ranker_out),
        "audit_out": args.audit_out,
        "classifier_rows": len(classifier_rows),
        "ranker_rows": len(ranker_rows),
        "classifier_label_counts": classifier_audit["label_counts"],
        "ranker_label_counts": ranker_audit["label_counts"],
        "classifier_source_counts": classifier_audit["source_counts"],
        "ranker_source_counts": ranker_audit["source_counts"],
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
