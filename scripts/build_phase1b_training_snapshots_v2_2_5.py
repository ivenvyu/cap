from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing jsonl: {path}")
    rows = []
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


def load_feature_id_map(pattern: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"no feature files matched: {pattern}")

    for file in files:
        for row in read_jsonl(Path(file)):
            old_id = row.get("previous_feature_snapshot_id")
            new_id = row.get("feature_snapshot_id")
            if old_id and new_id:
                out[str(old_id)] = {
                    "new_feature_snapshot_id": str(new_id),
                    "canonical_image_id": row.get("canonical_image_id"),
                    "duplicate_group_id": row.get("duplicate_group_id"),
                    "duplicate_canonicalization_status": row.get("duplicate_canonicalization_status"),
                }
    return out


def relink_rows(
    rows: list[dict[str, Any]],
    *,
    task: str,
    feature_id_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out = []
    missing = []

    for idx, row in enumerate(rows, start=1):
        old_fid = str(row.get("feature_snapshot_id", ""))
        m = feature_id_map.get(old_fid)

        if not m:
            missing.append({
                "training_snapshot_id": row.get("training_snapshot_id"),
                "pair_id": row.get("pair_id"),
                "campaign_id": row.get("campaign_id"),
                "old_feature_snapshot_id": old_fid,
                "reason": "missing_v2_2_5_feature_snapshot",
            })
            continue

        r = dict(row)
        r["snapshot_version"] = "v2_2_5"
        r["previous_training_snapshot_id"] = row.get("training_snapshot_id")
        r["training_snapshot_id"] = f"train_{task}_v2_2_5_{idx:05d}"
        r["previous_feature_snapshot_id"] = old_fid
        r["feature_snapshot_id"] = m["new_feature_snapshot_id"]
        r["canonical_image_id"] = m.get("canonical_image_id")
        r["duplicate_group_id"] = m.get("duplicate_group_id") or r.get("duplicate_group_id")
        r["duplicate_canonicalization_status"] = m.get("duplicate_canonicalization_status")
        r["feature_snapshot_version"] = "v2_2_5_duplicate_canonicalized"
        r["score_status"] = "diagnostic_only"
        r["threshold_status"] = "no_calibrated_threshold"
        r["relinked_at"] = utc_now()

        out.append(r)

    return out, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier-in", default="data/review/phase1b/v2_2_3/training_snapshot_classifier_v2_2_3.jsonl")
    ap.add_argument("--ranker-in", default="data/review/phase1b/v2_2_3/training_snapshot_ranker_v2_2_3.jsonl")
    ap.add_argument("--feature-glob", default="data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized/*.jsonl")
    ap.add_argument("--out-dir", default="data/review/phase1b/v2_2_5")
    ap.add_argument("--audit-out", default="audit/phase_1b/training_snapshot_v2_2_5_relink_audit.json")
    args = ap.parse_args()

    feature_id_map = load_feature_id_map(args.feature_glob)

    classifier_rows = read_jsonl(Path(args.classifier_in))
    ranker_rows = read_jsonl(Path(args.ranker_in))

    cls_out, cls_missing = relink_rows(classifier_rows, task="classifier", feature_id_map=feature_id_map)
    rank_out, rank_missing = relink_rows(ranker_rows, task="ranker", feature_id_map=feature_id_map)

    out_dir = Path(args.out_dir)
    cls_path = out_dir / "training_snapshot_classifier_v2_2_5.jsonl"
    rank_path = out_dir / "training_snapshot_ranker_v2_2_5.jsonl"

    write_jsonl(cls_path, cls_out)
    write_jsonl(rank_path, rank_out)

    audit = {
        "event": "training_snapshot_v2_2_5_relink_done",
        "created_at": utc_now(),
        "inputs": {
            "classifier_in": args.classifier_in,
            "ranker_in": args.ranker_in,
            "feature_glob": args.feature_glob,
        },
        "outputs": {
            "classifier": str(cls_path),
            "ranker": str(rank_path),
        },
        "classifier": {
            "input_rows": len(classifier_rows),
            "output_rows": len(cls_out),
            "missing_rows": len(cls_missing),
            "label_counts": dict(Counter(str(r.get("label")) for r in cls_out)),
            "campaign_counts": dict(Counter(str(r.get("campaign_id")) for r in cls_out)),
            "duplicate_canonicalization_status_counts": dict(Counter(str(r.get("duplicate_canonicalization_status")) for r in cls_out)),
        },
        "ranker": {
            "input_rows": len(ranker_rows),
            "output_rows": len(rank_out),
            "missing_rows": len(rank_missing),
            "label_counts": dict(Counter(str(r.get("label")) for r in rank_out)),
            "campaign_counts": dict(Counter(str(r.get("campaign_id")) for r in rank_out)),
            "duplicate_canonicalization_status_counts": dict(Counter(str(r.get("duplicate_canonicalization_status")) for r in rank_out)),
        },
        "missing": {
            "classifier": cls_missing[:50],
            "ranker": rank_missing[:50],
        },
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }

    write_json(Path(args.audit_out), audit)

    print(json.dumps({
        "event": "done",
        "classifier_out": str(cls_path),
        "ranker_out": str(rank_path),
        "audit_out": args.audit_out,
        "classifier_rows": len(cls_out),
        "ranker_rows": len(rank_out),
        "classifier_missing": len(cls_missing),
        "ranker_missing": len(rank_missing),
        "classifier_label_counts": audit["classifier"]["label_counts"],
        "classifier_duplicate_status_counts": audit["classifier"]["duplicate_canonicalization_status_counts"],
        "score_status": "diagnostic_only",
        "threshold_status": "no_calibrated_threshold",
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
