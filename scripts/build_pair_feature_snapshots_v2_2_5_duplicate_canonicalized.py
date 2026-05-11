from __future__ import annotations

import argparse
import glob
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PATH_DERIVED_FEATURES_TO_CANONICALIZE = [
    "path_has_architecture",
    "path_has_garden",
]


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
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def s(x: Any) -> str:
    if x is None:
        return ""
    return str(x)


def as_float_bool(x: Any) -> float:
    return 1.0 if bool(x) else 0.0


def load_canonical_map(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out = {}
    for row in rows:
        image_id = s(row.get("image_id"))
        if image_id:
            out[image_id] = row
    return out


def v2_2_5_feature_id(old_id: str, row: dict[str, Any]) -> str:
    if old_id.startswith("feat_v2_2_2__"):
        return old_id.replace("feat_v2_2_2__", "feat_v2_2_5__", 1)
    if old_id.startswith("feat_"):
        return "feat_v2_2_5__" + old_id.removeprefix("feat_")
    return (
        "feat_v2_2_5__"
        + s(row.get("campaign_id"))
        + "__"
        + s(row.get("image_id"))
        + "__"
        + s(row.get("layout_spec_id", "layout_default"))
    )


def canonicalize_row(
    row: dict[str, Any],
    *,
    canonical_by_image: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    image_id = s(row.get("image_id"))
    old_feature_snapshot_id = s(row.get("feature_snapshot_id"))
    old_features = row.get("features", {})
    if not isinstance(old_features, dict):
        old_features = {}

    features = dict(old_features)
    canonical = canonical_by_image.get(image_id)

    before_values = {
        k: old_features.get(k)
        for k in PATH_DERIVED_FEATURES_TO_CANONICALIZE
        if k in old_features
    }

    if canonical:
        duplicate_group_id = s(canonical.get("duplicate_group_id"))
        canonical_image_id = s(canonical.get("canonical_image_id"))
        is_canonical = bool(canonical.get("is_canonical"))
        duplicate_group_size = int(canonical.get("duplicate_group_size") or 1)
        duplicate_source_group_count = int(canonical.get("duplicate_source_group_count") or 0)
        duplicate_metadata_conflict = bool(canonical.get("metadata_conflict"))
        duplicate_has_architecture_source = bool(canonical.get("duplicate_has_architecture_source"))
        duplicate_has_garden_source = bool(canonical.get("duplicate_has_garden_source"))

        # 핵심: exact duplicate group에서는 path_has_*를 개별 파일 경로가 아니라 group-level source flag로 통일
        features["path_has_architecture"] = as_float_bool(duplicate_has_architecture_source)
        features["path_has_garden"] = as_float_bool(duplicate_has_garden_source)

        features["duplicate_group_size"] = float(duplicate_group_size)
        features["duplicate_is_member"] = 1.0
        features["duplicate_is_canonical"] = as_float_bool(is_canonical)
        features["duplicate_metadata_conflict"] = as_float_bool(duplicate_metadata_conflict)
        features["duplicate_source_group_count"] = float(duplicate_source_group_count)
        features["duplicate_has_architecture_source"] = as_float_bool(duplicate_has_architecture_source)
        features["duplicate_has_garden_source"] = as_float_bool(duplicate_has_garden_source)

        duplicate_status = "exact_duplicate_canonicalized"
    else:
        duplicate_group_id = s(row.get("duplicate_group_id")) or image_id
        canonical_image_id = image_id
        is_canonical = True
        duplicate_group_size = 1

        features["duplicate_group_size"] = 1.0
        features["duplicate_is_member"] = 0.0
        features["duplicate_is_canonical"] = 1.0
        features["duplicate_metadata_conflict"] = 0.0
        features["duplicate_source_group_count"] = 1.0
        features["duplicate_has_architecture_source"] = features.get("path_has_architecture", 0.0)
        features["duplicate_has_garden_source"] = features.get("path_has_garden", 0.0)

        duplicate_status = "not_in_exact_duplicate_canonical_map"

    new_row = dict(row)
    new_row["feature_snapshot_id"] = v2_2_5_feature_id(old_feature_snapshot_id, row)
    new_row["previous_feature_snapshot_id"] = old_feature_snapshot_id
    new_row["feature_snapshot_version"] = "v2_2_5_duplicate_canonicalized"
    new_row["duplicate_canonicalization_version"] = "duplicate_canonicalization_v2_2_5"
    new_row["duplicate_group_id"] = duplicate_group_id
    new_row["canonical_image_id"] = canonical_image_id
    new_row["is_canonical_duplicate_member"] = is_canonical
    new_row["duplicate_canonicalization_status"] = duplicate_status
    new_row["features"] = features

    after_values = {
        k: features.get(k)
        for k in PATH_DERIVED_FEATURES_TO_CANONICALIZE
        if k in features
    }

    audit = {
        "campaign_id": row.get("campaign_id"),
        "image_id": image_id,
        "duplicate_group_id": duplicate_group_id,
        "canonical_image_id": canonical_image_id,
        "is_canonical_duplicate_member": is_canonical,
        "old_feature_snapshot_id": old_feature_snapshot_id,
        "new_feature_snapshot_id": new_row["feature_snapshot_id"],
        "duplicate_canonicalization_status": duplicate_status,
        "path_derived_before": before_values,
        "path_derived_after": after_values,
        "path_derived_changed": before_values != after_values,
    }

    return new_row, audit


def build_markdown(summary: dict[str, Any]) -> str:
    lines = []
    lines.append("# v2.2.5 Canonicalized Feature Snapshot Report")
    lines.append("")
    lines.append("## 상태")
    lines.append("")
    lines.append("exact duplicate group의 metadata divergence를 줄이기 위해 v2.2.5 canonicalized feature snapshot을 생성했다.")
    lines.append("")
    lines.append("이 단계는 품질 threshold나 자동 accept/reject 규칙이 아니다.")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    lines.append(f"- input rows: {summary['input_rows']}")
    lines.append(f"- output rows: {summary['output_rows']}")
    lines.append(f"- exact duplicate canonicalized rows: {summary['status_counts'].get('exact_duplicate_canonicalized', 0)}")
    lines.append(f"- non-duplicate rows: {summary['status_counts'].get('not_in_exact_duplicate_canonical_map', 0)}")
    lines.append(f"- path-derived changed rows: {summary['path_derived_changed_rows']}")
    lines.append("")
    lines.append("## Canonicalization policy")
    lines.append("")
    lines.append("- exact duplicate group에 포함된 row에는 `canonical_image_id`를 부여한다.")
    lines.append("- `path_has_architecture`, `path_has_garden`은 개별 파일 경로가 아니라 duplicate group-level source flag로 통일한다.")
    lines.append("- `duplicate_has_architecture_source`, `duplicate_has_garden_source`, `duplicate_source_group_count`, `duplicate_metadata_conflict`를 feature로 추가한다.")
    lines.append("- DINOv2 campaign/family feature는 아직 재계산하지 않고, conflict audit 결과를 보존한다.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("- canonicalization은 이미지 품질 판단이 아니다.")
    lines.append("- canonicalization은 calibrated threshold가 아니다.")
    lines.append("- 이 snapshot만으로 production 성능을 주장하지 않는다.")
    lines.append("- DINOv2 anchor feature conflict는 다음 단계에서 group-aware recomputation 대상으로 남긴다.")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-glob", default="data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl")
    ap.add_argument("--canonical-map", default="data/ontology/duplicate_canonical_map_v2_2_5.jsonl")
    ap.add_argument("--out-dir", default="data/feature_snapshots/v2_2_5/phase1b_duplicate_canonicalized")
    ap.add_argument("--audit-out", default="audit/phase_1b/feature_snapshot_canonicalization_audit_v2_2_5.json")
    ap.add_argument("--report-out", default="audit/phase_1b/feature_snapshot_canonicalization_report_v2_2_5.md")
    args = ap.parse_args()

    canonical_by_image = load_canonical_map(Path(args.canonical_map))
    files = [Path(x) for x in sorted(glob.glob(args.input_glob))]
    if not files:
        raise RuntimeError(f"no input feature files matched: {args.input_glob}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_audit_rows = []
    input_rows = 0
    output_rows = 0
    out_files = []

    for fp in files:
        rows = read_jsonl(fp)
        input_rows += len(rows)

        new_rows = []
        for row in rows:
            new_row, audit = canonicalize_row(row, canonical_by_image=canonical_by_image)
            new_rows.append(new_row)
            all_audit_rows.append(audit)

        out_name = fp.name.replace("pair_feature_snapshots__", "pair_feature_snapshots_v2_2_5__")
        out_path = out_dir / out_name
        write_jsonl(out_path, new_rows)

        output_rows += len(new_rows)
        out_files.append(str(out_path))

    status_counts = Counter(r["duplicate_canonicalization_status"] for r in all_audit_rows)
    path_derived_changed_rows = sum(1 for r in all_audit_rows if r["path_derived_changed"])

    path_change_examples = [
        r for r in all_audit_rows
        if r["path_derived_changed"]
    ][:50]

    summary = {
        "event": "feature_snapshot_canonicalization_v2_2_5_done",
        "created_at": utc_now(),
        "input_glob": args.input_glob,
        "canonical_map": args.canonical_map,
        "out_dir": args.out_dir,
        "out_files": out_files,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "status_counts": dict(status_counts),
        "path_derived_changed_rows": path_derived_changed_rows,
        "path_change_examples": path_change_examples,
        "new_features_added": [
            "duplicate_group_size",
            "duplicate_is_member",
            "duplicate_is_canonical",
            "duplicate_metadata_conflict",
            "duplicate_source_group_count",
            "duplicate_has_architecture_source",
            "duplicate_has_garden_source",
        ],
        "canonicalized_path_features": PATH_DERIVED_FEATURES_TO_CANONICALIZE,
        "score_status": "diagnostic_only",
        "threshold_status": "diagnostic_warning_not_quality_threshold",
        "non_claims": [
            "canonicalization은 품질 threshold가 아니다.",
            "canonicalization은 accept/reject 자동 판정이 아니다.",
            "production model quality를 주장하지 않는다.",
        ],
    }

    write_json(Path(args.audit_out), summary)
    write_text(Path(args.report_out), build_markdown(summary))

    print(json.dumps({
        "event": "done",
        "out_dir": args.out_dir,
        "audit_out": args.audit_out,
        "report_out": args.report_out,
        "input_rows": input_rows,
        "output_rows": output_rows,
        "status_counts": dict(status_counts),
        "path_derived_changed_rows": path_derived_changed_rows,
        "score_status": "diagnostic_only",
        "threshold_status": "diagnostic_warning_not_quality_threshold",
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
