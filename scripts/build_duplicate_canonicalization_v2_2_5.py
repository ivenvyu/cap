from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PATH_DERIVED_FEATURES = [
    "path_has_architecture",
    "path_has_garden",
    "image_category_gallery",
    "image_category_tree",
    "image_category_flower",
    "image_category_course",
    "image_season_unknown",
]

DIAGNOSTIC_FEATURES_TO_AUDIT = [
    *PATH_DERIVED_FEATURES,
    "dinov2_campaign_margin",
    "dinov2_family_margin",
    "dinov2_campaign_pos_nn_sim",
    "dinov2_campaign_neg_nn_sim",
    "clip_positive_max_sim",
    "clip_positive_mean_sim",
    "clip_negative_max_sim",
    "clip_margin",
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


def image_id_of(row: dict[str, Any]) -> str:
    return s(row.get("image_id") or row.get("raw_image_id") or row.get("id"))


def unique_sorted(xs: list[Any]) -> list[str]:
    return sorted({s(x) for x in xs if x not in (None, "")})


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        image_id = image_id_of(row)
        if image_id:
            out[image_id] = row
    return out


def canonical_sort_key(member: dict[str, Any]) -> tuple[str, str, str, str]:
    image_id = s(member.get("image_id"))
    path = s(member.get("path") or member.get("resolved_path"))
    source_group = s(member.get("source_group"))
    filename = s(member.get("filename"))
    return (path, source_group, filename, image_id)


def build_duplicate_groups(
    duplicate_rows: list[dict[str, Any]],
    manifest_by_image: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in duplicate_rows:
        duplicate_group_id = s(row.get("duplicate_group_id"))
        image_id = image_id_of(row)
        if not duplicate_group_id or not image_id:
            continue

        manifest = manifest_by_image.get(image_id, {})
        merged = {
            **manifest,
            **row,
            "image_id": image_id,
            "duplicate_group_id": duplicate_group_id,
        }
        groups[duplicate_group_id].append(merged)

    # 실제 중복 그룹만 유지
    return {
        gid: members
        for gid, members in groups.items()
        if len({s(m.get("image_id")) for m in members}) > 1
    }


def group_summary(duplicate_group_id: str, members: list[dict[str, Any]]) -> dict[str, Any]:
    members_sorted = sorted(members, key=canonical_sort_key)
    canonical = members_sorted[0]
    canonical_image_id = s(canonical.get("image_id"))

    source_groups = unique_sorted([m.get("source_group") for m in members_sorted])
    categories = unique_sorted([m.get("category") for m in members_sorted])
    place_names = unique_sorted([m.get("place_name") for m in members_sorted])
    subject_names = unique_sorted([m.get("subject_name") for m in members_sorted])
    stems = unique_sorted([m.get("stem") for m in members_sorted])
    filenames = unique_sorted([m.get("filename") for m in members_sorted])
    paths = unique_sorted([m.get("path") for m in members_sorted])
    exact_hashes = unique_sorted([m.get("exact_file_sha256") for m in members_sorted])

    metadata_conflict_flags = {
        "source_group_conflict": len(source_groups) > 1,
        "category_conflict": len(categories) > 1,
        "place_name_conflict": len(place_names) > 1,
        "subject_name_conflict": len(subject_names) > 1,
        "path_conflict": len(paths) > 1,
        "filename_conflict": len(filenames) > 1,
        "exact_hash_conflict": len(exact_hashes) > 1,
    }

    return {
        "duplicate_group_id": duplicate_group_id,
        "canonical_image_id": canonical_image_id,
        "canonical_policy": "deterministic_sort_by_path_source_filename_image_id_v2_2_5",
        "duplicate_group_size": len(members_sorted),
        "member_image_ids": [s(m.get("image_id")) for m in members_sorted],
        "source_groups": source_groups,
        "categories": categories,
        "place_names": place_names,
        "subject_names": subject_names,
        "stems": stems,
        "filenames": filenames,
        "paths": paths,
        "exact_file_sha256_values": exact_hashes,
        "metadata_conflict_flags": metadata_conflict_flags,
        "metadata_conflict": any(metadata_conflict_flags.values()),
        "duplicate_has_architecture_source": "건축" in source_groups or "architecture" in source_groups,
        "duplicate_has_garden_source": "정원" in source_groups or "garden" in source_groups,
        "duplicate_source_group_count": len(source_groups),
    }


def build_canonical_map_rows(group_summaries: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for duplicate_group_id, g in sorted(group_summaries.items()):
        canonical_image_id = g["canonical_image_id"]
        for idx, image_id in enumerate(g["member_image_ids"], start=1):
            rows.append({
                "canonicalization_version": "duplicate_canonicalization_v2_2_5",
                "duplicate_group_id": duplicate_group_id,
                "image_id": image_id,
                "canonical_image_id": canonical_image_id,
                "is_canonical": image_id == canonical_image_id,
                "duplicate_member_index": idx,
                "duplicate_group_size": g["duplicate_group_size"],
                "canonical_policy": g["canonical_policy"],
                "source_groups": g["source_groups"],
                "categories": g["categories"],
                "place_names": g["place_names"],
                "paths": g["paths"],
                "metadata_conflict": g["metadata_conflict"],
                "metadata_conflict_flags": g["metadata_conflict_flags"],
                "duplicate_has_architecture_source": g["duplicate_has_architecture_source"],
                "duplicate_has_garden_source": g["duplicate_has_garden_source"],
                "duplicate_source_group_count": g["duplicate_source_group_count"],
            })

    return rows


def read_feature_snapshots(pattern: str) -> list[dict[str, Any]]:
    files = sorted(Path(".").glob(pattern)) if not any(ch in pattern for ch in "*?[]") else sorted(Path(".").glob(pattern))
    if not files:
        # Path.glob은 절대/상대 glob 섞일 때 제약이 있어 glob 모듈 fallback
        import glob
        files = [Path(x) for x in sorted(glob.glob(pattern))]

    rows: list[dict[str, Any]] = []
    for fp in files:
        if fp.exists():
            for row in read_jsonl(fp):
                row["_feature_file"] = str(fp)
                rows.append(row)
    return rows


def feature_value_signature(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, float):
        return f"{value:.10g}"
    return s(value)


def audit_feature_conflicts(
    feature_rows: list[dict[str, Any]],
    image_to_duplicate_group: dict[str, str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)

    for row in feature_rows:
        image_id = image_id_of(row)
        duplicate_group_id = s(row.get("duplicate_group_id")) or image_to_duplicate_group.get(image_id, "")
        if not duplicate_group_id:
            continue

        campaign_id = s(row.get("campaign_id"))
        features = row.get("features", {})
        if not isinstance(features, dict):
            continue

        for feature_name in DIAGNOSTIC_FEATURES_TO_AUDIT:
            if feature_name not in features:
                continue
            grouped[(campaign_id, duplicate_group_id, feature_name)].append({
                "image_id": image_id,
                "value": features.get(feature_name),
                "feature_snapshot_id": row.get("feature_snapshot_id"),
                "feature_file": row.get("_feature_file"),
            })

    conflicts: list[dict[str, Any]] = []
    for (campaign_id, duplicate_group_id, feature_name), items in sorted(grouped.items()):
        image_ids = sorted({s(x["image_id"]) for x in items})
        if len(image_ids) <= 1:
            continue

        value_by_image: dict[str, list[str]] = defaultdict(list)
        for item in items:
            value_by_image[s(item["image_id"])].append(feature_value_signature(item["value"]))

        collapsed = {
            image_id: sorted(set(values))
            for image_id, values in value_by_image.items()
        }
        unique_value_signatures = sorted({v for values in collapsed.values() for v in values})

        if len(unique_value_signatures) > 1:
            conflicts.append({
                "campaign_id": campaign_id,
                "duplicate_group_id": duplicate_group_id,
                "feature_name": feature_name,
                "feature_family": "path_derived" if feature_name in PATH_DERIVED_FEATURES else "diagnostic_numeric",
                "member_value_signatures": collapsed,
                "unique_value_signatures": unique_value_signatures,
                "member_count": len(image_ids),
                "interpretation": (
                    "동일 duplicate group 내부에서 feature 값이 다르다. "
                    "모델 입력 전에 canonical/group-level 처리 필요."
                ),
                "threshold_status": "diagnostic_warning_not_quality_threshold",
            })

    return conflicts


def build_markdown_report(audit: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# v2.2.5 Duplicate Canonicalization Report")
    lines.append("")
    lines.append("## 상태")
    lines.append("")
    lines.append("이 보고서는 exact duplicate group 단위의 canonical image_id와 metadata/feature 충돌을 진단한다.")
    lines.append("품질 threshold 또는 accept/reject 기준이 아니다.")
    lines.append("")
    lines.append("## 요약")
    lines.append("")
    lines.append(f"- duplicate groups: {audit['duplicate_group_count']}")
    lines.append(f"- canonical map rows: {audit['canonical_map_rows']}")
    lines.append(f"- metadata conflict groups: {audit['metadata_conflict_group_count']}")
    lines.append(f"- feature conflict rows: {audit['feature_conflict_count']}")
    lines.append("")
    lines.append("## 주요 결정")
    lines.append("")
    lines.append("1. exact duplicate group마다 deterministic canonical image_id를 지정한다.")
    lines.append("2. 동일 파일이 여러 경로/source_group에 존재하는 경우 단일 path feature로 덮지 않는다.")
    lines.append("3. `path_has_architecture`, `path_has_garden` 등 path-derived feature 충돌은 diagnostic warning으로 기록한다.")
    lines.append("4. 다음 단계에서 feature snapshot 생성 시 duplicate group-level canonical metadata를 사용할 수 있게 한다.")
    lines.append("")
    lines.append("## Non-claims")
    lines.append("")
    lines.append("- duplicate canonicalization은 품질 threshold가 아니다.")
    lines.append("- duplicate canonicalization은 이미지가 좋다/나쁘다를 판정하지 않는다.")
    lines.append("- feature conflict는 모델 입력 데이터 정규화 이슈이며, production 성능 claim이 아니다.")
    lines.append("")
    lines.append("## Metadata conflict examples")
    lines.append("")

    for item in audit.get("metadata_conflict_examples", [])[:10]:
        lines.append(f"### {item['duplicate_group_id']}")
        lines.append("")
        lines.append(f"- canonical_image_id: `{item['canonical_image_id']}`")
        lines.append(f"- members: {', '.join(item['member_image_ids'])}")
        lines.append(f"- source_groups: {', '.join(item['source_groups'])}")
        lines.append(f"- paths: {', '.join(item['paths'])}")
        lines.append("")

    lines.append("## Feature conflict examples")
    lines.append("")
    for item in audit.get("feature_conflict_examples", [])[:20]:
        lines.append(f"### {item['duplicate_group_id']} / {item['campaign_id']} / {item['feature_name']}")
        lines.append("")
        lines.append(f"- feature_family: {item['feature_family']}")
        lines.append(f"- values: `{json.dumps(item['member_value_signatures'], ensure_ascii=False)}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duplicate-groups", default="data/ontology/duplicate_groups_v1.jsonl")
    ap.add_argument("--manifest", default="data/ontology/raw_image_manifest_v2_2_1.jsonl")
    ap.add_argument("--feature-glob", default="data/feature_snapshots/v2_2_2/phase1b_dinov2_anchor/*.jsonl")
    ap.add_argument("--canonical-map-out", default="data/ontology/duplicate_canonical_map_v2_2_5.jsonl")
    ap.add_argument("--audit-out", default="audit/phase_1b/duplicate_feature_conflict_audit_v2_2_5.json")
    ap.add_argument("--report-out", default="audit/phase_1b/duplicate_canonicalization_report_v2_2_5.md")
    args = ap.parse_args()

    manifest_by_image = load_manifest(Path(args.manifest))
    duplicate_rows = read_jsonl(Path(args.duplicate_groups))
    groups = build_duplicate_groups(duplicate_rows, manifest_by_image)

    group_summaries = {
        gid: group_summary(gid, members)
        for gid, members in sorted(groups.items())
    }

    canonical_rows = build_canonical_map_rows(group_summaries)
    write_jsonl(Path(args.canonical_map_out), canonical_rows)

    image_to_duplicate_group = {
        row["image_id"]: row["duplicate_group_id"]
        for row in canonical_rows
    }

    feature_rows = read_feature_snapshots(args.feature_glob)
    feature_conflicts = audit_feature_conflicts(feature_rows, image_to_duplicate_group)

    metadata_conflict_groups = [
        g for g in group_summaries.values()
        if g["metadata_conflict"]
    ]

    feature_conflict_counter = Counter(x["feature_name"] for x in feature_conflicts)
    feature_conflict_family_counter = Counter(x["feature_family"] for x in feature_conflicts)

    audit = {
        "event": "duplicate_canonicalization_v2_2_5_done",
        "created_at": utc_now(),
        "inputs": {
            "duplicate_groups": args.duplicate_groups,
            "manifest": args.manifest,
            "feature_glob": args.feature_glob,
        },
        "outputs": {
            "canonical_map": args.canonical_map_out,
            "audit": args.audit_out,
            "report": args.report_out,
        },
        "duplicate_group_count": len(group_summaries),
        "canonical_map_rows": len(canonical_rows),
        "metadata_conflict_group_count": len(metadata_conflict_groups),
        "metadata_conflict_examples": metadata_conflict_groups[:20],
        "feature_conflict_count": len(feature_conflicts),
        "feature_conflict_counts_by_feature": dict(feature_conflict_counter),
        "feature_conflict_counts_by_family": dict(feature_conflict_family_counter),
        "feature_conflict_examples": feature_conflicts[:50],
        "score_status": "diagnostic_only",
        "threshold_status": "diagnostic_warning_not_quality_threshold",
        "non_claims": [
            "duplicate canonicalization은 품질 threshold가 아니다.",
            "이미지 accept/reject 판단으로 사용하지 않는다.",
            "production model quality를 주장하지 않는다.",
        ],
    }

    write_json(Path(args.audit_out), audit)
    write_text(Path(args.report_out), build_markdown_report(audit))

    print(json.dumps({
        "event": "done",
        "canonical_map_out": args.canonical_map_out,
        "audit_out": args.audit_out,
        "report_out": args.report_out,
        "duplicate_group_count": audit["duplicate_group_count"],
        "canonical_map_rows": audit["canonical_map_rows"],
        "metadata_conflict_group_count": audit["metadata_conflict_group_count"],
        "feature_conflict_count": audit["feature_conflict_count"],
        "feature_conflict_counts_by_feature": audit["feature_conflict_counts_by_feature"],
        "score_status": audit["score_status"],
        "threshold_status": audit["threshold_status"],
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
