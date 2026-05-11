from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INDEX_PATH = "data/search/ontology_effective_search_index_v1.jsonl"
REPORT_PATH = "audit/ontology/ontology_effective_search_index_query_smoke_v1.summary.json"


CAMPAIGN_QUERIES = {
    "봄 식물 프로그램": "phase1b_botanical_spring_program",
    "여름 정원 산책": "phase1b_summer_garden_walk",
    "가을 정원 산책": "phase1b_autumn_garden_walk",
    "겨울 실내 전시": "phase1b_indoor_gallery_winter_art",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise RuntimeError(f"missing index file: {path}")

    rows = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise RuntimeError(f"invalid JSONL at line {i}: {path}") from e

    if not rows:
        raise RuntimeError(f"empty index file: {path}")

    return rows


def image_subject(row: dict[str, Any]) -> str | None:
    image = row.get("image") or {}
    subject = image.get("subject_name")
    return str(subject) if subject else None


def image_path(row: dict[str, Any]) -> str | None:
    image = row.get("image") or {}
    path = image.get("path")
    return str(path) if path else None


def campaign_conflicts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts = []

    for row in rows:
        image = row.get("image") or {}
        image_id = image.get("image_id")
        effective = set(row.get("effective_campaigns") or [])

        for ex in row.get("excluded_campaigns") or []:
            campaign_id = ex.get("campaign_id")
            if campaign_id in effective:
                conflicts.append(
                    {
                        "image_id": image_id,
                        "path": image.get("path"),
                        "subject_name": image.get("subject_name"),
                        "campaign_id": campaign_id,
                        "reason": ex.get("reason"),
                    }
                )

    return conflicts


def summarize_campaign(rows: list[dict[str, Any]], campaign_id: str) -> dict[str, Any]:
    effective_rows = []
    excluded_rows = []

    for row in rows:
        effective = set(row.get("effective_campaigns") or [])

        if campaign_id in effective:
            effective_rows.append(row)

        for ex in row.get("excluded_campaigns") or []:
            if ex.get("campaign_id") == campaign_id:
                excluded_rows.append(
                    {
                        "image_id": (row.get("image") or {}).get("image_id"),
                        "path": image_path(row),
                        "subject_name": ex.get("subject_name") or image_subject(row),
                        "reason": ex.get("reason"),
                    }
                )

    effective_subjects = Counter(image_subject(r) or "(no_subject)" for r in effective_rows)
    excluded_subjects = Counter(str(r.get("subject_name") or "(no_subject)") for r in excluded_rows)

    examples_effective = []
    for row in effective_rows[:12]:
        examples_effective.append(
            {
                "image_id": (row.get("image") or {}).get("image_id"),
                "path": image_path(row),
                "subject_name": image_subject(row),
                "plant_bloom_prior": row.get("plant_bloom_prior"),
            }
        )

    return {
        "campaign_id": campaign_id,
        "effective_count": len(effective_rows),
        "excluded_count": len(excluded_rows),
        "effective_subject_counts": dict(sorted(effective_subjects.items())),
        "excluded_subject_counts": dict(sorted(excluded_subjects.items())),
        "effective_examples": examples_effective,
        "excluded_examples": excluded_rows[:20],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=INDEX_PATH)
    ap.add_argument("--report-out", default=REPORT_PATH)
    args = ap.parse_args()

    rows = load_jsonl(Path(args.index))

    conflicts = campaign_conflicts(rows)

    campaign_summaries = {
        query_name: summarize_campaign(rows, campaign_id)
        for query_name, campaign_id in CAMPAIGN_QUERIES.items()
    }

    image_count = len(rows)
    images_with_effective_campaigns = sum(1 for r in rows if r.get("effective_campaigns"))
    images_with_exclusions = sum(1 for r in rows if r.get("excluded_campaigns"))

    effective_campaign_counts = Counter()
    excluded_campaign_counts = Counter()

    for row in rows:
        for campaign_id in row.get("effective_campaigns") or []:
            effective_campaign_counts[str(campaign_id)] += 1

        for ex in row.get("excluded_campaigns") or []:
            excluded_campaign_counts[str(ex.get("campaign_id"))] += 1

    report = {
        "report_name": "ontology_effective_search_index_query_smoke_v1",
        "index": args.index,
        "image_count": image_count,
        "images_with_effective_campaigns": images_with_effective_campaigns,
        "images_with_exclusions": images_with_exclusions,
        "effective_campaign_counts": dict(sorted(effective_campaign_counts.items())),
        "excluded_campaign_counts": dict(sorted(excluded_campaign_counts.items())),
        "campaign_summaries": campaign_summaries,
        "consistency_check": {
            "excluded_and_effective_same_campaign_conflicts": len(conflicts),
            "conflict_examples": conflicts[:20],
            "passed": len(conflicts) == 0,
        },
        "해석": (
            "이 smoke test는 effective search index에서 flower-season exclusion이 적용된 "
            "campaign-image pair가 effective_campaigns에 동시에 남아 있지 않은지 확인한다. "
            "검색 품질의 최종 평가는 아니며, DB export 일관성 확인용이다."
        ),
    }

    out = Path(args.report_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))

    if conflicts:
        raise RuntimeError("effective index consistency check failed: excluded campaign still appears in effective_campaigns")


if __name__ == "__main__":
    main()
