from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INDEX_VERSION = "ontology_effective_search_index_v1"
SCORE_STATUS = "diagnostic_only"


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return (
        conn.execute(
            """
            SELECT COUNT(*)
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name = ?
            """,
            (name,),
        ).fetchone()[0]
        > 0
    )


def fetch_rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def load_tags(conn: sqlite3.Connection) -> tuple[dict[str, dict[str, list[str]]], dict[str, list[dict[str, Any]]]]:
    if not table_exists(conn, "image_tag_assertions"):
        return {}, {}

    rows = fetch_rows(
        conn,
        """
        SELECT
            a.image_id,
            v.axis_id,
            v.tag_name,
            a.label_source,
            a.confidence_status
        FROM image_tag_assertions a
        JOIN tag_values v
          ON a.tag_id = v.tag_id
        ORDER BY a.image_id, v.axis_id, v.tag_name
        """
    )

    tag_map: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    assertions: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        image_id = str(r["image_id"])
        axis_id = str(r["axis_id"])
        tag_name = str(r["tag_name"])

        tag_map[image_id][axis_id].append(tag_name)
        assertions[image_id].append(
            {
                "axis_id": axis_id,
                "tag_name": tag_name,
                "label_source": r.get("label_source"),
                "confidence_status": r.get("confidence_status"),
            }
        )

    tag_map_clean = {
        image_id: {axis: sorted(set(tags)) for axis, tags in axes.items()}
        for image_id, axes in tag_map.items()
    }

    return tag_map_clean, assertions


def load_plant_bloom(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    required = ["plant_names", "plant_entities", "plant_bloom_priors"]
    if not all(table_exists(conn, t) for t in required):
        return {}

    rows = fetch_rows(
        conn,
        """
        SELECT
            n.name,
            n.plant_id,
            e.plant_type,
            b.bloom_value,
            b.bloom_type,
            b.confidence_status
        FROM plant_names n
        JOIN plant_entities e
          ON n.plant_id = e.plant_id
        LEFT JOIN plant_bloom_priors b
          ON n.plant_id = b.plant_id
        ORDER BY n.name, b.bloom_type, b.bloom_value
        """
    )

    out: dict[str, dict[str, Any]] = {}

    for r in rows:
        name = str(r["name"])
        item = out.setdefault(
            name,
            {
                "plant_id": str(r["plant_id"]),
                "plant_type": str(r["plant_type"]),
                "bloom_values": [],
                "bloom_months": [],
                "bloom_seasons": [],
                "confidence_statuses": [],
            },
        )

        value = r.get("bloom_value")
        btype = r.get("bloom_type")

        if value is not None:
            value = str(value)
            item["bloom_values"].append(value)

            if btype == "month":
                item["bloom_months"].append(value)
            elif btype == "season":
                item["bloom_seasons"].append(value)

        if r.get("confidence_status") is not None:
            item["confidence_statuses"].append(str(r["confidence_status"]))

    for item in out.values():
        for key in ["bloom_values", "bloom_months", "bloom_seasons", "confidence_statuses"]:
            item[key] = sorted(set(item[key]))

    return out


def load_effective_campaigns(conn: sqlite3.Connection) -> dict[str, list[str]]:
    if not table_exists(conn, "v_effective_retrieval_candidates_v1"):
        return {}

    rows = fetch_rows(
        conn,
        """
        SELECT image_id, campaign_id
        FROM v_effective_retrieval_candidates_v1
        ORDER BY image_id, campaign_id
        """
    )

    out: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        out[str(r["image_id"])].append(str(r["campaign_id"]))

    return {k: sorted(set(v)) for k, v in out.items()}


def load_excluded_campaigns(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    if not table_exists(conn, "v_effective_retrieval_candidates_excluded_v1"):
        return {}

    rows = fetch_rows(
        conn,
        """
        SELECT
            image_id,
            campaign_id,
            excluded_subject_name,
            exclusion_reason,
            exclusion_source_id
        FROM v_effective_retrieval_candidates_excluded_v1
        ORDER BY image_id, campaign_id
        """
    )

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        out[str(r["image_id"])].append(
            {
                "campaign_id": r.get("campaign_id"),
                "subject_name": r.get("excluded_subject_name"),
                "reason": r.get("exclusion_reason"),
                "source_id": r.get("exclusion_source_id"),
            }
        )

    return dict(out)


def load_cue_scores(conn: sqlite3.Connection) -> dict[str, list[dict[str, Any]]]:
    if not table_exists(conn, "campaign_image_cue_scores"):
        return {}

    rows = fetch_rows(
        conn,
        """
        SELECT
            image_id,
            campaign_id,
            cue_id,
            model_name,
            score_version,
            score,
            score_status
        FROM campaign_image_cue_scores
        ORDER BY image_id, campaign_id, cue_id
        """
    )

    out: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for r in rows:
        out[str(r["image_id"])].append(
            {
                "campaign_id": r.get("campaign_id"),
                "cue_id": r.get("cue_id"),
                "score": r.get("score"),
                "model_name": r.get("model_name"),
                "score_version": r.get("score_version"),
                "score_status": r.get("score_status"),
            }
        )

    return dict(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--out", default="data/search/ontology_effective_search_index_v1.jsonl")
    ap.add_argument("--summary-out", default="audit/ontology/ontology_effective_search_index_v1.summary.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise RuntimeError(f"foreign_key_check failed: {[tuple(r) for r in fk[:10]]}")

    images = fetch_rows(
        conn,
        """
        SELECT
            image_id,
            path,
            resolved_path,
            category,
            source_group,
            place_name,
            subject_name,
            metadata_status
        FROM images
        ORDER BY image_id
        """
    )

    tag_map, tag_assertions = load_tags(conn)
    plant_bloom = load_plant_bloom(conn)
    effective_campaigns = load_effective_campaigns(conn)
    excluded_campaigns = load_excluded_campaigns(conn)
    cue_scores = load_cue_scores(conn)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    campaign_effective_counts = Counter()
    campaign_excluded_counts = Counter()
    tag_axis_counts = Counter()
    plant_subject_count = 0

    with out_path.open("w", encoding="utf-8") as f:
        for img in images:
            image_id = str(img["image_id"])
            subject_name = img.get("subject_name")

            axes = tag_map.get(image_id, {})
            plant_info = plant_bloom.get(str(subject_name)) if subject_name else None

            if plant_info:
                plant_subject_count += 1

            eff = effective_campaigns.get(image_id, [])
            excl = excluded_campaigns.get(image_id, [])

            for cid in eff:
                campaign_effective_counts[cid] += 1

            for e in excl:
                campaign_excluded_counts[str(e.get("campaign_id"))] += 1

            for axis, tags in axes.items():
                if tags:
                    tag_axis_counts[axis] += 1

            row = {
                "index_version": INDEX_VERSION,
                "score_status": SCORE_STATUS,
                "image": {
                    "image_id": image_id,
                    "path": img.get("path"),
                    "resolved_path": img.get("resolved_path"),
                    "category": img.get("category"),
                    "source_group": img.get("source_group"),
                    "place_name": img.get("place_name"),
                    "subject_name": subject_name,
                    "metadata_status": img.get("metadata_status"),
                },
                "ontology_tags": axes,
                "ontology_tag_assertions": tag_assertions.get(image_id, []),
                "plant_bloom_prior": plant_info,
                "effective_campaigns": eff,
                "excluded_campaigns": excl,
                "campaign_visual_cue_scores": cue_scores.get(image_id, []),
                "retrieval_policy": {
                    "uses_effective_retrieval_candidates": True,
                    "flower_season_exclusion_applied": bool(excl),
                    "flower_season_exclusion_is_campaign_specific": True,
                    "interpretation": "이 row는 검색/RAG용 진단 index이며, calibrated 품질 판정이 아니다.",
                },
            }

            f.write(jdump(row) + "\n")
            counts["images"] += 1

    summary = {
        "event": "done",
        "index_version": INDEX_VERSION,
        "db": args.db,
        "out": args.out,
        "images_exported": counts["images"],
        "images_with_ontology_tags": len(tag_map),
        "images_with_plant_bloom_prior": plant_subject_count,
        "images_with_effective_campaigns": len(effective_campaigns),
        "images_with_excluded_campaigns": len(excluded_campaigns),
        "effective_campaign_counts": dict(sorted(campaign_effective_counts.items())),
        "excluded_campaign_counts": dict(sorted(campaign_excluded_counts.items())),
        "tag_axis_image_counts": dict(sorted(tag_axis_counts.items())),
        "score_status": SCORE_STATUS,
        "interpretation": "DB source of truth에서 ontology tag, plant bloom prior, effective retrieval eligibility를 합쳐 검색/RAG용 JSONL index를 export했다.",
    }

    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
