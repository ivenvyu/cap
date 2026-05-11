from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SCORE_VERSION = "botanical_bloom_season_prior_v1"
SCORE_STATUS = "diagnostic_only"

SEASON_VALUES = {"spring", "summer", "autumn", "winter"}

KO_TO_EN_SEASON = {
    "봄": "spring",
    "여름": "summer",
    "가을": "autumn",
    "겨울": "winter",
}


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS campaign_image_botanical_bloom_priors (
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            subject_name TEXT,
            plant_id TEXT,
            campaign_season TEXT NOT NULL,
            bloom_values_json TEXT NOT NULL,
            bloom_seasons_json TEXT NOT NULL,
            match_status TEXT NOT NULL,
            evidence_status TEXT NOT NULL,
            score_status TEXT NOT NULL,
            score_version TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (campaign_id, image_id, score_version),
            FOREIGN KEY (campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY (image_id) REFERENCES images(image_id),
            FOREIGN KEY (plant_id) REFERENCES plant_entities(plant_id)
        );

        CREATE INDEX IF NOT EXISTS idx_campaign_image_botanical_priors_campaign
            ON campaign_image_botanical_bloom_priors(campaign_id);

        CREATE INDEX IF NOT EXISTS idx_campaign_image_botanical_priors_image
            ON campaign_image_botanical_bloom_priors(image_id);

        CREATE INDEX IF NOT EXISTS idx_campaign_image_botanical_priors_status
            ON campaign_image_botanical_bloom_priors(match_status);
        """
    )


def load_plant_lookup(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
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
    ).fetchall()

    lookup: dict[str, dict[str, Any]] = {}

    for r in rows:
        name = str(r["name"])
        plant_id = str(r["plant_id"])

        item = lookup.setdefault(
            name,
            {
                "plant_id": plant_id,
                "plant_type": str(r["plant_type"]),
                "bloom_values": [],
                "bloom_seasons": [],
                "confidence_statuses": [],
            },
        )

        if r["bloom_value"] is not None:
            bloom_value = str(r["bloom_value"])
            item["bloom_values"].append(bloom_value)

            if bloom_value in KO_TO_EN_SEASON:
                item["bloom_seasons"].append(KO_TO_EN_SEASON[bloom_value])

            if r["confidence_status"] is not None:
                item["confidence_statuses"].append(str(r["confidence_status"]))

    for item in lookup.values():
        item["bloom_values"] = sorted(set(item["bloom_values"]))
        item["bloom_seasons"] = sorted(set(item["bloom_seasons"]))
        item["confidence_statuses"] = sorted(set(item["confidence_statuses"]))

    return lookup


def classify_match(campaign_season: str, subject_name: str | None, plant: dict[str, Any] | None) -> str:
    if not subject_name:
        return "no_subject_name"

    if plant is None:
        return "subject_not_in_plant_knowledge"

    bloom_seasons = set(plant.get("bloom_seasons", []))
    if not bloom_seasons:
        return "known_plant_without_season_prior"

    if campaign_season in bloom_seasons:
        return "direct_season_match"

    return "known_season_mismatch"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--score-version", default=SCORE_VERSION)
    ap.add_argument("--summary-out", default="audit/ontology/botanical_bloom_season_priors_v1.summary.json")
    ap.add_argument("--campaign-like", default="phase%")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    ensure_table(conn)

    fk = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk:
        raise RuntimeError(f"foreign_key_check failed before scoring: {[tuple(r) for r in fk[:10]]}")

    plant_lookup = load_plant_lookup(conn)
    if not plant_lookup:
        raise RuntimeError("plant lookup is empty; run ingest_sayuwon_plant_bloom_priors_to_db.py first")

    campaigns = conn.execute(
        """
        SELECT campaign_id, season
        FROM campaigns
        WHERE campaign_id LIKE ?
          AND season IS NOT NULL
        ORDER BY campaign_id
        """,
        (args.campaign_like,),
    ).fetchall()

    campaigns = [dict(r) for r in campaigns if str(r["season"]) in SEASON_VALUES]
    if not campaigns:
        raise RuntimeError("no seasonal campaigns found")

    images = conn.execute(
        """
        SELECT
            image_id,
            path,
            category,
            source_group,
            place_name,
            subject_name
        FROM images
        ORDER BY image_id
        """
    ).fetchall()

    conn.execute(
        """
        DELETE FROM campaign_image_botanical_bloom_priors
        WHERE score_version = ?
        """,
        (args.score_version,),
    )

    inserted = 0
    status_counts: Counter[str] = Counter()
    status_by_campaign: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for campaign in campaigns:
        campaign_id = str(campaign["campaign_id"])
        campaign_season = str(campaign["season"])

        for image in images:
            subject_name = image["subject_name"]
            subject_name_str = str(subject_name) if subject_name is not None else None
            plant = plant_lookup.get(subject_name_str) if subject_name_str else None

            match_status = classify_match(
                campaign_season=campaign_season,
                subject_name=subject_name_str,
                plant=plant,
            )

            bloom_values = plant["bloom_values"] if plant else []
            bloom_seasons = plant["bloom_seasons"] if plant else []
            plant_id = plant["plant_id"] if plant else None

            if match_status == "direct_season_match":
                evidence_status = "domain_prior_supports_campaign_season"
            elif match_status == "known_season_mismatch":
                evidence_status = "domain_prior_conflicts_with_campaign_season"
            elif match_status in {"no_subject_name", "subject_not_in_plant_knowledge"}:
                evidence_status = "domain_prior_unavailable"
            else:
                evidence_status = "domain_prior_incomplete"

            raw = {
                "campaign_id": campaign_id,
                "campaign_season": campaign_season,
                "image_id": image["image_id"],
                "path": image["path"],
                "subject_name": subject_name_str,
                "plant_id": plant_id,
                "bloom_values": bloom_values,
                "bloom_seasons": bloom_seasons,
                "match_status": match_status,
                "evidence_status": evidence_status,
                "score_status": SCORE_STATUS,
                "score_version": args.score_version,
                "interpretation": (
                    "botanical domain prior for campaign-image retrieval; "
                    "not a temporal_axis image label"
                ),
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO campaign_image_botanical_bloom_priors
                (campaign_id, image_id, subject_name, plant_id, campaign_season,
                 bloom_values_json, bloom_seasons_json, match_status, evidence_status,
                 score_status, score_version, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    image["image_id"],
                    subject_name_str,
                    plant_id,
                    campaign_season,
                    jdump(bloom_values),
                    jdump(bloom_seasons),
                    match_status,
                    evidence_status,
                    SCORE_STATUS,
                    args.score_version,
                    jdump(raw),
                ),
            )
            inserted += 1
            status_counts[match_status] += 1
            status_by_campaign[campaign_id][match_status] += 1

            if len(examples[match_status]) < 10:
                examples[match_status].append(
                    {
                        "campaign_id": campaign_id,
                        "campaign_season": campaign_season,
                        "image_id": image["image_id"],
                        "path": image["path"],
                        "subject_name": subject_name_str,
                        "plant_id": plant_id,
                        "bloom_values": bloom_values,
                        "bloom_seasons": bloom_seasons,
                    }
                )

    conn.commit()

    fk_after = conn.execute("PRAGMA foreign_key_check").fetchall()
    if fk_after:
        raise RuntimeError(f"foreign_key_check failed after scoring: {[tuple(r) for r in fk_after[:10]]}")

    summary = {
        "event": "done",
        "db": args.db,
        "score_version": args.score_version,
        "score_status": SCORE_STATUS,
        "interpretation": "botanical domain prior for retrieval only; not temporal_axis label",
        "campaigns": len(campaigns),
        "images": len(images),
        "rows_inserted": inserted,
        "match_status_counts": dict(sorted(status_counts.items())),
        "match_status_by_campaign": {
            cid: dict(sorted(counter.items()))
            for cid, counter in sorted(status_by_campaign.items())
        },
        "examples": dict(examples),
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jdump(summary) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
