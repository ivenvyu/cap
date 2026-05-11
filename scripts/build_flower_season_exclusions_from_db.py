from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SOURCE_ID = "flower_season_exclusion_v1"
SCORE_STATUS = "diagnostic_only"

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
        CREATE TABLE IF NOT EXISTS campaign_image_flower_season_exclusions (
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            requested_season TEXT NOT NULL,
            subject_name TEXT NOT NULL,
            plant_id TEXT NOT NULL,
            bloom_seasons_json TEXT NOT NULL,
            exclusion_reason TEXT NOT NULL,
            source_id TEXT NOT NULL,
            score_status TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (campaign_id, image_id, source_id),
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(plant_id) REFERENCES plant_entities(plant_id)
        );

        CREATE INDEX IF NOT EXISTS idx_campaign_image_flower_exclusions_campaign
            ON campaign_image_flower_season_exclusions(campaign_id);

        CREATE INDEX IF NOT EXISTS idx_campaign_image_flower_exclusions_image
            ON campaign_image_flower_season_exclusions(image_id);
        """
    )


def load_flower_bloom_lookup(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT
            n.name,
            e.plant_id,
            e.plant_type,
            b.bloom_value,
            b.bloom_type
        FROM plant_names n
        JOIN plant_entities e
          ON n.plant_id = e.plant_id
        LEFT JOIN plant_bloom_priors b
          ON n.plant_id = b.plant_id
        ORDER BY n.name, b.bloom_value
        """
    ).fetchall()

    lookup: dict[str, dict[str, Any]] = {}

    for r in rows:
        plant_type = str(r["plant_type"] or "")
        if "flower" not in plant_type.split(","):
            continue

        name = str(r["name"])
        item = lookup.setdefault(
            name,
            {
                "plant_id": str(r["plant_id"]),
                "plant_type": plant_type,
                "bloom_values": [],
                "bloom_seasons": [],
            },
        )

        if r["bloom_value"] is not None:
            value = str(r["bloom_value"])
            item["bloom_values"].append(value)
            if value in KO_TO_EN_SEASON:
                item["bloom_seasons"].append(KO_TO_EN_SEASON[value])

    for item in lookup.values():
        item["bloom_values"] = sorted(set(item["bloom_values"]))
        item["bloom_seasons"] = sorted(set(item["bloom_seasons"]))

    return lookup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--source-id", default=SOURCE_ID)
    ap.add_argument("--summary-out", default="audit/ontology/flower_season_exclusions_v1.summary.json")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    ensure_table(conn)

    flower_lookup = load_flower_bloom_lookup(conn)
    if not flower_lookup:
        raise RuntimeError("no flower bloom lookup found; ingest plant bloom priors first")

    campaigns = conn.execute(
        """
        SELECT campaign_id, season
        FROM campaigns
        WHERE season IN ('spring', 'summer', 'autumn', 'winter')
        ORDER BY campaign_id
        """
    ).fetchall()

    images = conn.execute(
        """
        SELECT image_id, path, subject_name
        FROM images
        ORDER BY image_id
        """
    ).fetchall()

    conn.execute(
        "DELETE FROM campaign_image_flower_season_exclusions WHERE source_id = ?",
        (args.source_id,),
    )

    excluded = 0
    checked_flower_images = 0
    excluded_by_campaign: dict[str, Counter[str]] = defaultdict(Counter)
    examples = []

    for c in campaigns:
        campaign_id = str(c["campaign_id"])
        requested_season = str(c["season"])

        for img in images:
            subject_name = img["subject_name"]
            if not subject_name:
                continue

            subject_name = str(subject_name)
            flower = flower_lookup.get(subject_name)
            if flower is None:
                continue

            checked_flower_images += 1
            bloom_seasons = set(flower["bloom_seasons"])

            # 핵심 규칙:
            # 계절 프롬프트가 있고, 꽃 DB에 있는 꽃인데, 해당 계절에 피지 않으면 제외.
            if bloom_seasons and requested_season not in bloom_seasons:
                raw = {
                    "campaign_id": campaign_id,
                    "image_id": img["image_id"],
                    "path": img["path"],
                    "requested_season": requested_season,
                    "subject_name": subject_name,
                    "plant_id": flower["plant_id"],
                    "bloom_values": flower["bloom_values"],
                    "bloom_seasons": flower["bloom_seasons"],
                    "rule": "exclude if flower bloom seasons do not include requested campaign season",
                }

                conn.execute(
                    """
                    INSERT OR REPLACE INTO campaign_image_flower_season_exclusions
                    (campaign_id, image_id, requested_season, subject_name, plant_id,
                     bloom_seasons_json, exclusion_reason, source_id, score_status, raw_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        campaign_id,
                        img["image_id"],
                        requested_season,
                        subject_name,
                        flower["plant_id"],
                        jdump(flower["bloom_seasons"]),
                        "flower_not_in_requested_season",
                        args.source_id,
                        SCORE_STATUS,
                        jdump(raw),
                    ),
                )

                excluded += 1
                excluded_by_campaign[campaign_id][subject_name] += 1

                if len(examples) < 30:
                    examples.append(raw)

    conn.commit()

    result = {
        "event": "done",
        "db": args.db,
        "source_id": args.source_id,
        "campaigns": len(campaigns),
        "flower_subjects_known": len(flower_lookup),
        "checked_campaign_flower_pairs": checked_flower_images,
        "excluded_rows": excluded,
        "excluded_by_campaign_subject": {
            cid: dict(counter)
            for cid, counter in sorted(excluded_by_campaign.items())
        },
        "examples": examples,
        "score_status": SCORE_STATUS,
        "rule": "if campaign has season and image subject is a known flower whose bloom seasons do not include that season, exclude that campaign-image pair",
    }

    out = Path(args.summary_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(jdump(result) + "\n", encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
