from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import yaml


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"yaml root must be object: {path}")
    return data


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS visual_cues (
            cue_id TEXT PRIMARY KEY,
            policy_id TEXT NOT NULL,
            cue_group TEXT NOT NULL,
            cue_type TEXT NOT NULL,
            prompts_json TEXT NOT NULL,
            ontology_write_allowed_if_human_verified INTEGER NOT NULL,
            verified_ontology_tags_json TEXT NOT NULL,
            score_status TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS campaign_visual_cue_requirements (
            campaign_id TEXT NOT NULL,
            cue_id TEXT NOT NULL,
            policy_id TEXT NOT NULL,
            requirement_role TEXT NOT NULL,
            source_field TEXT NOT NULL,
            source_value TEXT NOT NULL,
            score_status TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY(campaign_id, cue_id, policy_id),
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY(cue_id) REFERENCES visual_cues(cue_id)
        );

        CREATE TABLE IF NOT EXISTS campaign_image_cue_scores (
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            cue_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            score_version TEXT NOT NULL,
            score REAL NOT NULL,
            score_status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY(campaign_id, image_id, cue_id, model_name, score_version),
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(cue_id) REFERENCES visual_cues(cue_id)
        );
        """
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--policy", default="configs/seasonal_prompt_cue_policy_v1.yaml")
    args = ap.parse_args()

    conn = connect(Path(args.db))
    ensure_tables(conn)

    policy = read_yaml(Path(args.policy))
    policy_id = policy["policy_id"]
    score_status = policy.get("score_status", "diagnostic_only")

    conn.execute(
        "DELETE FROM campaign_visual_cue_requirements WHERE policy_id = ?",
        (policy_id,),
    )
    conn.execute(
        "DELETE FROM visual_cues WHERE policy_id = ?",
        (policy_id,),
    )

    cue_count = 0
    for cue_id, cue in policy["visual_cues"].items():
        conn.execute(
            """
            INSERT OR REPLACE INTO visual_cues
            (cue_id, policy_id, cue_group, cue_type, prompts_json,
             ontology_write_allowed_if_human_verified, verified_ontology_tags_json,
             score_status, raw_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cue_id,
                policy_id,
                cue["cue_group"],
                cue["cue_type"],
                jdump(cue.get("prompts", [])),
                1 if cue.get("ontology_write_allowed_if_human_verified") else 0,
                jdump(cue.get("verified_ontology_tags", {})),
                score_status,
                jdump(cue),
            ),
        )
        cue_count += 1

    campaigns = conn.execute(
        """
        SELECT campaign_id, season
        FROM campaigns
        ORDER BY campaign_id
        """
    ).fetchall()

    req_count = 0
    unmapped_campaigns = []

    for c in campaigns:
        campaign_id = str(c["campaign_id"])
        season = str(c["season"] or "").strip()

        mapping = policy.get("campaign_season_mapping", {}).get(season)
        if not mapping:
            unmapped_campaigns.append({"campaign_id": campaign_id, "season": season})
            continue

        for cue_id in mapping.get("preferred_cues", []):
            if cue_id not in policy["visual_cues"]:
                raise RuntimeError(f"unknown cue_id in mapping: {cue_id}")

            raw = {
                "campaign_id": campaign_id,
                "season": season,
                "cue_id": cue_id,
                "policy_id": policy_id,
                "requirement_role": "preferred",
            }

            conn.execute(
                """
                INSERT OR REPLACE INTO campaign_visual_cue_requirements
                (campaign_id, cue_id, policy_id, requirement_role, source_field,
                 source_value, score_status, raw_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    campaign_id,
                    cue_id,
                    policy_id,
                    "preferred",
                    "campaigns.season",
                    season,
                    score_status,
                    jdump(raw),
                ),
            )
            req_count += 1

    conn.commit()

    by_campaign = {
        str(r["campaign_id"]): int(r["n"])
        for r in conn.execute(
            """
            SELECT campaign_id, COUNT(*) AS n
            FROM campaign_visual_cue_requirements
            WHERE policy_id = ?
            GROUP BY campaign_id
            ORDER BY campaign_id
            """,
            (policy_id,),
        )
    }

    print(json.dumps({
        "event": "done",
        "db": args.db,
        "policy": args.policy,
        "policy_id": policy_id,
        "visual_cues": cue_count,
        "campaign_visual_cue_requirements": req_count,
        "requirements_by_campaign": by_campaign,
        "unmapped_campaigns": unmapped_campaigns,
        "score_status": score_status,
        "threshold_status": policy.get("threshold_status"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
