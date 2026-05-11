from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any


MONTH_VALUES = {
    "1월", "2월", "3월", "4월", "5월", "6월",
    "7월", "8월", "9월", "10월", "11월", "12월",
}

SEASON_VALUES = {"봄", "여름", "가을", "겨울"}


def connect(path: Path) -> sqlite3.Connection:
    if not path.exists():
        raise RuntimeError(f"missing DB: {path}")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS plant_entities (
            plant_id TEXT PRIMARY KEY,
            plant_type TEXT NOT NULL,
            scientific_name_json TEXT NOT NULL,
            flower_language_json TEXT NOT NULL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS plant_names (
            plant_id TEXT NOT NULL,
            name TEXT NOT NULL,
            lang TEXT NOT NULL,
            PRIMARY KEY (plant_id, name, lang),
            FOREIGN KEY (plant_id) REFERENCES plant_entities(plant_id)
        );

        CREATE TABLE IF NOT EXISTS plant_bloom_priors (
            plant_id TEXT NOT NULL,
            bloom_value TEXT NOT NULL,
            bloom_type TEXT NOT NULL,
            source_id TEXT NOT NULL,
            confidence_status TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (plant_id, bloom_value, bloom_type, source_id),
            FOREIGN KEY (plant_id) REFERENCES plant_entities(plant_id)
        );

        CREATE INDEX IF NOT EXISTS idx_plant_names_name
            ON plant_names(name);

        CREATE INDEX IF NOT EXISTS idx_plant_bloom_priors_value
            ON plant_bloom_priors(bloom_value, bloom_type);
        """
    )


def bloom_type(value: str) -> str:
    if value in MONTH_VALUES:
        return "month"
    if value in SEASON_VALUES:
        return "season"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--knowledge", default="configs/domain_knowledge/sayuwon_entity_knowledge_v1.json")
    ap.add_argument("--source-id", default="sayuwon_entity_knowledge_v1")
    args = ap.parse_args()

    knowledge_path = Path(args.knowledge)
    if not knowledge_path.exists():
        raise RuntimeError(f"missing knowledge JSON: {knowledge_path}")

    data = json.loads(knowledge_path.read_text(encoding="utf-8"))
    plants = data.get("plant_entities", [])
    if not plants:
        raise RuntimeError("no plant_entities found")

    conn = connect(Path(args.db))
    ensure_tables(conn)

    # This source file is the source of truth for plant bloom priors.
    conn.execute("DELETE FROM plant_bloom_priors WHERE source_id = ?", (args.source_id,))
    conn.execute("DELETE FROM plant_names")
    conn.execute("DELETE FROM plant_entities")

    plant_count = 0
    name_count = 0
    bloom_count = 0

    for plant in plants:
        plant_id = plant["plant_id"]
        plant_types = plant.get("plant_type", [])
        plant_type = ",".join(plant_types) if isinstance(plant_types, list) else str(plant_types)

        conn.execute(
            """
            INSERT OR REPLACE INTO plant_entities
            (plant_id, plant_type, scientific_name_json, flower_language_json, raw_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                plant_id,
                plant_type,
                jdump(plant.get("학명", [])),
                jdump(plant.get("꽃말", [])),
                jdump(plant),
            ),
        )
        plant_count += 1

        for name in plant.get("name_ko", []):
            conn.execute(
                """
                INSERT OR REPLACE INTO plant_names
                (plant_id, name, lang)
                VALUES (?, ?, ?)
                """,
                (plant_id, name, "ko"),
            )
            name_count += 1

        for bloom in plant.get("개화시기", []):
            value = str(bloom)
            bt = bloom_type(value)
            conn.execute(
                """
                INSERT OR REPLACE INTO plant_bloom_priors
                (plant_id, bloom_value, bloom_type, source_id, confidence_status, raw_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    plant_id,
                    value,
                    bt,
                    args.source_id,
                    "domain_knowledge_prior",
                    jdump({
                        "plant_id": plant_id,
                        "bloom_value": value,
                        "bloom_type": bt,
                    }),
                ),
            )
            bloom_count += 1

    conn.commit()

    print(json.dumps({
        "event": "done",
        "db": args.db,
        "knowledge": args.knowledge,
        "source_id": args.source_id,
        "plant_entities": plant_count,
        "plant_names": name_count,
        "plant_bloom_priors": bloom_count,
        "confidence_status": "domain_knowledge_prior",
        "db_role": "operational_source_of_truth"
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
