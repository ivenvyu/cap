from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "ontology_db_v1"
DB_BUILD_ID = "db_build_phase1b_source_of_truth_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def jdump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def exec_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS db_builds (
            db_build_id TEXT PRIMARY KEY,
            schema_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            score_status TEXT NOT NULL,
            source_summary_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS artifact_sources (
            artifact_id TEXT PRIMARY KEY,
            artifact_path TEXT NOT NULL,
            artifact_kind TEXT NOT NULL,
            artifact_version TEXT,
            content_hash TEXT NOT NULL,
            row_count INTEGER,
            imported_at TEXT NOT NULL,
            db_build_id TEXT NOT NULL,
            FOREIGN KEY(db_build_id) REFERENCES db_builds(db_build_id)
        );

        CREATE TABLE IF NOT EXISTS import_batches (
            import_batch_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            table_name TEXT NOT NULL,
            imported_rows INTEGER NOT NULL,
            imported_at TEXT NOT NULL,
            import_script TEXT NOT NULL,
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS images (
            image_id TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            resolved_path TEXT,
            filename TEXT,
            category TEXT,
            source_group TEXT,
            place_name TEXT,
            subject_name TEXT,
            extension TEXT,
            season_tags_json TEXT,
            visual_tags_json TEXT,
            mood_tags_json TEXT,
            metadata_status TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS campaigns (
            campaign_id TEXT PRIMARY KEY,
            campaign_version TEXT,
            campaign_family TEXT,
            purpose_type TEXT,
            space_type TEXT,
            season TEXT,
            target_channel TEXT,
            campaign_text TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS image_embeddings (
            image_id TEXT NOT NULL,
            model_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            embedding_version TEXT NOT NULL,
            embedding_dim INTEGER NOT NULL,
            embedding_row INTEGER NOT NULL,
            npy_path TEXT NOT NULL,
            index_path TEXT NOT NULL,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            PRIMARY KEY(image_id, model_type),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS image_duplicates (
            image_id TEXT PRIMARY KEY,
            duplicate_group_id TEXT NOT NULL,
            duplicate_group_version TEXT,
            exact_file_sha256 TEXT,
            exact_duplicate_status TEXT,
            exact_duplicate_group_size INTEGER,
            nearest_dinov2_neighbor_id TEXT,
            nearest_dinov2_cosine_sim REAL,
            nearest_dinov2_status TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS image_clusters (
            image_id TEXT PRIMARY KEY,
            cluster_version TEXT,
            cluster_status TEXT,
            dinov2_cluster_id_coarse TEXT,
            dinov2_cluster_id_mid TEXT,
            dinov2_cluster_id_fine TEXT,
            cluster_size_coarse INTEGER,
            cluster_size_mid INTEGER,
            cluster_size_fine INTEGER,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS image_regions (
            image_id TEXT NOT NULL,
            region_name TEXT NOT NULL,
            box_json TEXT,
            edge_density REAL,
            contrast_std REAL,
            brightness_mean REAL,
            brightness_std REAL,
            saturation_mean REAL,
            safe_score REAL,
            score_status TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            PRIMARY KEY(image_id, region_name),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS retrieval_candidates (
            retrieval_candidate_id TEXT PRIMARY KEY,
            retrieval_batch_id TEXT,
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            rank INTEGER,
            clip_margin REAL,
            clip_positive_max_sim REAL,
            clip_negative_max_sim REAL,
            score_status TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS pair_features (
            feature_snapshot_id TEXT PRIMARY KEY,
            pair_id TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            layout_spec_id TEXT,
            duplicate_group_id TEXT,
            batch_id TEXT,
            feature_status TEXT,
            snapshot_version TEXT,
            features_json TEXT NOT NULL,
            clip_margin REAL,
            clip_positive_max_sim REAL,
            clip_negative_max_sim REAL,
            clip_rank_percentile REAL,
            required_region_safe_min REAL,
            required_region_safe_mean REAL,
            edge_density REAL,
            brightness REAL,
            created_at TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS review_events (
            review_event_id TEXT PRIMARY KEY,
            timestamp TEXT,
            annotator_id TEXT,
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            pair_id TEXT NOT NULL,
            feature_snapshot_id TEXT,
            duplicate_group_id TEXT,
            decision_label TEXT NOT NULL,
            decision_numeric INTEGER,
            issue_tags_json TEXT,
            notes TEXT,
            queue_stage TEXT,
            source_bucket TEXT,
            layout_spec_id TEXT,
            preview_renderer_version TEXT,
            model_scores_json TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(feature_snapshot_id) REFERENCES pair_features(feature_snapshot_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS training_snapshots (
            training_snapshot_id TEXT PRIMARY KEY,
            snapshot_kind TEXT NOT NULL,
            snapshot_version TEXT,
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            pair_id TEXT NOT NULL,
            feature_snapshot_id TEXT,
            layout_spec_id TEXT,
            label INTEGER NOT NULL,
            decision_label TEXT,
            group_id TEXT,
            label_status TEXT,
            issue_tags_json TEXT,
            raw_json TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            FOREIGN KEY(campaign_id) REFERENCES campaigns(campaign_id),
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(feature_snapshot_id) REFERENCES pair_features(feature_snapshot_id),
            FOREIGN KEY(artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS training_sets (
            training_set_id TEXT PRIMARY KEY,
            snapshot_kind TEXT NOT NULL,
            phase TEXT NOT NULL,
            policy_id TEXT,
            description TEXT,
            created_at TEXT NOT NULL,
            source_artifact_id TEXT,
            row_count INTEGER NOT NULL,
            score_status TEXT NOT NULL,
            FOREIGN KEY(source_artifact_id) REFERENCES artifact_sources(artifact_id)
        );

        CREATE TABLE IF NOT EXISTS training_set_items (
            training_set_id TEXT NOT NULL,
            training_snapshot_id TEXT NOT NULL,
            item_order INTEGER NOT NULL,
            PRIMARY KEY(training_set_id, training_snapshot_id),
            FOREIGN KEY(training_set_id) REFERENCES training_sets(training_set_id),
            FOREIGN KEY(training_snapshot_id) REFERENCES training_snapshots(training_snapshot_id)
        );

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

        CREATE TABLE IF NOT EXISTS retrieval_candidate_filter_decisions (
            filter_run_id TEXT NOT NULL,
            candidate_key TEXT NOT NULL,
            source_table TEXT NOT NULL,
            campaign_id TEXT NOT NULL,
            image_id TEXT NOT NULL,
            image_path TEXT,
            subject_name TEXT,
            retained INTEGER NOT NULL,
            exclusion_reason TEXT,
            exclusion_source_id TEXT,
            score_status TEXT NOT NULL,
            raw_candidate_json TEXT NOT NULL,
            raw_exclusion_json TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY(filter_run_id, candidate_key)
        );

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

        CREATE TABLE IF NOT EXISTS tag_axes (
            axis_id TEXT PRIMARY KEY,
            axis_name TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS tag_values (
            tag_id TEXT PRIMARY KEY,
            axis_id TEXT NOT NULL,
            tag_name TEXT NOT NULL,
            description TEXT,
            status TEXT NOT NULL,
            FOREIGN KEY(axis_id) REFERENCES tag_axes(axis_id)
        );

        CREATE TABLE IF NOT EXISTS image_tag_assertions (
            assertion_id TEXT PRIMARY KEY,
            image_id TEXT NOT NULL,
            tag_id TEXT NOT NULL,
            label_source TEXT NOT NULL,
            confidence_status TEXT NOT NULL,
            derived_from TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(image_id) REFERENCES images(image_id),
            FOREIGN KEY(tag_id) REFERENCES tag_values(tag_id)
        );

        CREATE TABLE IF NOT EXISTS cluster_label_queue (
            queue_id TEXT PRIMARY KEY,
            cluster_version TEXT NOT NULL,
            cluster_level TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            cluster_size INTEGER NOT NULL,
            representative_image_ids_json TEXT NOT NULL,
            representative_image_paths_json TEXT NOT NULL,
            representative_method TEXT NOT NULL,
            queue_status TEXT NOT NULL,
            score_status TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS cluster_tag_assertions (
            assertion_id TEXT PRIMARY KEY,
            cluster_version TEXT NOT NULL,
            cluster_level TEXT NOT NULL,
            cluster_id TEXT NOT NULL,
            tag_id TEXT NOT NULL,
            label_source TEXT NOT NULL,
            confidence_status TEXT NOT NULL,
            annotator_id TEXT,
            notes TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(tag_id) REFERENCES tag_values(tag_id)
        );

        CREATE INDEX IF NOT EXISTS idx_images_category ON images(category);
        CREATE INDEX IF NOT EXISTS idx_images_source_group ON images(source_group);
        CREATE INDEX IF NOT EXISTS idx_pair_features_campaign ON pair_features(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_review_events_campaign ON review_events(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_training_snapshots_kind ON training_snapshots(snapshot_kind);
        CREATE INDEX IF NOT EXISTS idx_training_sets_kind ON training_sets(snapshot_kind);
        CREATE INDEX IF NOT EXISTS idx_training_set_items_snapshot ON training_set_items(training_snapshot_id);
        CREATE INDEX IF NOT EXISTS idx_retrieval_campaign ON retrieval_candidates(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_cluster_label_queue_level ON cluster_label_queue(cluster_level, cluster_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_visual_cue_requirements_campaign ON campaign_visual_cue_requirements(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_image_cue_scores_campaign ON campaign_image_cue_scores(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_image_cue_scores_image ON campaign_image_cue_scores(image_id);
        CREATE INDEX IF NOT EXISTS idx_plant_names_name ON plant_names(name);
        CREATE INDEX IF NOT EXISTS idx_plant_bloom_priors_value ON plant_bloom_priors(bloom_value, bloom_type);
        CREATE INDEX IF NOT EXISTS idx_campaign_image_botanical_priors_campaign ON campaign_image_botanical_bloom_priors(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_image_botanical_priors_image ON campaign_image_botanical_bloom_priors(image_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_image_botanical_priors_status ON campaign_image_botanical_bloom_priors(match_status);
        CREATE INDEX IF NOT EXISTS idx_campaign_image_flower_exclusions_campaign ON campaign_image_flower_season_exclusions(campaign_id);
        CREATE INDEX IF NOT EXISTS idx_campaign_image_flower_exclusions_image ON campaign_image_flower_season_exclusions(image_id);
        CREATE INDEX IF NOT EXISTS idx_retrieval_candidate_filter_decisions_campaign
            ON retrieval_candidate_filter_decisions(filter_run_id, campaign_id);
        CREATE INDEX IF NOT EXISTS idx_retrieval_candidate_filter_decisions_image
            ON retrieval_candidate_filter_decisions(filter_run_id, image_id);
        CREATE INDEX IF NOT EXISTS idx_retrieval_candidate_filter_decisions_retained
            ON retrieval_candidate_filter_decisions(filter_run_id, retained);
        """
    )


def reset_tables(conn: sqlite3.Connection) -> None:
    tables = [
        "image_tag_assertions",
        "cluster_tag_assertions",
        "cluster_label_queue",
        "retrieval_candidate_filter_decisions",
        "campaign_image_flower_season_exclusions",
        "campaign_image_botanical_bloom_priors",
        "plant_bloom_priors",
        "plant_names",
        "plant_entities",
        "campaign_image_cue_scores",
        "campaign_visual_cue_requirements",
        "visual_cues",
        "tag_values",
        "tag_axes",
        "training_set_items",
        "training_sets",
        "training_snapshots",
        "review_events",
        "pair_features",
        "retrieval_candidates",
        "image_regions",
        "image_clusters",
        "image_duplicates",
        "image_embeddings",
        "campaigns",
        "images",
        "import_batches",
        "artifact_sources",
        "db_builds",
    ]
    for t in tables:
        conn.execute(f"DELETE FROM {t}")


def register_artifact(
    conn: sqlite3.Connection,
    path: Path,
    artifact_kind: str,
    artifact_version: str | None,
    row_count: int,
) -> str:
    artifact_id = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:16]
    conn.execute(
        """
        INSERT OR REPLACE INTO artifact_sources
        (artifact_id, artifact_path, artifact_kind, artifact_version, content_hash, row_count, imported_at, db_build_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            artifact_id,
            str(path),
            artifact_kind,
            artifact_version,
            sha256_file(path),
            row_count,
            utc_now(),
            DB_BUILD_ID,
        ),
    )
    return artifact_id


def register_batch(
    conn: sqlite3.Connection,
    artifact_id: str,
    table_name: str,
    imported_rows: int,
) -> None:
    batch_id = hashlib.sha1(f"{artifact_id}:{table_name}".encode("utf-8")).hexdigest()[:16]
    conn.execute(
        """
        INSERT OR REPLACE INTO import_batches
        (import_batch_id, artifact_id, table_name, imported_rows, imported_at, import_script)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (batch_id, artifact_id, table_name, imported_rows, utc_now(), "scripts/build_ontology_db.py"),
    )


def import_images(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "raw_image_manifest", "raw_image_manifest_v2_2_1", len(rows))

    for r in rows:
        image_path = r.get("path", "")
        filename = Path(image_path).name if image_path else r.get("filename")
        conn.execute(
            """
            INSERT OR REPLACE INTO images
            (image_id, path, resolved_path, filename, category, source_group, place_name, subject_name,
             extension, season_tags_json, visual_tags_json, mood_tags_json, metadata_status, raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["image_id"],
                image_path,
                r.get("resolved_path"),
                filename,
                r.get("category"),
                r.get("source_group"),
                r.get("place_name"),
                r.get("subject_name"),
                r.get("extension") or Path(image_path).suffix.lower().lstrip("."),
                jdump(r.get("season_tags", [])),
                jdump(r.get("visual_tags", [])),
                jdump(r.get("mood_tags", [])),
                r.get("metadata_status", "machine_suggested"),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "images", len(rows))
    return len(rows)


def import_embedding_index(
    conn: sqlite3.Connection,
    path: Path,
    npy_path: Path,
    model_type: str,
) -> int:
    rows = read_csv_rows(path)
    artifact_id = register_artifact(conn, path, f"{model_type}_embedding_index", None, len(rows))

    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO image_embeddings
            (image_id, model_type, model_name, embedding_version, embedding_dim, embedding_row,
             npy_path, index_path, raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["image_id"],
                model_type,
                r.get("model_name", ""),
                r.get("embedding_version", ""),
                int(float(r.get("embedding_dim", 0))),
                int(float(r.get("embedding_row", 0))),
                str(npy_path),
                str(path),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "image_embeddings", len(rows))
    return len(rows)


def import_duplicates(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "duplicate_groups", "duplicate_groups_v1", len(rows))

    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO image_duplicates
            (image_id, duplicate_group_id, duplicate_group_version, exact_file_sha256,
             exact_duplicate_status, exact_duplicate_group_size,
             nearest_dinov2_neighbor_id, nearest_dinov2_cosine_sim, nearest_dinov2_status,
             raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["image_id"],
                r.get("duplicate_group_id"),
                r.get("duplicate_group_version") or r.get("version"),
                r.get("exact_file_sha256"),
                r.get("exact_duplicate_status"),
                r.get("exact_duplicate_group_size"),
                r.get("nearest_dinov2_neighbor_id"),
                r.get("nearest_dinov2_cosine_sim"),
                r.get("nearest_dinov2_status", "diagnostic_only"),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "image_duplicates", len(rows))
    return len(rows)


def import_clusters(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "dinov2_clusters", "dinov2_clusters_v1", len(rows))

    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO image_clusters
            (image_id, cluster_version, cluster_status, dinov2_cluster_id_coarse,
             dinov2_cluster_id_mid, dinov2_cluster_id_fine,
             cluster_size_coarse, cluster_size_mid, cluster_size_fine,
             raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["image_id"],
                r.get("cluster_version") or r.get("version"),
                r.get("cluster_status", "diagnostic_only"),
                r.get("dinov2_cluster_id_coarse") or r.get("cluster_id_coarse"),
                r.get("dinov2_cluster_id_mid") or r.get("cluster_id_mid"),
                r.get("dinov2_cluster_id_fine") or r.get("cluster_id_fine"),
                r.get("cluster_size_coarse"),
                r.get("cluster_size_mid"),
                r.get("cluster_size_fine"),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "image_clusters", len(rows))
    return len(rows)


def iter_region_rows(row: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    regions = row.get("regions")
    if isinstance(regions, dict):
        for name, payload in regions.items():
            if isinstance(payload, dict):
                yield name, payload
        return

    region_safety = row.get("region_safety")
    if isinstance(region_safety, dict):
        for name, payload in region_safety.items():
            if isinstance(payload, dict):
                yield name, payload
        return

    if "region_name" in row:
        yield str(row["region_name"]), row


def import_regions(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "region_safety_maps", "region_safety_maps_v1", len(rows))

    imported = 0
    for r in rows:
        image_id = r["image_id"]
        for region_name, region in iter_region_rows(r):
            conn.execute(
                """
                INSERT OR REPLACE INTO image_regions
                (image_id, region_name, box_json, edge_density, contrast_std, brightness_mean, brightness_std,
                 saturation_mean, safe_score, score_status, raw_json, artifact_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    image_id,
                    region_name,
                    jdump(region.get("box") or region.get("bbox")),
                    region.get("edge_density"),
                    region.get("contrast_std"),
                    region.get("brightness_mean"),
                    region.get("brightness_std"),
                    region.get("saturation_mean"),
                    region.get("safe_score") or region.get("safe_diag"),
                    r.get("score_status", "diagnostic_only"),
                    jdump(region),
                    artifact_id,
                ),
            )
            imported += 1

    register_batch(conn, artifact_id, "image_regions", imported)
    return imported


def import_campaign_file(conn: sqlite3.Connection, path: Path) -> int:
    r = read_json(path)
    artifact_id = register_artifact(conn, path, "campaign_payload", r.get("campaign_version"), 1)

    campaign_text = (
        r.get("campaign_text")
        or r.get("prompt")
        or r.get("brief")
        or r.get("description")
        or ""
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO campaigns
        (campaign_id, campaign_version, campaign_family, purpose_type, space_type,
         season, target_channel, campaign_text, raw_json, artifact_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            r["campaign_id"],
            r.get("campaign_version"),
            r.get("campaign_family"),
            r.get("purpose_type"),
            r.get("space_type"),
            r.get("season"),
            r.get("target_channel"),
            campaign_text,
            jdump(r),
            artifact_id,
        ),
    )
    register_batch(conn, artifact_id, "campaigns", 1)
    return 1


def import_retrieval_candidates(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "retrieval_candidates", None, len(rows))

    for i, r in enumerate(rows, start=1):
        campaign_id = r.get("campaign_id")
        image_id = r.get("image_id")
        candidate_id = (
            r.get("retrieval_candidate_id")
            or f"retr_{hashlib.sha1(f'{path}:{campaign_id}:{image_id}:{i}'.encode()).hexdigest()[:16]}"
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO retrieval_candidates
            (retrieval_candidate_id, retrieval_batch_id, campaign_id, image_id, rank,
             clip_margin, clip_positive_max_sim, clip_negative_max_sim, score_status, raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_id,
                r.get("retrieval_batch_id"),
                campaign_id,
                image_id,
                r.get("rank") or r.get("retrieval_rank") or i,
                r.get("clip_margin"),
                r.get("clip_positive_max_sim"),
                r.get("clip_negative_max_sim"),
                r.get("score_status", "diagnostic_only"),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "retrieval_candidates", len(rows))
    return len(rows)


def import_pair_features(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "pair_features", None, len(rows))

    for r in rows:
        features = r.get("features", {})
        conn.execute(
            """
            INSERT OR REPLACE INTO pair_features
            (feature_snapshot_id, pair_id, campaign_id, image_id, layout_spec_id, duplicate_group_id,
             batch_id, feature_status, snapshot_version, features_json,
             clip_margin, clip_positive_max_sim, clip_negative_max_sim, clip_rank_percentile,
             required_region_safe_min, required_region_safe_mean, edge_density, brightness,
             created_at, raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["feature_snapshot_id"],
                r["pair_id"],
                r["campaign_id"],
                r["image_id"],
                r.get("layout_spec_id"),
                r.get("duplicate_group_id"),
                r.get("batch_id"),
                r.get("feature_status", "diagnostic_only"),
                r.get("snapshot_version"),
                jdump(features),
                features.get("clip_margin"),
                features.get("clip_positive_max_sim"),
                features.get("clip_negative_max_sim"),
                features.get("clip_rank_percentile"),
                features.get("required_region_safe_min"),
                features.get("required_region_safe_mean"),
                features.get("edge_density"),
                features.get("brightness"),
                r.get("created_at"),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "pair_features", len(rows))
    return len(rows)


def import_review_events(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "review_events", None, len(rows))
    decision_numeric_map = {"reject": 0, "acceptable": 1, "accept": 2, "best": 3}

    for r in rows:
        decision = r.get("decision", {})
        review_context = r.get("review_context", {})
        model_scores = r.get("model_score_at_review", {})

        label = decision.get("label") or r.get("decision_label")
        numeric = decision_numeric_map.get(label)

        conn.execute(
            """
            INSERT OR REPLACE INTO review_events
            (review_event_id, timestamp, annotator_id, campaign_id, image_id, pair_id, feature_snapshot_id,
             duplicate_group_id, decision_label, decision_numeric, issue_tags_json, notes,
             queue_stage, source_bucket, layout_spec_id, preview_renderer_version, model_scores_json,
             raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["review_event_id"],
                r.get("timestamp"),
                r.get("annotator_id"),
                r["campaign_id"],
                r["image_id"],
                r["pair_id"],
                r.get("feature_snapshot_id"),
                r.get("duplicate_group_id"),
                label,
                numeric,
                jdump(decision.get("issue_tags", [])),
                decision.get("notes", ""),
                review_context.get("queue_stage"),
                review_context.get("source_bucket"),
                review_context.get("layout_spec_id"),
                review_context.get("preview_renderer_version"),
                jdump(model_scores),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "review_events", len(rows))
    return len(rows)


def import_training_snapshots(conn: sqlite3.Connection, path: Path) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(conn, path, "training_snapshots", None, len(rows))

    for r in rows:
        conn.execute(
            """
            INSERT OR REPLACE INTO training_snapshots
            (training_snapshot_id, snapshot_kind, snapshot_version, campaign_id, image_id, pair_id,
             feature_snapshot_id, layout_spec_id, label, decision_label, group_id, label_status,
             issue_tags_json, raw_json, artifact_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                r["training_snapshot_id"],
                r["snapshot_kind"],
                r.get("snapshot_version"),
                r["campaign_id"],
                r["image_id"],
                r["pair_id"],
                r.get("feature_snapshot_id"),
                r.get("layout_spec_id"),
                int(r["label"]),
                r.get("decision_label"),
                r.get("group_id"),
                r.get("label_status"),
                jdump(r.get("issue_tags", [])),
                jdump(r),
                artifact_id,
            ),
        )

    register_batch(conn, artifact_id, "training_snapshots", len(rows))
    return len(rows)



def import_training_set_membership(
    conn: sqlite3.Connection,
    path: Path,
    training_set_id: str,
    snapshot_kind: str,
    phase: str,
    policy_id: str,
    description: str,
) -> int:
    rows = read_jsonl(path)
    artifact_id = register_artifact(
        conn,
        path,
        "training_set_membership",
        training_set_id,
        len(rows),
    )

    conn.execute(
        """
        INSERT OR REPLACE INTO training_sets
        (training_set_id, snapshot_kind, phase, policy_id, description, created_at,
         source_artifact_id, row_count, score_status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            training_set_id,
            snapshot_kind,
            phase,
            policy_id,
            description,
            utc_now(),
            artifact_id,
            len(rows),
            "diagnostic_only",
        ),
    )

    for i, r in enumerate(rows):
        training_snapshot_id = r["training_snapshot_id"]

        exists = conn.execute(
            "SELECT 1 FROM training_snapshots WHERE training_snapshot_id = ?",
            (training_snapshot_id,),
        ).fetchone()
        if exists is None:
            raise RuntimeError(
                f"training set {training_set_id} references missing training_snapshot_id: "
                f"{training_snapshot_id}"
            )

        conn.execute(
            """
            INSERT OR REPLACE INTO training_set_items
            (training_set_id, training_snapshot_id, item_order)
            VALUES (?, ?, ?)
            """,
            (training_set_id, training_snapshot_id, i),
        )

    register_batch(conn, artifact_id, "training_set_items", len(rows))
    return len(rows)


def import_tag_seed(conn: sqlite3.Connection) -> None:
    axes = {
        "space_axis": ["garden", "architecture", "indoor", "path", "water", "forest", "flowerbed", "courtyard"],
        "temporal_axis": ["spring", "summer", "autumn", "winter", "daytime", "evening", "season_unknown"],
        "weather_light_axis": ["sunny", "cloudy", "foggy", "rainy", "snowy", "backlit", "low_light", "unknown"],
        "subject_axis": ["flower", "tree", "plant_detail", "building", "sculpture", "path", "water_feature", "visitor", "empty_space"],
        "mood_axis": ["quiet", "active", "solemn", "bright", "mysterious", "calm", "dense", "open"],
        "usage_axis": ["sns", "brochure", "proposal", "poster", "web_banner", "archive_reference"],
        "design_affordance_axis": ["text_overlay_easy", "text_overlay_hard", "good_background_texture", "strong_subject_center", "needs_crop", "layout_unknown"],
    }

    for axis_id, tags in axes.items():
        conn.execute(
            "INSERT OR REPLACE INTO tag_axes(axis_id, axis_name, description, status) VALUES (?, ?, ?, ?)",
            (axis_id, axis_id, "seed ontology axis v1", "active"),
        )
        for tag in tags:
            tag_id = f"{axis_id}:{tag}"
            conn.execute(
                "INSERT OR REPLACE INTO tag_values(tag_id, axis_id, tag_name, description, status) VALUES (?, ?, ?, ?, ?)",
                (tag_id, axis_id, tag, "seed ontology tag v1", "active"),
            )


def glob_existing(patterns: list[str]) -> list[Path]:
    out: list[Path] = []
    for pattern in patterns:
        out.extend(sorted(Path(".").glob(pattern)))
    return [p for p in out if p.exists() and p.is_file() and p.stat().st_size > 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/db/cap_reranker_ontology.db")
    ap.add_argument("--reset", action="store_true", default=True)
    args = ap.parse_args()

    conn = connect(Path(args.db))
    exec_schema(conn)

    if args.reset:
        reset_tables(conn)

    conn.execute(
        """
        INSERT OR REPLACE INTO db_builds
        (db_build_id, schema_version, created_at, score_status, source_summary_json)
        VALUES (?, ?, ?, ?, ?)
        """,
        (DB_BUILD_ID, SCHEMA_VERSION, utc_now(), "diagnostic_only", "{}"),
    )

    counts: dict[str, int] = {}

    required = [
        Path("data/ontology/raw_image_manifest_v2_2_1.jsonl"),
        Path("data/embeddings/clip_image_index.csv"),
        Path("data/embeddings/dinov2_image_index.csv"),
        Path("data/ontology/duplicate_groups_v1.jsonl"),
        Path("data/ontology/dinov2_clusters_v1.jsonl"),
        Path("data/ontology/region_safety_maps_v1.jsonl"),
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise RuntimeError(f"missing required import artifacts: {missing}")

    counts["images"] = import_images(conn, required[0])
    counts["clip_embeddings"] = import_embedding_index(
        conn,
        required[1],
        Path("data/embeddings/clip_image_embeddings.npy"),
        "clip",
    )
    counts["dinov2_embeddings"] = import_embedding_index(
        conn,
        required[2],
        Path("data/embeddings/dinov2_image_embeddings.npy"),
        "dinov2",
    )
    counts["duplicates"] = import_duplicates(conn, required[3])
    counts["clusters"] = import_clusters(conn, required[4])
    counts["regions"] = import_regions(conn, required[5])

    campaign_files = glob_existing([
        "examples/campaigns/phase1a_summer_garden_walk.json",
        "examples/campaigns/phase1b/*.json",
    ])
    counts["campaigns"] = 0
    for p in campaign_files:
        counts["campaigns"] += import_campaign_file(conn, p)

    retrieval_files = glob_existing([
        "data/retrieval/clip_retrieval_candidates_v1.jsonl",
        "data/retrieval/phase1b/clip_retrieval_candidates__*.jsonl",
    ])
    counts["retrieval_candidates"] = 0
    for p in retrieval_files:
        counts["retrieval_candidates"] += import_retrieval_candidates(conn, p)

    pair_feature_files = glob_existing([
        "data/feature_snapshots/v2_2_1/phase1a_summer_garden_walk.jsonl",
        "data/feature_snapshots/v2_2_1/phase1b/pair_feature_snapshots__*.jsonl",
    ])
    counts["pair_features"] = 0
    for p in pair_feature_files:
        counts["pair_features"] += import_pair_features(conn, p)

    review_event_files = glob_existing([
        "data/review/review_events_phase1a_v1.jsonl",
        "data/review/phase1b/review_events_phase1b_v1.jsonl",
    ])
    counts["review_events"] = 0
    for p in review_event_files:
        counts["review_events"] += import_review_events(conn, p)

    training_files = glob_existing([
        "data/review/training_snapshot_phase1a_classifier_v1.jsonl",
        "data/review/training_snapshot_phase1a_ranker_v1.jsonl",
        "data/review/phase1b/training_snapshot_phase1b_classifier_v1.jsonl",
        "data/review/phase1b/training_snapshot_phase1b_ranker_v1.jsonl",
    ])
    counts["training_snapshots_rows_read"] = 0
    for p in training_files:
        counts["training_snapshots_rows_read"] += import_training_snapshots(conn, p)

    filtered_classifier = Path(
        "data/review/phase1b/filtered/training_snapshot_phase1b_classifier_v1__filtered.jsonl"
    )
    filtered_ranker = Path(
        "data/review/phase1b/filtered/training_snapshot_phase1b_ranker_v1__filtered.jsonl"
    )

    counts["training_set_items"] = 0
    if filtered_classifier.exists():
        counts["training_set_items"] += import_training_set_membership(
            conn=conn,
            path=filtered_classifier,
            training_set_id="phase1b_filtered_classifier_v1",
            snapshot_kind="classifier",
            phase="phase_1b",
            policy_id="phase1b_training_filter_v1",
            description="Phase 1b filtered classifier training set excluding indoor/winter coverage-gap campaign.",
        )

    if filtered_ranker.exists():
        counts["training_set_items"] += import_training_set_membership(
            conn=conn,
            path=filtered_ranker,
            training_set_id="phase1b_filtered_ranker_v1",
            snapshot_kind="ranker",
            phase="phase_1b",
            policy_id="phase1b_training_filter_v1",
            description="Phase 1b filtered ranker training set excluding indoor/winter coverage-gap campaign.",
        )

    counts["training_snapshots_unique"] = int(
        conn.execute("SELECT COUNT(*) FROM training_snapshots").fetchone()[0]
    )
    counts["training_sets"] = int(
        conn.execute("SELECT COUNT(*) FROM training_sets").fetchone()[0]
    )

    import_tag_seed(conn)

    conn.execute(
        "UPDATE db_builds SET source_summary_json = ? WHERE db_build_id = ?",
        (jdump(counts), DB_BUILD_ID),
    )

    conn.commit()

    print(json.dumps({
        "event": "done",
        "db": args.db,
        "schema_version": SCHEMA_VERSION,
        "db_build_id": DB_BUILD_ID,
        "counts": counts,
        "score_status": "diagnostic_only",
        "db_role": "operational_source_of_truth",
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
