from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def counter_from_tags(cells: pd.Series) -> Counter:
    c = Counter()
    for cell in cells.fillna(""):
        for tag in str(cell).split(","):
            tag = tag.strip()
            if tag:
                c[tag] += 1
    return c


def counter_to_plain(counter: Counter) -> dict[str, int]:
    return {str(k): int(v) for k, v in counter.items()}


def require_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing or empty file: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign-dir", default="examples/campaigns/phase1b")
    ap.add_argument("--retrieval-dir", default="data/retrieval/phase1b")
    ap.add_argument("--feature-dir", default="data/feature_snapshots/v2_2_1/phase1b")
    ap.add_argument("--review-dir", default="data/review/phase1b")
    ap.add_argument("--event-dir", default="data/review/phase1b/events")
    ap.add_argument("--training-dir", default="data/review/phase1b/training")
    ap.add_argument("--out", default="audit/phase_1b/phase_1b_first_round_report.json")
    args = ap.parse_args()

    campaign_dir = Path(args.campaign_dir)
    retrieval_dir = Path(args.retrieval_dir)
    feature_dir = Path(args.feature_dir)
    review_dir = Path(args.review_dir)
    event_dir = Path(args.event_dir)
    training_dir = Path(args.training_dir)

    campaign_paths = sorted(campaign_dir.glob("*.json"))
    if not campaign_paths:
        raise RuntimeError(f"no campaign files found: {campaign_dir}")

    campaigns = [read_json(p) for p in campaign_paths]
    campaign_ids = [c["campaign_id"] for c in campaigns]

    if len(campaign_ids) != len(set(campaign_ids)):
        raise RuntimeError("duplicate campaign_id found")

    required_paths: dict[str, Path] = {}
    for campaign_id in campaign_ids:
        required_paths[f"retrieval::{campaign_id}"] = retrieval_dir / f"clip_retrieval_candidates__{campaign_id}.jsonl"
        required_paths[f"feature::{campaign_id}"] = feature_dir / f"pair_feature_snapshots__{campaign_id}.jsonl"
        required_paths[f"review_queue::{campaign_id}"] = review_dir / f"review_queue__{campaign_id}.csv"
        required_paths[f"labeled_review_queue::{campaign_id}"] = review_dir / f"review_queue__{campaign_id}.labeled.csv"
        required_paths[f"review_events::{campaign_id}"] = event_dir / f"review_events__{campaign_id}.jsonl"
        required_paths[f"classifier_training::{campaign_id}"] = training_dir / f"training_snapshot_classifier__{campaign_id}.jsonl"
        required_paths[f"ranker_training::{campaign_id}"] = training_dir / f"training_snapshot_ranker__{campaign_id}.jsonl"

    required_paths["review_events_cumulative"] = review_dir / "review_events_phase1b_v1.jsonl"
    required_paths["classifier_training_cumulative"] = review_dir / "training_snapshot_phase1b_classifier_v1.jsonl"
    required_paths["ranker_training_cumulative"] = review_dir / "training_snapshot_phase1b_ranker_v1.jsonl"

    for path in required_paths.values():
        require_file(path)

    campaign_family_counts = Counter(c.get("campaign_family") for c in campaigns)
    purpose_counts = Counter(c.get("purpose_type") for c in campaigns)
    space_counts = Counter(c.get("space_type") for c in campaigns)
    season_counts = Counter(c.get("season") for c in campaigns)

    campaign_summaries: dict[str, Any] = {}

    cumulative_decisions = Counter()
    cumulative_decision_labels = Counter()
    cumulative_issue_tags = Counter()
    cumulative_buckets = Counter()
    cumulative_rows = 0

    bucket_label_matrix: dict[str, Counter] = defaultdict(Counter)
    campaign_label_matrix: dict[str, Counter] = defaultdict(Counter)

    for campaign in campaigns:
        campaign_id = campaign["campaign_id"]

        retrieval_rows = read_jsonl(retrieval_dir / f"clip_retrieval_candidates__{campaign_id}.jsonl")
        feature_rows = read_jsonl(feature_dir / f"pair_feature_snapshots__{campaign_id}.jsonl")
        queue_df = pd.read_csv(review_dir / f"review_queue__{campaign_id}.csv")
        labeled_df = pd.read_csv(review_dir / f"review_queue__{campaign_id}.labeled.csv")
        event_rows = read_jsonl(event_dir / f"review_events__{campaign_id}.jsonl")
        classifier_rows = read_jsonl(training_dir / f"training_snapshot_classifier__{campaign_id}.jsonl")
        ranker_rows = read_jsonl(training_dir / f"training_snapshot_ranker__{campaign_id}.jsonl")

        if len(retrieval_rows) != 50:
            raise RuntimeError(f"{campaign_id}: expected 50 retrieval rows, got {len(retrieval_rows)}")
        if len(feature_rows) != 50:
            raise RuntimeError(f"{campaign_id}: expected 50 feature rows, got {len(feature_rows)}")
        if len(queue_df) != 30:
            raise RuntimeError(f"{campaign_id}: expected 30 queue rows, got {len(queue_df)}")
        if len(labeled_df) != 30:
            raise RuntimeError(f"{campaign_id}: expected 30 labeled rows, got {len(labeled_df)}")
        if len(event_rows) != 30:
            raise RuntimeError(f"{campaign_id}: expected 30 event rows, got {len(event_rows)}")
        if len(classifier_rows) != 30 or len(ranker_rows) != 30:
            raise RuntimeError(f"{campaign_id}: expected 30 training rows for classifier/ranker")

        if queue_df["duplicate_group_id"].nunique() != len(queue_df):
            raise RuntimeError(f"{campaign_id}: duplicate suppression failed in review queue")

        blank_rejects = labeled_df[
            (labeled_df["decision"] == 0)
            & (labeled_df["issue_tags"].fillna("").str.strip() == "")
        ]
        if len(blank_rejects):
            raise RuntimeError(f"{campaign_id}: blank reject issue_tags found")

        decision_counts = Counter(labeled_df["decision"])
        decision_label_counts = Counter(labeled_df["decision_label"])
        issue_tag_counts = counter_from_tags(labeled_df["issue_tags"])
        bucket_counts = Counter(queue_df["source_bucket"])

        cumulative_decisions.update(decision_counts)
        cumulative_decision_labels.update(decision_label_counts)
        cumulative_issue_tags.update(issue_tag_counts)
        cumulative_buckets.update(bucket_counts)
        cumulative_rows += len(labeled_df)

        for event in event_rows:
            bucket = event["review_context"]["source_bucket"]
            label = event["decision"]["label"]
            bucket_label_matrix[bucket][label] += 1
            campaign_label_matrix[campaign_id][label] += 1

        positive_count = int(decision_counts.get(1, 0) + decision_counts.get(2, 0))
        reject_count = int(decision_counts.get(0, 0))

        campaign_summaries[campaign_id] = {
            "campaign_family": campaign.get("campaign_family"),
            "purpose_type": campaign.get("purpose_type"),
            "space_type": campaign.get("space_type"),
            "season": campaign.get("season"),
            "retrieval_rows": len(retrieval_rows),
            "pair_feature_snapshot_rows": len(feature_rows),
            "review_queue_rows": len(queue_df),
            "unique_duplicate_groups_in_queue": int(queue_df["duplicate_group_id"].nunique()),
            "review_event_rows": len(event_rows),
            "classifier_training_rows": len(classifier_rows),
            "ranker_training_rows": len(ranker_rows),
            "decision_counts": counter_to_plain(decision_counts),
            "decision_label_counts": counter_to_plain(decision_label_counts),
            "issue_tag_counts": counter_to_plain(issue_tag_counts),
            "bucket_counts": counter_to_plain(bucket_counts),
            "positive_count_accept_or_acceptable": positive_count,
            "reject_count": reject_count,
            "diagnostic_positive_rate": positive_count / len(labeled_df),
            "diagnostic_reject_rate": reject_count / len(labeled_df),
        }

    cumulative_events = read_jsonl(review_dir / "review_events_phase1b_v1.jsonl")
    cumulative_classifier = read_jsonl(review_dir / "training_snapshot_phase1b_classifier_v1.jsonl")
    cumulative_ranker = read_jsonl(review_dir / "training_snapshot_phase1b_ranker_v1.jsonl")

    if len(cumulative_events) != 150:
        raise RuntimeError(f"expected 150 cumulative review events, got {len(cumulative_events)}")
    if len(cumulative_classifier) != 150:
        raise RuntimeError(f"expected 150 cumulative classifier rows, got {len(cumulative_classifier)}")
    if len(cumulative_ranker) != 150:
        raise RuntimeError(f"expected 150 cumulative ranker rows, got {len(cumulative_ranker)}")

    event_ids = [r["review_event_id"] for r in cumulative_events]
    if len(event_ids) != len(set(event_ids)):
        raise RuntimeError("duplicate review_event_id found")

    classifier_label_counts = Counter(r["label"] for r in cumulative_classifier)
    ranker_label_counts = Counter(r["label"] for r in cumulative_ranker)

    layout_related_tags = {
        "text_region_conflict",
        "low_contrast",
        "too_busy_background",
        "visual_hierarchy_weak",
    }
    layout_issue_count = sum(cumulative_issue_tags.get(tag, 0) for tag in layout_related_tags)

    report = {
        "metadata": {
            "spec_version": "v2.2.1",
            "phase": "phase_1b",
            "run_id": "phase1b_first_round_cold_start_all_campaigns",
            "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "score_status": "diagnostic_only",
            "report_status": "first_round_audit",
            "support_explanation_status": "deferred_from_phase_1b",
        },

        "phase_scope_alignment": {
            "minimum_campaigns_required": 5,
            "minimum_campaign_families_required": 4,
            "actual_campaigns": len(campaigns),
            "actual_campaign_families": len(campaign_family_counts),
            "first_round_anchor_policy": "cold_start_all_campaigns",
            "support_explanation": "deferred",
            "runner_scope": "shell_script_runner",
            "threshold_policy": "no calibrated accept/reject threshold produced",
        },

        "campaign_distribution": {
            "campaign_ids": campaign_ids,
            "campaign_family_counts": counter_to_plain(campaign_family_counts),
            "purpose_type_counts": counter_to_plain(purpose_counts),
            "space_type_counts": counter_to_plain(space_counts),
            "season_counts": counter_to_plain(season_counts),
        },

        "artifact_counts": {
            "retrieval_files": len(campaign_ids),
            "retrieval_rows_total": 50 * len(campaign_ids),
            "pair_feature_snapshot_files": len(campaign_ids),
            "pair_feature_snapshot_rows_total": 50 * len(campaign_ids),
            "review_queue_files": len(campaign_ids),
            "review_queue_rows_total": int(cumulative_rows),
            "review_event_files": len(campaign_ids),
            "review_event_rows_total": len(cumulative_events),
            "classifier_training_rows_total": len(cumulative_classifier),
            "ranker_training_rows_total": len(cumulative_ranker),
        },

        "cumulative_review_summary": {
            "decision_counts": counter_to_plain(cumulative_decisions),
            "decision_label_counts": counter_to_plain(cumulative_decision_labels),
            "issue_tag_counts": counter_to_plain(cumulative_issue_tags),
            "bucket_counts": counter_to_plain(cumulative_buckets),
            "classifier_label_counts": counter_to_plain(classifier_label_counts),
            "ranker_label_counts": counter_to_plain(ranker_label_counts),
        },

        "bucket_label_matrix": {
            bucket: counter_to_plain(counter)
            for bucket, counter in sorted(bucket_label_matrix.items())
        },

        "campaign_label_matrix": {
            campaign_id: counter_to_plain(counter)
            for campaign_id, counter in sorted(campaign_label_matrix.items())
        },

        "campaign_summaries": campaign_summaries,

        "diagnostic_findings": [
            {
                "name": "indoor_gallery_winter_pool_coverage_gap",
                "status": "diagnostic_warning",
                "evidence": {
                    "campaign_id": "phase1b_indoor_gallery_winter_art",
                    "reject_count": campaign_summaries["phase1b_indoor_gallery_winter_art"]["reject_count"],
                    "review_rows": campaign_summaries["phase1b_indoor_gallery_winter_art"]["review_queue_rows"],
                    "issue_tags": campaign_summaries["phase1b_indoor_gallery_winter_art"]["issue_tag_counts"],
                },
                "interpretation": (
                    "The indoor/winter campaign produced 29 rejects out of 30 reviewed candidates. "
                    "This is most likely an image-pool coverage problem rather than a calibrated model-quality result."
                ),
            },
            {
                "name": "issue_tags_concentrated_in_semantic_and_season",
                "status": "diagnostic_warning",
                "evidence": {
                    "semantic_mismatch": cumulative_issue_tags.get("semantic_mismatch", 0),
                    "season_mismatch": cumulative_issue_tags.get("season_mismatch", 0),
                    "all_issue_tags": counter_to_plain(cumulative_issue_tags),
                },
                "interpretation": (
                    "Most rejects are explained by semantic or season mismatch. "
                    "This is useful for campaign coverage diagnosis, but it also shows that layout/critic labels remain sparse."
                ),
            },
            {
                "name": "layout_related_issue_tags_sparse",
                "status": "preview_renderer_trigger",
                "evidence": {
                    "layout_related_issue_count": layout_issue_count,
                    "layout_related_tags": {
                        tag: cumulative_issue_tags.get(tag, 0)
                        for tag in sorted(layout_related_tags)
                    },
                },
                "interpretation": (
                    "Layout-related issue tags are nearly absent. "
                    "This supports the Phase 1b scope decision that preview renderer v1 should be considered before critic training."
                ),
            },
            {
                "name": "classifier_data_available_but_imbalanced",
                "status": "diagnostic_note",
                "evidence": {
                    "classifier_label_counts": counter_to_plain(classifier_label_counts),
                    "ranker_label_counts": counter_to_plain(ranker_label_counts),
                },
                "interpretation": (
                    "Classifier and ranker snapshots now have 150 rows each. "
                    "This is enough for schema and smoke training checks, but not enough for production quality claims."
                ),
            },
            {
                "name": "heldout_campaign_split_minimum_structure_available",
                "status": "diagnostic_note",
                "evidence": {
                    "campaign_count": len(campaigns),
                    "campaign_family_count": len(campaign_family_counts),
                    "campaign_rows": {
                        campaign_id: campaign_summaries[campaign_id]["review_event_rows"]
                        for campaign_id in campaign_ids
                    },
                },
                "interpretation": (
                    "The minimum formal structure for held-out campaign splitting now exists. "
                    "Any metric based on it remains diagnostic_only until more data and leakage checks are added."
                ),
            },
        ],

        "recommended_next_actions": [
            {
                "priority": 1,
                "action": "expand_raw_image_pool",
                "rationale": (
                    "Indoor/winter campaign coverage is weak. "
                    "Adding more raw images should come before interpreting model failure."
                ),
            },
            {
                "priority": 2,
                "action": "add_preview_renderer_v1",
                "rationale": (
                    "Layout-related issue tags are sparse. "
                    "A box-overlay preview renderer should improve text-region and background-complexity labeling."
                ),
            },
            {
                "priority": 3,
                "action": "run_classifier_smoke_training",
                "rationale": (
                    "There are now 150 classifier rows. "
                    "Smoke training can validate feature plumbing, but results must remain diagnostic_only."
                ),
            },
        ],

        "non_claims": [
            "No production reranker quality is claimed.",
            "No calibrated accept/reject threshold is claimed.",
            "No LGBMRanker generalization quality is claimed.",
            "No Visual Critic performance is claimed.",
            "No final poster/PPTX/Canva quality is claimed.",
            "Support explanation remains deferred from Phase 1b.",
        ],
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "event": "done",
        "out": str(out_path),
        "campaigns": len(campaigns),
        "campaign_families": len(campaign_family_counts),
        "review_events": len(cumulative_events),
        "classifier_rows": len(cumulative_classifier),
        "ranker_rows": len(cumulative_ranker),
        "decision_label_counts": counter_to_plain(cumulative_decision_labels),
        "issue_tag_counts": counter_to_plain(cumulative_issue_tags),
        "score_status": "diagnostic_only",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
