#!/usr/bin/env bash
set -euo pipefail

REVIEW_DIR="${1:-data/review/phase1b}"
EVENT_DIR="${2:-data/review/phase1b/events}"
FEATURE_DIR="${3:-data/feature_snapshots/v2_2_1/phase1b}"
OUT_DIR="${4:-data/review/phase1b/training}"

mkdir -p "${OUT_DIR}"

echo "review_dir=${REVIEW_DIR}"
echo "event_dir=${EVENT_DIR}"
echo "feature_dir=${FEATURE_DIR}"
echo "out_dir=${OUT_DIR}"

for event_path in "${EVENT_DIR}"/review_events__*.jsonl; do
  filename="$(basename "${event_path}")"
  campaign_id="${filename#review_events__}"
  campaign_id="${campaign_id%.jsonl}"

  feature_path="${FEATURE_DIR}/pair_feature_snapshots__${campaign_id}.jsonl"

  classifier_out="${OUT_DIR}/training_snapshot_classifier__${campaign_id}.jsonl"
  ranker_out="${OUT_DIR}/training_snapshot_ranker__${campaign_id}.jsonl"

  if [[ ! -s "${feature_path}" ]]; then
    echo "missing feature snapshot file: ${feature_path}" >&2
    exit 1
  fi

  echo
  echo "=== TrainingSnapshot: ${campaign_id} ==="

  python scripts/build_phase1a_training_snapshots.py \
    --review-events "${event_path}" \
    --feature-snapshots "${feature_path}" \
    --classifier-out "${classifier_out}" \
    --ranker-out "${ranker_out}" \
    --snapshot-version-prefix "train_labels_phase1b_v1__${campaign_id}"
done

cat "${OUT_DIR}"/training_snapshot_classifier__*.jsonl > "${REVIEW_DIR}/training_snapshot_phase1b_classifier_v1.jsonl"
cat "${OUT_DIR}"/training_snapshot_ranker__*.jsonl > "${REVIEW_DIR}/training_snapshot_phase1b_ranker_v1.jsonl"

echo
echo "wrote cumulative classifier: ${REVIEW_DIR}/training_snapshot_phase1b_classifier_v1.jsonl"
echo "wrote cumulative ranker: ${REVIEW_DIR}/training_snapshot_phase1b_ranker_v1.jsonl"
echo "PHASE 1B TRAINING SNAPSHOTS DONE"
