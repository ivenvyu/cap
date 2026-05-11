#!/usr/bin/env bash
set -euo pipefail

REVIEW_DIR="${1:-data/review/phase1b}"
FEATURE_DIR="${2:-data/feature_snapshots/v2_2_1/phase1b}"
OUT_DIR="${3:-data/review/phase1b/events}"

mkdir -p "${OUT_DIR}"

echo "review_dir=${REVIEW_DIR}"
echo "feature_dir=${FEATURE_DIR}"
echo "out_dir=${OUT_DIR}"

for review_csv in "${REVIEW_DIR}"/review_queue__*.labeled.csv; do
  filename="$(basename "${review_csv}")"
  campaign_id="${filename#review_queue__}"
  campaign_id="${campaign_id%.labeled.csv}"

  feature_path="${FEATURE_DIR}/pair_feature_snapshots__${campaign_id}.jsonl"
  out_path="${OUT_DIR}/review_events__${campaign_id}.jsonl"
  queue_version="review_queue_phase1b_v1__${campaign_id}"

  if [[ ! -s "${feature_path}" ]]; then
    echo "missing feature snapshot file: ${feature_path}" >&2
    exit 1
  fi

  echo
  echo "=== ReviewEvent ingest: ${campaign_id} ==="

  python scripts/ingest_phase1a_review_events.py \
    --review-csv "${review_csv}" \
    --feature-snapshots "${feature_path}" \
    --out "${out_path}" \
    --queue-version "${queue_version}" \
    --annotator-id "human_001"
done

cat "${OUT_DIR}"/review_events__*.jsonl > "${REVIEW_DIR}/review_events_phase1b_v1.jsonl"

echo
echo "wrote cumulative: ${REVIEW_DIR}/review_events_phase1b_v1.jsonl"
echo "PHASE 1B REVIEW EVENT INGEST DONE"
