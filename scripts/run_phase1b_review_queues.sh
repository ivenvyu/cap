#!/usr/bin/env bash
set -euo pipefail

FEATURE_DIR="${1:-data/feature_snapshots/v2_2_1/phase1b}"
OUT_DIR="${2:-data/review/phase1b}"

CLIP_BUDGET="${CLIP_BUDGET:-12}"
LAYOUT_SAFE_BUDGET="${LAYOUT_SAFE_BUDGET:-6}"
CLUSTER_BUDGET="${CLUSTER_BUDGET:-8}"
UNCERTAINTY_BUDGET="${UNCERTAINTY_BUDGET:-3}"
RANDOM_BUDGET="${RANDOM_BUDGET:-1}"

mkdir -p "${OUT_DIR}"

echo "feature_dir=${FEATURE_DIR}"
echo "out_dir=${OUT_DIR}"
echo "clip_budget=${CLIP_BUDGET}"
echo "layout_safe_budget=${LAYOUT_SAFE_BUDGET}"
echo "cluster_budget=${CLUSTER_BUDGET}"
echo "uncertainty_budget=${UNCERTAINTY_BUDGET}"
echo "random_budget=${RANDOM_BUDGET}"

for feature_path in "${FEATURE_DIR}"/pair_feature_snapshots__*.jsonl; do
  filename="$(basename "${feature_path}")"
  campaign_id="${filename#pair_feature_snapshots__}"
  campaign_id="${campaign_id%.jsonl}"

  out_path="${OUT_DIR}/review_queue__${campaign_id}.csv"
  queue_id="review_queue_phase1b_v1__${campaign_id}"

  echo
  echo "=== Review queue: ${campaign_id} ==="

  python scripts/build_phase1a_review_queue.py \
    --features "${feature_path}" \
    --manifest data/ontology/raw_image_manifest_v2_2_1.jsonl \
    --out "${out_path}" \
    --queue-id "${queue_id}" \
    --clip-budget "${CLIP_BUDGET}" \
    --layout-safe-budget "${LAYOUT_SAFE_BUDGET}" \
    --cluster-budget "${CLUSTER_BUDGET}" \
    --uncertainty-budget "${UNCERTAINTY_BUDGET}" \
    --random-budget "${RANDOM_BUDGET}"
done

echo
echo "PHASE 1B REVIEW QUEUES DONE"
