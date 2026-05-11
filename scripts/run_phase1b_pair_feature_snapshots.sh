#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_DIR="${1:-examples/campaigns/phase1b}"
RETRIEVAL_DIR="${2:-data/retrieval/phase1b}"
OUT_DIR="${3:-data/feature_snapshots/v2_2_2/phase1b}"

mkdir -p "${OUT_DIR}"

echo "campaign_dir=${CAMPAIGN_DIR}"
echo "retrieval_dir=${RETRIEVAL_DIR}"
echo "out_dir=${OUT_DIR}"

for campaign_path in "${CAMPAIGN_DIR}"/*.json; do
  campaign_id="$(
    python - "$campaign_path" <<'PY'
import json
import sys
from pathlib import Path

p = Path(sys.argv[1])
data = json.loads(p.read_text(encoding="utf-8"))
print(data["campaign_id"])
PY
  )"

  retrieval_path="${RETRIEVAL_DIR}/clip_retrieval_candidates__${campaign_id}.jsonl"
  out_path="${OUT_DIR}/pair_feature_snapshots__${campaign_id}.jsonl"

  if [[ ! -s "${retrieval_path}" ]]; then
    echo "missing retrieval file: ${retrieval_path}" >&2
    exit 1
  fi

  echo
  echo "=== PairFeatureSnapshot: ${campaign_id} ==="

  python scripts/build_pair_feature_snapshots.py \
    --campaign "${campaign_path}" \
    --retrieval "${retrieval_path}" \
    --manifest data/ontology/raw_image_manifest_v2_2_1.jsonl \
    --duplicates data/ontology/duplicate_groups_v1.jsonl \
    --clusters data/ontology/dinov2_clusters_v1.jsonl \
    --region-safety data/ontology/region_safety_maps_v1.jsonl \
    --out "${out_path}" \
    --batch-id "phase1b_pair_features_v2_2_2__${campaign_id}" \
    --snapshot-version v2_2_2
done

echo
echo "PHASE 1B PAIR FEATURE SNAPSHOTS DONE"
