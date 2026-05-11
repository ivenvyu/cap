#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN_DIR="${1:-examples/campaigns/phase1b}"
OUT_DIR="${2:-data/retrieval/phase1b}"
TOP_K="${TOP_K:-50}"
DEVICE="${DEVICE:-auto}"

mkdir -p "${OUT_DIR}"

echo "campaign_dir=${CAMPAIGN_DIR}"
echo "out_dir=${OUT_DIR}"
echo "top_k=${TOP_K}"
echo "device=${DEVICE}"

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

  echo
  echo "=== CLIP retrieval: ${campaign_id} ==="

  python scripts/build_clip_retrieval_candidates.py \
    --campaign "${campaign_path}" \
    --prompt-bank configs/prompt_template_bank_v1.yaml \
    --clip-embeddings data/embeddings/clip_image_embeddings.npy \
    --clip-index data/embeddings/clip_image_index.csv \
    --prompt-set-out "${OUT_DIR}/prompt_set__${campaign_id}.jsonl" \
    --out "${OUT_DIR}/clip_retrieval_candidates__${campaign_id}.jsonl" \
    --device "${DEVICE}" \
    --top-k "${TOP_K}"
done

echo
echo "PHASE 1B CLIP RETRIEVAL DONE"
