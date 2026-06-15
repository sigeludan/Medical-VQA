#!/usr/bin/env bash
# Fast VQA-RAD download via hf-mirror + aria2 (works well in China / AutoDL)
set -euo pipefail

OUT_DIR="${1:-/root/autodl-tmp/VQA/data/vqa-rad-raw}"
BASE="https://hf-mirror.com/datasets/flaviagiammarino/vqa-rad/resolve/main"

mkdir -p "${OUT_DIR}/data" "${OUT_DIR}/scripts"

download_one() {
  local out="$1"
  local url="$2"
  echo ">> $(basename "$out")"
  aria2c -x 16 -s 16 -k 1M -c --file-allocation=none \
    -o "$(basename "$out")" -d "$(dirname "$out")" "$url"
}

download_one "${OUT_DIR}/README.md" "${BASE}/README.md"
download_one "${OUT_DIR}/scripts/processing.py" "${BASE}/scripts/processing.py"
download_one "${OUT_DIR}/data/train-00000-of-00001-eb8844602202be60.parquet" \
  "${BASE}/data/train-00000-of-00001-eb8844602202be60.parquet"
download_one "${OUT_DIR}/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet" \
  "${BASE}/data/test-00000-of-00001-e5bc3d208bb4deeb.parquet"

echo ""
echo "Done. Files:"
ls -lh "${OUT_DIR}/data/"*.parquet
echo "Total: $(du -sh "${OUT_DIR}" | cut -f1)"
