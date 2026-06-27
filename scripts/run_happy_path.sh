#!/usr/bin/env bash

set -euo pipefail

# Usage:
#   ./scripts/run_happy_path.sh [SYMBOL] [MAX_ROWS_BATCH]
# Example:
#   ./scripts/run_happy_path.sh BTCUSDT 10000000

SYMBOL="${1:-BTCUSDT}"
MAX_ROWS_BATCH="${2:-10000000}"

echo "Running happy path for symbol=${SYMBOL}, max_rows_batch=${MAX_ROWS_BATCH}"

python 00_upload_history.py --only-symbols "${SYMBOL}"
python 02_add_features.py --symbol "${SYMBOL}" --max-rows-batch "${MAX_ROWS_BATCH}"
python 03_add_targets.py --symbol "${SYMBOL}" --max-rows-batch "${MAX_ROWS_BATCH}"
python 05_chunks_marker.py \
  --symbol "${SYMBOL}" \
  --start-date 2024-01-01T00:00:00Z \
  --end-date 2026-03-01T00:00:00Z \
  --tables-to-mark spot_trades \
  --debug

echo "Happy path completed."
