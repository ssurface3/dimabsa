#!/usr/bin/env bash
set -euo pipefail

usage="Usage: $0 <model_path> <data_dir> [output_dir] [batch_size] [max_len]"
if [ "$#" -lt 2 ]; then
  echo "$usage"
  exit 1
fi

MODEL_PATH="$1"
DATA_DIR="$2"
OUT_DIR="${3:-subs}"
BATCH_SIZE="${4:-32}"
MAX_LEN="${5:-50}"

mkdir -p "$OUT_DIR"

model_tag=$(basename "$MODEL_PATH" | tr '/' '-')

shopt -s nullglob
for f in "$DATA_DIR"/*dev*.jsonl "$DATA_DIR"/*.jsonl; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .jsonl)
  out="$OUT_DIR/${model_tag}_${base}.json"
  echo "Running: $f -> $out"
  python dimabsa/generate_sub.py --model_path "$MODEL_PATH" --test_data "$f" --output_file "$out" --batch_size "$BATCH_SIZE" --max_len "$MAX_LEN"
  echo "Wrote $out"
done

echo "All done."