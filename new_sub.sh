#!/bin/bash
set -e

MODEL_PATH="/kaggle/working/dimabsa/models/jhu-clsp/mmBERT-base-finetuned-dimabsa-laptop-alltasks/final"
DATA_DIR="/kaggle/working/dimabsa/data_sub_dimabsa"
SUBMISSION_FOLDER="subtask_1" 
BATCH_SIZE=32
MAX_LEN=50

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_NAME=$(basename "$(dirname "$MODEL_PATH")")
ZIP_NAME="${MODEL_NAME}_submission.zip"


rm -rf "$SUBMISSION_FOLDER"
mkdir -p "$SUBMISSION_FOLDER"

echo "Model: $MODEL_PATH"
echo "Target: $SUBMISSION_FOLDER"

shopt -s nullglob
for f in "$DATA_DIR"/*dev*.jsonl; do
  [ -f "$f" ] || continue
  
  filename=$(basename "$f" .jsonl)
  
  if [[ "$filename" == *"train"* ]]; then
    continue
  fi


  clean_name=${filename%%_dev*}
  clean_name=${clean_name%%_test*}
  
  output_file="$SUBMISSION_FOLDER/pred_${clean_name}.jsonl"
  
  echo "Generating: pred_${clean_name}.jsonl"
  
  python /kaggle/working/dimabsa/generate_submssion.py \
      --model_path "$MODEL_PATH" \
      --test_file "$f" \
      --output_file "$output_file" \
      --batch_size "$BATCH_SIZE" \
      --max_len "$MAX_LEN"
done

cd "$SCRIPT_DIR"

echo "Zipping..."
rm -f "$ZIP_NAME"


zip -r "$ZIP_NAME" "$SUBMISSION_FOLDER"

echo "Created: $ZIP_NAME"
echo "Structure check:"
unzip -l "$ZIP_NAME" | head -n 10