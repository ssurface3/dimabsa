#!/bin/bash

USE_HUB=false
HF_REPO_ID="ssurface/dimabsa-2026-subtask1-private-mmbert-alldata_last"
HF_TOKEN=""

LOCAL_MODEL_PATH="/kaggle/working/dimabsa/test_models/models/jhu-clsp/mmBERT-base-finetuned-dimabsa-laptop-alltasks/checkpoint-2520"
TEST_DATA="/kaggle/working/dimabsa/dimabsa_test_gold"
BATCH_SIZE=32
MAX_LEN=50
BASE_TOKENIZER='jhu-clsp/mmbert-base'
if [ "$USE_HUB" = true ]; then
    MODEL_PATH=$HF_REPO_ID
    export HF_TOKEN=$HF_TOKEN
else
    MODEL_PATH=$LOCAL_MODEL_PATH
fi

MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="predictions/$MODEL_NAME"

mkdir -p "$OUTPUT_DIR"
sleep 5
python generate_sub.py\
    --model_path "$MODEL_PATH" \
    --test_data "$TEST_DATA" \
    --output_file "$OUTPUT_DIR/submission.jsonl" \
    --batch_size "$BATCH_SIZE" \
    --max_len "$MAX_LEN" \
    --base_tokenizer "$BASE_TOKENIZER"

python evaluate.py \
    --pred_file "$OUTPUT_DIR" \
    --gold_file "$TEST_DATA" \
    --output_dir "$OUTPUT_DIR/stats/$HF_REPO_ID"