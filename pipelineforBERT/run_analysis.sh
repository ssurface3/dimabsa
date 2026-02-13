#!/bin/bash


PRED_DIR="/kaggle/working/dimabsa/test_models/predictions/checkpoint-2520"
GOLD_DIR="/kaggle/working/dimabsa/dimabsa_test_gold" # Or the dev folder

OUTPUT_FILE="error_analysis_merged.jsonl"


if [ "$1" != "" ]; then
    PRED_DIR="$1"
fi

if [ "$2" != "" ]; then
    GOLD_DIR="$2"
fi

if [ "$3" != "" ]; then
    OUTPUT_FILE="$3"
fi

echo "Running Error Analysis..."
echo "Prediction Directory: $PRED_DIR"
echo "Gold Directory:       $GOLD_DIR"
echo "Output File:          $OUTPUT_FILE"

python /kaggle/working/dimabsa/pipelineforBERT/analyze_errors.py\
    --pred_dir "$PRED_DIR" \
    --gold_dir "$GOLD_DIR" \
    --output_file "$OUTPUT_FILE"

echo "Done. Results saved to $OUTPUT_FILE"
