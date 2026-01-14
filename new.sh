#!/bin/bash
set -e

MODEL="jhu-clsp/mmBERT-base"
BS=18
ACCUM=4
LR=1e-5
EPOCHS=5 
EXP_ID="jhu-clsp/mmBERT-base-finetuned-dimabsa-laptop-alltasks"

TRAIN_DATA="/kaggle/working/dimabsa/data_split/train.jsonl"
EVAL_DATA="/kaggle/working/dimabsa/data_split/dev.jsonl"
TEST_DATA="/kaggle/working/dimabsa/data_split/test.jsonl"
export TORCHDYNAMO_DISABLE=1

mkdir -p results

echo "----------------------------------------------------"
echo "Starting Training: $MODEL"
echo "----------------------------------------------------"

python train.py \
    --model_name "$MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --output_dir "$EXP_ID" \
    --epochs $EPOCHS \
    --batch_size $BS \
    --grad_accum $ACCUM \
    --lr $LR 

echo "training is over"

echo "----------------------------------------------------"
echo "Starting Prediction"
echo "----------------------------------------------------"

python generate_sub.py \
    --model_path "models/$EXP_ID/final" \
    --test_data "$TEST_DATA" \
    --output_file "results/${EXP_ID}_submission.json" \
    --batch_size 16

echo "submission is generated"

