#!/bin/bash
set -e

MODEL="microsoft/deberta-v3-large"
BS=4
ACCUM=4
LR=1.2e-5
EPOCHS=5
EXP_ID="deberta_v3_large_new_loss_warmup_lower_lr"

TRAIN_DATA="/kaggle/working/dimabsa/data/eng_laptop_train_alltasks.jsonl"
TEST_DATA="data/eng_laptop_dev_alltasks.jsonl" 

mkdir -p results

echo "----------------------------------------------------"
echo "Starting Training: $MODEL"
echo "----------------------------------------------------"

python train.py \
    --model_name "$MODEL" \
    --data_path "$TRAIN_DATA" \
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