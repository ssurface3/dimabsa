#!/bin/bash
set -e
MODEL="jhu-clsp/mmBERT-base"
BS=18
ACCUM=4
LR=1e-5
EPOCHS=5 
LOSS_TYPE="CE * 0.1 + custom loss * 0.9"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXP_ID="mmBERT-base-finetuned-${TIMESTAMP}"

# TRAIN_DATA="/kaggle/working/dimabsa/initial_data_train/"
TRAIN_DATA="/kaggle/working/dimabsa/initial_data_train/rus_restaurant_train_alltasks.jsonl"
# EVAL_DATA="/kaggle/working/dimabsa/dimabsa marked dev"
EVAL_DATA="/kaggle/working/dimabsa/dimabsa_marked_dev/rus_restaurant_dev_task1.jsonl"
# TEST_DATA="/kaggle/working/dimabsa/data_final/test_with_chinese.jsonl"
export TORCHDYNAMO_DISABLE=1

mkdir -p results

echo "----------------------------------------------------"
echo "Starting Training: $MODEL"
echo "----------------------------------------------------"

python /kaggle/working/dimabsa/bertcls_new/train_cls_bins.py \
    --model_name "$MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --output_dir "$EXP_ID" \
    --epochs $EPOCHS \
    --batch_size $BS \
    --grad_accum $ACCUM \
    --lr $LR \
    --loss_type "$LOSS_TYPE"

echo "training is over"

# echo "----------------------------------------------------"
# echo "Starting inference on test set"
# echo "----------------------------------------------------"

# python al.py \
#     --model_path "models/$EXP_ID/final" \
#     --data_path "$TEST_DATA" 
# echo "test is tested"

