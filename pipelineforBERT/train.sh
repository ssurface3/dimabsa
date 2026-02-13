#!/bin/bash
set -e

export TF_CPP_MIN_LOG_LEVEL=3
export GRPC_VERBOSITY=ERROR
export GLOG_minloglevel=3

MODEL="jhu-clsp/mmBERT-base"
BS=8
ACCUM=4
LR=2e-5
EPOCHS=10 
EXP_ID="jhu-clsp/mmBERT-base-finetuned-dimabsa-laptop-alltasks"

TRAIN_DATA="/kaggle/working/dimabsa_data/train.jsonl"
EVAL_DATA="/kaggle/working/dimabsa_data/dev.jsonl"
TEST_DATA="/kaggle/working/dimabsa_data/dev.jsonl"
export TORCHDYNAMO_DISABLE=1

mkdir -p results

echo "----------------------------------------------------"
echo "Starting Training: $MODEL"
echo "----------------------------------------------------"

python /kaggle/working/dimabsa/pipelineforBERT/train_mse.py \
    --model_name "$MODEL" \
    --train_data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --test_data_path "$TEST_DATA" \
    --output_dir "$EXP_ID" \
    --epochs $EPOCHS \
    --batch_size $BS \
    --grad_accum $ACCUM \
    --lr $LR 

# echo "training is over"

# echo "----------------------------------------------------"
# echo "Starting inference on test set"
# echo "----------------------------------------------------"

# python /kaggle/working/dimabsa/al.py \
#     --model_path "models/$EXP_ID/final" \
#     --data_path "$TEST_DATA" 
# echo "test is tested"

