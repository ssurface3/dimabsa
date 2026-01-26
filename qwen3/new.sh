#!/bin/bash
set -e

# export TF_CPP_MIN_LOG_LEVEL=3
# export GRPC_VERBOSITY=ERROR
# export GLOG_minloglevel=3

# MODEL="Qwen/Qwen3-Embedding-8B"
MODEL='Qwen/Qwen3-0.6B'
BS=8
ACCUM=4
LR=1e-5
EPOCHS=3
EXP_ID="Qwen/Qwen3-0.6B-dimabsa-alltasks"

TRAIN_DATA="/kaggle/working/dimabsa/out_put_plit_no_emobank/train.jsonl"
EVAL_DATA="/kaggle/working/dimabsa/out_put_plit_no_emobank/dev.jsonl"
TEST_DATA="/kaggle/working/dimabsa/out_put_plit_no_emobank/test.jsonl"
# export TORCHDYNAMO_DISABLE=1

mkdir -p results

echo "----------------------------------------------------"
echo "Starting Training: $MODEL"
echo "----------------------------------------------------"

python /kaggle/working/dimabsa/qwen3/train_qwen.py \
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
echo "Starting inference on test set"
echo "----------------------------------------------------"




# python /Users/anatoliifrolov/Downloads/dimABSA/convert_chinese.py \
#     --chinese_dir "/Users/anatoliifrolov/Downloads/dimABSA/chinese_data" \
#     --splits_dir "/Users/anatoliifrolov/Downloads/dimABSA/data/data_split_chinese"