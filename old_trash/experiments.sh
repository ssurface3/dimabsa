#!/bin/bash
set -e


DATA_DIR="data"
RESULTS_DIR="results"
LOGS_DIR="logs"
EPOCHS=4

# Check for auth file
if [ ! -f "mycreds.txt" ]; then
    echo "Error: 'mycreds.txt' not found. Run 'python setup_drive.py' first!"
    exit 1
fi

EXPERIMENTS=(
    "microsoft/deberta-v3-base 16 1 2e-5"    
    "microsoft/deberta-v3-large 4 4 1e-5"
    "Qwen/Qwen2.5-0.5B 8 2 2e-5"
    "jhu-clsp/mmBERT-base 16 1 2e-5"
    "roberta-large 4 4 1e-5"
)

echo "start"


for EXP_CONFIG in "${EXPERIMENTS[@]}"
do
    set -- $EXP_CONFIG
    MODEL_NAME=$1
    BS=$2
    ACCUM=$3
    LR=$4

    SAFE_NAME=$(echo $MODEL_NAME | tr '/' '_')
    EXP_ID="${SAFE_NAME}_bs${BS}_acc${ACCUM}_lr${LR}"

    echo "----------------------------------------------------"
    echo "model: $MODEL_NAME"
    echo "Params: Batch=$BS | Accum=$ACCUM | LR=$LR"
    echo "----------------------------------------------------"


    python train.py \
        --model_name "$MODEL_NAME" \
        --data_path "$DATA_DIR/train.jsonl" \
        --output_dir "$EXP_ID" \
        --epochs $EPOCHS \
        --batch_size $BS \
        --grad_accum $ACCUM \
        --lr $LR

    SUBMISSION_FILE="$RESULTS_DIR/${EXP_ID}_submission.json"
    
    python predict.py \
        --model_path "models/$EXP_ID/final" \
        --test_data "$DATA_DIR/test.jsonl" \
        --output_file "$SUBMISSION_FILE"


    ZIP_NAME="${EXP_ID}_results.zip"
    echo "Zipping & Uploading"
    
    zip -r $ZIP_NAME "$SUBMISSION_FILE" "$LOGS_DIR"
    
    python upload_file.py $ZIP_NAME
    
    
    # rm $ZIP_NAME
    # rm -rf "models/$EXP_ID" 
    
    echo "Finished $MODEL_NAME"
done

echo "check the drive to look for experiments "