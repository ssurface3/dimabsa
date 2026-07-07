#!/bin/bash
# ============================================================
# SFT re-run with completion-only loss fix.
# Trains all levels (0-6) sequentially.
# bs=64, grad_accum=1 → effective batch size 64.
#
# Usage:
#   bash run_sft_new.sh                  # all levels
#   bash run_sft_new.sh --levels 1 2 3  # specific levels
# ============================================================

set -o pipefail

PROJECT_DIR="/home/jovyan/.mlspace/envs/student_env/grpo"
CONDA_ENV="grpo"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"

EPOCHS=2
LR="2e-4"

# Sequence lengths per level
MAX_SEQ_L0=2048
MAX_SEQ_L6=256
MAX_SEQ_DEFAULT=1024

# Batch sizes scaled so peak activation memory stays constant:
#   L6:  bs=64 × seq=256   → 16k tokens/step
#   L0:  bs=8  × seq=2048  → 16k tokens/step, accum=8 → eff_bs=64
#   L1-5: bs=16 × seq=1024 → 16k tokens/step, accum=4 → eff_bs=64
BS_L6=64;   GRAD_ACCUM_L6=1
BS_L0=8;    GRAD_ACCUM_L0=8
BS_DEFAULT=16; GRAD_ACCUM_DEFAULT=4

# ── parse optional --levels override ─────────────────────────────────────────
ALL_LEVELS=(0 1 2 3 4 5 6)
LEVELS=()
PARSE_LEVELS=0
for arg in "$@"; do
    if [ "$arg" = "--levels" ]; then
        PARSE_LEVELS=1
    elif [ "$PARSE_LEVELS" -eq 1 ]; then
        LEVELS+=("$arg")
    fi
done
if [ ${#LEVELS[@]} -eq 0 ]; then
    LEVELS=("${ALL_LEVELS[@]}")
fi

# ── env setup ────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"
if command -v conda &> /dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    set +u; conda activate "$CONDA_ENV" 2>/dev/null || true; set -u
fi

LOG_DIR="./logs/sft_new"
mkdir -p "$LOG_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "Starting SFT re-run: levels=${LEVELS[*]}  epochs=$EPOCHS  (bs/accum per level: L6=${BS_L6}/1  L0=${BS_L0}/8  L1-5=${BS_DEFAULT}/4)"

# ── per-level training ────────────────────────────────────────────────────────
for L in "${LEVELS[@]}"; do

    OUT="./checkpoints_new/level_${L}"
    FINAL="$OUT/final_adapter"
    LOGF="$LOG_DIR/l${L}_sft.log"

    if [ -d "$FINAL" ]; then
        log "SKIP level $L — final_adapter already exists at $FINAL"
        continue
    fi

    # ── Level 6: standalone script ──────────────────────────────────────────
    if [ "$L" -eq 6 ]; then
        log "Training level 6 (No-Think SFT) → $OUT  bs=$BS_L6 accum=$GRAD_ACCUM_L6"
        python3 logic_train/train_l6.py \
            --base_model                    "$BASE_MODEL" \
            --output_dir                    "$OUT" \
            --data_dir                      "./initial_data/prepared_train/level_6" \
            --num_train_epochs              "$EPOCHS" \
            --per_device_train_batch_size   "$BS_L6" \
            --gradient_accumulation_steps   "$GRAD_ACCUM_L6" \
            --learning_rate                 "$LR" \
            --max_seq_length                "$MAX_SEQ_L6" \
            2>&1 | tee "$LOGF"

    # ── Levels 0–5: shared helpers/train.py ─────────────────────────────────
    else
        DATA_DIR="./initial_data/prepared_train/level_${L}"

        if [ ! -d "$DATA_DIR" ]; then
            log "SKIP level $L — data dir missing: $DATA_DIR"
            continue
        fi

        if [ "$L" -eq 0 ]; then
            MAX_SEQ="$MAX_SEQ_L0"
            BS="$BS_L0"
            GRAD_ACCUM="$GRAD_ACCUM_L0"
        else
            MAX_SEQ="$MAX_SEQ_DEFAULT"
            BS="$BS_DEFAULT"
            GRAD_ACCUM="$GRAD_ACCUM_DEFAULT"
        fi

        log "Training level $L → $OUT  (max_seq=$MAX_SEQ  bs=$BS  accum=$GRAD_ACCUM)"
        PYTHONPATH="helpers" python3 helpers/train.py \
            --base_model                    "$BASE_MODEL" \
            --level                         "$L" \
            --data_dir                      "$DATA_DIR" \
            --output_dir                    "$OUT" \
            --num_train_epochs              "$EPOCHS" \
            --per_device_train_batch_size   "$BS" \
            --gradient_accumulation_steps   "$GRAD_ACCUM" \
            --learning_rate                 "$LR" \
            --max_seq_length                "$MAX_SEQ" \
            2>&1 | tee "$LOGF"
    fi

    if [ $? -eq 0 ]; then
        log "OK level $L → $FINAL"
    else
        log "FAIL level $L (see $LOGF)"
        exit 1
    fi

done

log "All SFT runs complete."
log "Adapters in ./checkpoints_new/level_*/ final_adapter/"
