#!/bin/bash
# ============================================================
# GSM8K evaluator for Gemma models.
#
# Sources:
#   gemma_sft   — base + SFT adapter only
#   gemma_grpo  — base + SFT merged + GRPO adapter
#
# Usage:
#   bash run_eval_gemma.sh [SOURCE] [--levels 1 2 3] [--model <hf-id>]
#
# Examples:
#   bash run_eval_gemma.sh gemma_sft
#   bash run_eval_gemma.sh gemma_grpo
#   bash run_eval_gemma.sh gemma_grpo --levels 1 4 5
#   bash run_eval_gemma.sh gemma_sft  --levels 4 --model google/gemma-3-4b-it
# ============================================================

SOURCE="${1:-gemma_sft}"

set -o pipefail

PROJECT_DIR="/home/jovyan/.mlspace/envs/student_env/grpo"
BASE_GEMMA="google/gemma-4-E4B"          # override with --model flag

SFT_CKPT_ROOT="$PROJECT_DIR/checkpoints_gemma_sft"
GRPO_CKPT_ROOT="$PROJECT_DIR/checkpoints_gemma_grpo"

# ── Parse flags from remaining args ──────────────────────────────────────────
LEVELS=()
MODEL_OVERRIDE=""
PARSE_LEVELS=0
PARSE_MODEL=0
for arg in "${@:2}"; do
  case "$arg" in
    --levels) PARSE_LEVELS=1; PARSE_MODEL=0 ;;
    --model)  PARSE_MODEL=1;  PARSE_LEVELS=0 ;;
    *)
      if [ "$PARSE_LEVELS" -eq 1 ]; then
        LEVELS+=("$arg")
      elif [ "$PARSE_MODEL" -eq 1 ]; then
        MODEL_OVERRIDE="$arg"
        PARSE_MODEL=0
      fi
      ;;
  esac
done
[ -n "$MODEL_OVERRIDE" ] && BASE_GEMMA="$MODEL_OVERRIDE"
[ ${#LEVELS[@]} -eq 0 ]  && LEVELS=(1 2 3 4 5)

# ── Environment ───────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u; conda activate grpo 2>/dev/null || true; set -u
fi

cd "$PROJECT_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

LEVEL_NAMES=([1]="level 1" [2]="level 2" [3]="level 3" [4]="level 4" [5]="level 5")

case "$SOURCE" in
  gemma_sft)
    RESULTS_DIR="$PROJECT_DIR/eval_results_gemma_sft"
    ;;
  gemma_grpo)
    RESULTS_DIR="$PROJECT_DIR/eval_results_gemma_grpo"
    ;;
  *)
    echo "Unknown source: $SOURCE"
    echo "Options: gemma_sft  gemma_grpo"
    exit 1
    ;;
esac

mkdir -p "$RESULTS_DIR"
log "Source: $SOURCE  |  Model: $BASE_GEMMA  |  Levels: ${LEVELS[*]}  |  Results: $RESULTS_DIR"
echo ""

# ── Main loop ─────────────────────────────────────────────────────────────────
for L in "${LEVELS[@]}"; do
  LEVEL_STR="${LEVEL_NAMES[$L]}"
  OUTPUT="$RESULTS_DIR/l${L}.json"
  SFT_ADAPTER="$SFT_CKPT_ROOT/level_${L}/final_adapter"
  GRPO_ADAPTER="$GRPO_CKPT_ROOT/level_${L}/final_adapter"

  if [ -f "$OUTPUT" ]; then
    log "SKIP l${L} — results exist at $OUTPUT (delete to rerun)"
    continue
  fi

  case "$SOURCE" in
    gemma_sft)
      if [ ! -d "$SFT_ADAPTER" ]; then
        log "SKIP l${L} — SFT adapter not found: $SFT_ADAPTER"
        continue
      fi
      log "Evaluating gemma_sft l${L}..."
      python3 logic_inf/eval_gemma.py \
        --model   "$BASE_GEMMA" \
        --adapter "$SFT_ADAPTER" \
        --level   "$LEVEL_STR" \
        --output  "$OUTPUT" \
        --batch_size 8
      ;;

    gemma_grpo)
      if [ ! -d "$SFT_ADAPTER" ]; then
        log "SKIP l${L} — SFT adapter not found: $SFT_ADAPTER"
        continue
      fi
      if [ ! -d "$GRPO_ADAPTER" ]; then
        log "SKIP l${L} — GRPO adapter not found: $GRPO_ADAPTER"
        continue
      fi
      log "Evaluating gemma_grpo l${L}..."
      python3 logic_inf/eval_gemma.py \
        --model       "$BASE_GEMMA" \
        --sft_adapter "$SFT_ADAPTER" \
        --adapter     "$GRPO_ADAPTER" \
        --level       "$LEVEL_STR" \
        --output      "$OUTPUT" \
        --batch_size 8
      ;;
  esac

  if [ $? -eq 0 ]; then
    log "OK l${L} → $OUTPUT"
  else
    log "FAIL l${L}"
  fi
  echo "---"
done

# ── Summary table ─────────────────────────────────────────────────────────────
echo ""
echo "====== GSM8K Results — $SOURCE ======"
printf "%-8s %-12s %-10s %-12s %-12s\n" "Level" "Style" "Accuracy" "Think Tok" "Think Chr"
echo "--------------------------------------------------------------"
for L in "${LEVELS[@]}"; do
  FILE="$RESULTS_DIR/l${L}.json"
  [ -f "$FILE" ] || continue
  python3 -c "
import json
with open('$FILE') as f: r = json.load(f)
print(f\"{r['level']:<8} {r.get('style','?'):<12} {r['accuracy']*100:>8.1f}%  {r['mean_think_tokens']:>10.1f}  {r['mean_think_chars']:>10.1f}\")
"
done
echo "--------------------------------------------------------------"
echo "Full results in $RESULTS_DIR/"
