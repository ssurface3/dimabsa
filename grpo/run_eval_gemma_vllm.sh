#!/bin/bash
# ============================================================
# Fast GSM8K eval for Gemma using vLLM (run from vllm_env).
#
# Sources:
#   gemma_sft   — base + SFT adapter
#   gemma_grpo  — base + SFT merged + GRPO adapter
#
# The merged model is cached per level so the slow merge only
# runs once. Delete merged_vllm/ inside the checkpoint dir to force re-merge.
#
# Usage:
#   bash run_eval_gemma_vllm.sh [SOURCE] [--levels 1 2 3]
#
# Examples:
#   bash run_eval_gemma_vllm.sh gemma_sft
#   bash run_eval_gemma_vllm.sh gemma_grpo --levels 1 2
# ============================================================

SOURCE="${1:-gemma_sft}"

set -o pipefail

PROJECT_DIR="/home/jovyan/.mlspace/envs/student_env/grpo"
SFT_CKPT_ROOT="$PROJECT_DIR/checkpoints_gemma_sft"
GRPO_CKPT_ROOT="$PROJECT_DIR/checkpoints_gemma_grpo"

# ── Parse flags ───────────────────────────────────────────────────────────────
LEVELS=()
PARSE_LEVELS=0
for arg in "${@:2}"; do
  case "$arg" in
    --levels) PARSE_LEVELS=1 ;;
    *) [ "$PARSE_LEVELS" -eq 1 ] && LEVELS+=("$arg") ;;
  esac
done
[ ${#LEVELS[@]} -eq 0 ] && LEVELS=(1 2 3 4 5)

# ── Environment ───────────────────────────────────────────────────────────────
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  set +u; conda activate vllm_env 2>/dev/null || true; set -u
fi

cd "$PROJECT_DIR"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

LEVEL_NAMES=([1]="level 1" [2]="level 2" [3]="level 3" [4]="level 4" [5]="level 5")

case "$SOURCE" in
  gemma_sft)  RESULTS_DIR="$PROJECT_DIR/eval_results_gemma_sft"  ;;
  gemma_grpo) RESULTS_DIR="$PROJECT_DIR/eval_results_gemma_grpo" ;;
  *)
    echo "Unknown source: $SOURCE. Options: gemma_sft  gemma_grpo"
    exit 1 ;;
esac

mkdir -p "$RESULTS_DIR"
log "Source: $SOURCE  |  Levels: ${LEVELS[*]}  |  Results: $RESULTS_DIR"
echo ""

# Resolve adapter: prefer final_adapter/, fall back to latest checkpoint-N/
resolve_adapter() {
  local root="$1"
  [ -d "$root/final_adapter" ] && { echo "$root/final_adapter"; return; }
  local latest
  latest=$(ls -d "$root"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
  [ -d "$latest" ] && echo "$latest" || echo ""
}

# ── Main loop ─────────────────────────────────────────────────────────────────
for L in "${LEVELS[@]}"; do
  LEVEL_STR="${LEVEL_NAMES[$L]}"
  OUTPUT="$RESULTS_DIR/l${L}.json"
  SFT_ADAPTER=$(resolve_adapter "$SFT_CKPT_ROOT/level_${L}")
  GRPO_ADAPTER=$(resolve_adapter "$GRPO_CKPT_ROOT/level_${L}")

  if [ -f "$OUTPUT" ]; then
    log "SKIP l${L} — results exist at $OUTPUT (delete to rerun)"
    continue
  fi

  case "$SOURCE" in
    gemma_sft)
      if [ -z "$SFT_ADAPTER" ]; then
        log "SKIP l${L} — no SFT adapter in $SFT_CKPT_ROOT/level_${L}/"
        continue
      fi
      log "Evaluating gemma_sft l${L} via vLLM (adapter: $(basename $SFT_ADAPTER))..."
      python3 logic_inf/eval_gemma_vllm.py \
        --adapter "$SFT_ADAPTER" \
        --level   "$LEVEL_STR" \
        --output  "$OUTPUT"
      ;;

    gemma_grpo)
      if [ -z "$SFT_ADAPTER" ]; then
        log "SKIP l${L} — no SFT adapter in $SFT_CKPT_ROOT/level_${L}/"
        continue
      fi
      if [ -z "$GRPO_ADAPTER" ]; then
        log "SKIP l${L} — no GRPO adapter in $GRPO_CKPT_ROOT/level_${L}/"
        continue
      fi
      log "Evaluating gemma_grpo l${L} via vLLM (grpo: $(basename $GRPO_ADAPTER))..."
      python3 logic_inf/eval_gemma_vllm.py \
        --sft_adapter "$SFT_ADAPTER" \
        --adapter     "$GRPO_ADAPTER" \
        --level       "$LEVEL_STR" \
        --output      "$OUTPUT"
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
echo "====== GSM8K Results — $SOURCE (vLLM) ======"
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
