#!/usr/bin/env bash
set -Eeuo pipefail

# Run the complete GPU workflow with live output, durable logs, automatic inference resume,
# checkpoint resume for interrupted training, periodic GPU status, and final result summaries.

MODE="${1:-}"
shift || true
IN_TMUX=0
case "${1:-}" in
  "") ;;
  --inside-tmux) IN_TMUX=1 ;;
  *) echo "Usage: bash $0 {smoke|full}" >&2; exit 2 ;;
esac
if [ "$MODE" != "smoke" ] && [ "$MODE" != "full" ]; then
  echo "Usage: bash $0 {smoke|full}" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_PATH="$REPO_ROOT/scripts/run_pipeline.sh"
DATA_ROOT="${TSAD_DATA_ROOT:-/root/autodl-tmp}"
SESSION_NAME="${TSAD_TMUX_SESSION:-tsad}"
RUN_ID="${TSAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

if [ "$IN_TMUX" -eq 0 ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is missing; run scripts/init_autodl.sh first" >&2
    exit 1
  fi
  WINDOW_NAME="${MODE}-${RUN_ID}"
  printf -v TMUX_COMMAND \
    'TSAD_RUN_ID=%q bash %q %q --inside-tmux; status=$?; echo; echo "[pipeline] finished with status $status"; echo "[pipeline] this tmux shell will stay open for inspection"; exec bash' \
    "$RUN_ID" "$SCRIPT_PATH" "$MODE"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-window -d -t "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  else
    tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  fi
  echo "[pipeline] attaching to tmux session '$SESSION_NAME'"
  exec tmux attach-session -t "$SESSION_NAME"
fi

cd "$REPO_ROOT"
if [ "$(uname -s)" != "Linux" ]; then
  echo "This script must run inside the AutoDL Linux SSH session, not on macOS." >&2
  exit 1
fi
if [ ! -f .venv/bin/activate ]; then
  echo "Missing .venv; run bash scripts/init_autodl.sh first." >&2
  exit 1
fi
# shellcheck disable=SC1091
source .venv/bin/activate

LOG_DIR="${TSAD_LOG_DIR:-$DATA_ROOT/logs/tsad-v2}"
LOG_FILE="$LOG_DIR/${MODE}_${RUN_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TOKENIZERS_PARALLELISM=false

CURRENT_STAGE="startup"
STAGE_STARTED=$SECONDS
MONITOR_PID=""

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

stage() {
  if [ "$CURRENT_STAGE" != "startup" ]; then
    log "DONE  $CURRENT_STAGE ($((SECONDS - STAGE_STARTED))s)"
  fi
  CURRENT_STAGE="$1"
  STAGE_STARTED=$SECONDS
  log "===== $CURRENT_STAGE ====="
}

latest_checkpoint() {
  local checkpoint_root="$1"
  if [ ! -d "$checkpoint_root" ]; then
    return 0
  fi
  find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    -print 2>/dev/null | sort -V | tail -n 1
}

start_gpu_monitor() {
  local interval="${TSAD_MONITOR_INTERVAL:-60}"
  (
    set +e
    while true; do
      printf '[%s] [gpu] ' "$(date '+%Y-%m-%d %H:%M:%S')"
      nvidia-smi \
        --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits | tr '\n' ';'
      printf '\n'
      sleep "$interval"
    done
  ) &
  MONITOR_PID=$!
  log "GPU monitor started: pid=$MONITOR_PID interval=${interval}s"
}

on_error() {
  local status=$?
  log "FAILED stage='$CURRENT_STAGE' line=$1 status=$status command=$2"
  log "Full log: $LOG_FILE"
  return "$status"
}

on_exit() {
  local status=$?
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  if [ "$status" -eq 0 ]; then
    log "pipeline_exit=success"
  else
    log "pipeline_exit=failure status=$status"
  fi
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap on_exit EXIT

run_sft() {
  local final_dir="$1"
  if [ -f "$final_dir/adapter_config.json" ]; then
    log "SFT final adapter already exists; skipping training: $final_dir"
    return
  fi
  local checkpoint
  checkpoint="$(latest_checkpoint "$(dirname "$final_dir")/checkpoints")"
  if [ -n "$checkpoint" ]; then
    log "resuming SFT from $checkpoint"
    python -m tsad_v2 "${CONFIG_ARGS[@]}" train-sft --resume "$checkpoint"
  else
    python -m tsad_v2 "${CONFIG_ARGS[@]}" train-sft
  fi
  test -f "$final_dir/adapter_config.json"
}

run_rl() {
  local sft_adapter="$1"
  local final_dir="$2"
  if [ -f "$final_dir/adapter_config.json" ]; then
    log "RL final adapter already exists; skipping training: $final_dir"
    return
  fi
  local checkpoint
  checkpoint="$(latest_checkpoint "$(dirname "$final_dir")/checkpoints")"
  if [ -n "$checkpoint" ]; then
    log "resuming RL from $checkpoint"
    python -m tsad_v2 "${CONFIG_ARGS[@]}" train-rl \
      --sft-adapter "$sft_adapter" --resume "$checkpoint"
  else
    python -m tsad_v2 "${CONFIG_ARGS[@]}" train-rl --sft-adapter "$sft_adapter"
  fi
  test -f "$final_dir/adapter_config.json"
}

if [ "$MODE" = "smoke" ]; then
  CONFIG_ARGS=(--config configs/base.yaml --config configs/smoke.yaml)
  UCR_DIR="data/processed/ucr_smoke"
  OUTPUT_ROOT="outputs/smoke"
else
  CONFIG_ARGS=(--config configs/base.yaml)
  UCR_DIR="data/processed/ucr"
  OUTPUT_ROOT="outputs"
fi

BASELINE_DIR="$OUTPUT_ROOT/baseline"
SFT_DIR="$OUTPUT_ROOT/sft"
SFT_EVAL_DIR="$OUTPUT_ROOT/sft_ucr"
RL_DIR="$OUTPUT_ROOT/rl"
RL_EVAL_DIR="$OUTPUT_ROOT/rl_ucr"
COMPARISON="$OUTPUT_ROOT/comparison.csv"

log "TSAD v2 pipeline mode=$MODE"
log "run_id=$RUN_ID commit=$(git rev-parse --short HEAD)"
log "log_file=$LOG_FILE"
log "user=$(whoami) host=$(hostname) cwd=$(pwd)"
nvidia-smi
start_gpu_monitor

if [ "$MODE" = "full" ]; then
  stage "1/8 prepare and validate full datasets"
  python -m tsad_v2 "${CONFIG_ARGS[@]}" generate-synthetic
  python -m tsad_v2 "${CONFIG_ARGS[@]}" \
    validate-manifest data/processed/synthetic/train.jsonl
  python -m tsad_v2 "${CONFIG_ARGS[@]}" \
    validate-manifest data/processed/synthetic/val.jsonl
  python -m tsad_v2 "${CONFIG_ARGS[@]}" prepare-ucr
else
  stage "1/8 verify smoke prerequisites"
  python -m tsad_v2 "${CONFIG_ARGS[@]}" \
    validate-manifest data/processed/synthetic_smoke/train.jsonl
  python -m tsad_v2 "${CONFIG_ARGS[@]}" \
    validate-manifest data/processed/synthetic_smoke/val.jsonl
fi
test -s "$UCR_DIR/series.jsonl"
test -s "$UCR_DIR/windows.jsonl"

stage "2/8 zero-shot baseline inference"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer \
  "$UCR_DIR/windows.jsonl" "$BASELINE_DIR/predictions.jsonl"

stage "3/8 zero-shot baseline evaluation"
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$BASELINE_DIR/predictions.jsonl" "$BASELINE_DIR"

stage "4/8 QLoRA SFT training"
run_sft "$SFT_DIR/final_adapter"

stage "5/8 SFT inference and evaluation"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer \
  "$UCR_DIR/windows.jsonl" "$SFT_EVAL_DIR/predictions.jsonl" \
  --adapter "$SFT_DIR/final_adapter"
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$SFT_EVAL_DIR/predictions.jsonl" "$SFT_EVAL_DIR"

stage "6/8 interval GRPO training"
run_rl "$SFT_DIR/final_adapter" "$RL_DIR/final_adapter"

stage "7/8 GRPO inference and evaluation"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer \
  "$UCR_DIR/windows.jsonl" "$RL_EVAL_DIR/predictions.jsonl" \
  --adapter "$RL_DIR/final_adapter"
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$RL_EVAL_DIR/predictions.jsonl" "$RL_EVAL_DIR"

stage "8/8 compare and print results"
python -m tsad_v2 compare \
  "$BASELINE_DIR/summary.json" \
  "$SFT_EVAL_DIR/summary.json" \
  "$RL_EVAL_DIR/summary.json" \
  --output "$COMPARISON"
log "baseline summary:"
sed -n '1,200p' "$BASELINE_DIR/summary.json"
log "SFT summary:"
sed -n '1,200p' "$SFT_EVAL_DIR/summary.json"
log "GRPO summary:"
sed -n '1,200p' "$RL_EVAL_DIR/summary.json"
log "comparison:"
sed -n '1,20p' "$COMPARISON"

log "DONE  $CURRENT_STAGE ($((SECONDS - STAGE_STARTED))s)"
log "===== pipeline complete ====="
log "mode=$MODE outputs=$OUTPUT_ROOT comparison=$COMPARISON"
log "log=$LOG_FILE"
