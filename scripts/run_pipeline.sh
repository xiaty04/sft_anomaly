#!/usr/bin/env bash
set -Eeuo pipefail

# Run C0/T0/V0, generate paired UCR-derived synthetic data, then run T1/V1 SFT.

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
CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ "$IN_TMUX" -eq 0 ] && [ -z "${TMUX:-}" ]; then
  command -v tmux >/dev/null 2>&1 || {
    echo "tmux is missing; run scripts/init_autodl.sh first" >&2
    exit 1
  }
  WINDOW_NAME="${MODE}-${RUN_ID}"
  TMUX_FORMAT='TSAD_RUN_ID=%q PYTORCH_CUDA_ALLOC_CONF=%q bash %q %q --inside-tmux; '
  TMUX_FORMAT+='status=$?; echo; echo "[pipeline] status=$status"; exec bash'
  printf -v TMUX_COMMAND "$TMUX_FORMAT" \
    "$RUN_ID" "$CUDA_ALLOC_CONF" "$SCRIPT_PATH" "$MODE"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-window -d -t "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  else
    tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  fi
  exec tmux attach-session -t "$SESSION_NAME"
fi

cd "$REPO_ROOT"
if [ "$(uname -s)" != "Linux" ]; then
  echo "Run this script inside the AutoDL Linux SSH session." >&2
  exit 1
fi
test -f .venv/bin/activate || {
  echo "Missing .venv; run scripts/init_autodl.sh first." >&2
  exit 1
}
# shellcheck disable=SC1091
source .venv/bin/activate

LOG_DIR="${TSAD_LOG_DIR:-$DATA_ROOT/logs/tsad-v2}"
LOG_FILE="$LOG_DIR/${MODE}_${RUN_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="$CUDA_ALLOC_CONF"

CURRENT_STAGE="startup"
STAGE_STARTED=$SECONDS
MONITOR_PID=""

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }

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
  [ -d "$checkpoint_root" ] || return 0
  find "$checkpoint_root" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
    -print 2>/dev/null | sort -V | tail -n 1
}

run_sft() {
  local modality="$1"
  local final_dir="$2"
  if [ -f "$final_dir/adapter_config.json" ]; then
    log "$modality SFT adapter exists; skipping: $final_dir"
    return
  fi
  local checkpoint
  checkpoint="$(latest_checkpoint "$(dirname "$final_dir")/checkpoints")"
  if [ -n "$checkpoint" ]; then
    python -m tsad_v2 "${CONFIG_ARGS[@]}" train-sft \
      --modality "$modality" --resume "$checkpoint"
  else
    python -m tsad_v2 "${CONFIG_ARGS[@]}" train-sft --modality "$modality"
  fi
  test -f "$final_dir/adapter_config.json"
}

on_error() {
  local status=$?
  log "FAILED stage='$CURRENT_STAGE' line=$1 status=$status command=$2"
  log "log=$LOG_FILE"
  return "$status"
}

on_exit() {
  local status=$?
  if [ -n "$MONITOR_PID" ]; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
  log "pipeline_exit=$status"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR
trap on_exit EXIT

(
  set +e
  while true; do
    printf '[%s] [gpu] ' "$(date '+%Y-%m-%d %H:%M:%S')"
    nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total \
      --format=csv,noheader,nounits | tr '\n' ';'
    printf '\n'
    sleep "${TSAD_MONITOR_INTERVAL:-60}"
  done
) &
MONITOR_PID=$!

if [ "$MODE" = "smoke" ]; then
  CONFIG_ARGS=(--config configs/base.yaml --config configs/smoke.yaml)
  UCR_DIR="data/processed/ucr_smoke"
  SYNTHETIC_DIR="data/processed/synthetic_smoke"
  OUTPUT_ROOT="outputs/smoke"
  EXPECTED_UCR_SAMPLES=4
else
  CONFIG_ARGS=(--config configs/base.yaml)
  UCR_DIR="data/processed/ucr"
  SYNTHETIC_DIR="data/processed/synthetic"
  OUTPUT_ROOT="outputs"
  EXPECTED_UCR_SAMPLES=250
fi

EVAL_ROOT="$OUTPUT_ROOT/series_level"
C0_DIR="$EVAL_ROOT/c0_isolation_forest"
T0_DIR="$EVAL_ROOT/t0_text"
V0_DIR="$EVAL_ROOT/v0_vision"
SFT_ROOT="$OUTPUT_ROOT/sft"
T1_DIR="$EVAL_ROOT/t1_text"
V1_DIR="$EVAL_ROOT/v1_vision"
COMPARISON="$EVAL_ROOT/comparison.csv"

log "TSAD v2 baseline-to-SFT pipeline mode=$MODE"
log "run_id=$RUN_ID commit=$(git rev-parse --short HEAD) log=$LOG_FILE"
log "pytorch_cuda_alloc_conf=$PYTORCH_CUDA_ALLOC_CONF"
nvidia-smi

stage "1/10 prepare one sample per UCR series"
python -m tsad_v2 "${CONFIG_ARGS[@]}" prepare-ucr
test -s "$UCR_DIR/series.jsonl"
UCR_SAMPLES="$(wc -l < "$UCR_DIR/series.jsonl" | tr -d ' ')"
[ "$UCR_SAMPLES" -eq "$EXPECTED_UCR_SAMPLES" ] || {
  echo "Expected $EXPECTED_UCR_SAMPLES UCR series samples, found $UCR_SAMPLES." >&2
  exit 1
}
log "ucr_series_samples=$UCR_SAMPLES inference_calls_per_method=$UCR_SAMPLES"

stage "2/10 C0 Isolation Forest"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer-isolation-forest \
  "$UCR_DIR/series.jsonl" "$C0_DIR/predictions.jsonl"
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$C0_DIR/predictions.jsonl" "$C0_DIR"

stage "3/10 T0 text zero-shot"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer \
  "$UCR_DIR/series.jsonl" "$T0_DIR/predictions.jsonl" --modality text
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$T0_DIR/predictions.jsonl" "$T0_DIR"

stage "4/10 V0 vision zero-shot"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer \
  "$UCR_DIR/series.jsonl" "$V0_DIR/predictions.jsonl" --modality vision
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$V0_DIR/predictions.jsonl" "$V0_DIR"

stage "5/10 generate paired UCR-derived synthetic data"
python -m tsad_v2 "${CONFIG_ARGS[@]}" generate-synthetic
python -m tsad_v2 "${CONFIG_ARGS[@]}" validate-manifest "$SYNTHETIC_DIR/train.jsonl"
python -m tsad_v2 "${CONFIG_ARGS[@]}" validate-manifest "$SYNTHETIC_DIR/val.jsonl"

stage "6/10 T-SFT training"
run_sft text "$SFT_ROOT/text/final_adapter"

stage "7/10 T1 text SFT evaluation"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer \
  "$UCR_DIR/series.jsonl" "$T1_DIR/predictions.jsonl" --modality text \
  --adapter "$SFT_ROOT/text/final_adapter"
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$T1_DIR/predictions.jsonl" "$T1_DIR"

stage "8/10 V-SFT training"
run_sft vision "$SFT_ROOT/vision/final_adapter"

stage "9/10 V1 vision SFT evaluation"
python -m tsad_v2 "${CONFIG_ARGS[@]}" infer \
  "$UCR_DIR/series.jsonl" "$V1_DIR/predictions.jsonl" --modality vision \
  --adapter "$SFT_ROOT/vision/final_adapter"
python -m tsad_v2 "${CONFIG_ARGS[@]}" evaluate \
  "$UCR_DIR/series.jsonl" "$V1_DIR/predictions.jsonl" "$V1_DIR"

stage "10/10 compare and package metrics"
python -m tsad_v2 compare \
  "$C0_DIR/summary.json" "$T0_DIR/summary.json" "$V0_DIR/summary.json" \
  "$T1_DIR/summary.json" "$V1_DIR/summary.json" --output "$COMPARISON"
METRICS_DIR="$DATA_ROOT/results/tsad-v2/${MODE}_${RUN_ID}"
mkdir -p "$METRICS_DIR"
cp "$COMPARISON" "$METRICS_DIR/"
for method_dir in "$C0_DIR" "$T0_DIR" "$V0_DIR" "$T1_DIR" "$V1_DIR"; do
  method="$(basename "$method_dir")"
  mkdir -p "$METRICS_DIR/$method"
  cp "$method_dir/summary.json" "$method_dir/per_sample.jsonl" "$METRICS_DIR/$method/"
done
tar -czf "$METRICS_DIR.tar.gz" -C "$(dirname "$METRICS_DIR")" "$(basename "$METRICS_DIR")"

log "DONE  $CURRENT_STAGE ($((SECONDS - STAGE_STARTED))s)"
log "pipeline complete outputs=$OUTPUT_ROOT metrics=$METRICS_DIR.tar.gz"
sed -n '1,20p' "$COMPARISON"
