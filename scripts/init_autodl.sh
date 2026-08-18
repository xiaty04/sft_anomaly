#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare a fresh AutoDL instance through the point immediately before the GPU smoke run.
# The script is self-contained so it can be downloaded to /tmp and executed before cloning.

DATA_ROOT="${TSAD_DATA_ROOT:-/root/autodl-tmp}"
REPO_URL="${TSAD_REPO_URL:-https://github.com/tianyu-04/sft_anomaly.git}"
REPO_DIR="${TSAD_REPO_DIR:-$DATA_ROOT/sft_anomaly}"
SESSION_NAME="${TSAD_TMUX_SESSION:-tsad}"
UCR_URL="${TSAD_UCR_URL:-https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip}"
RUN_ID="${TSAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
IN_TMUX=0
NO_TMUX=0

usage() {
  echo "Usage: bash $0 [--no-tmux]" >&2
}

case "${1:-}" in
  "") ;;
  --inside-tmux) IN_TMUX=1 ;;
  --no-tmux) NO_TMUX=1 ;;
  *) usage; exit 2 ;;
esac

# Install tmux before relaunching so the remaining long run survives an SSH disconnect.
if [ "$IN_TMUX" -eq 0 ] && [ "$NO_TMUX" -eq 0 ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    echo "[bootstrap] installing tmux"
    apt-get update
    apt-get install -y tmux
  fi
  WINDOW_NAME="init-${RUN_ID}"
  printf -v TMUX_COMMAND \
    'TSAD_RUN_ID=%q bash %q --inside-tmux; status=$?; echo; echo "[init] finished with status $status"; echo "[init] this tmux shell will stay open for inspection"; exec bash' \
    "$RUN_ID" "$SCRIPT_PATH"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-window -d -t "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  else
    tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  fi
  echo "[bootstrap] attaching to tmux session '$SESSION_NAME'"
  exec tmux attach-session -t "$SESSION_NAME"
fi

LOG_DIR="${TSAD_LOG_DIR:-$DATA_ROOT/logs/tsad-v2}"
LOG_FILE="$LOG_DIR/init_${RUN_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
export TRANSFORMERS_VERBOSITY="${TRANSFORMERS_VERBOSITY:-info}"
export HF_HUB_DISABLE_PROGRESS_BARS=0
export TOKENIZERS_PARALLELISM=false

CURRENT_STAGE="bootstrap"
STAGE_STARTED=$SECONDS

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

fail() {
  log "FAILED stage='$CURRENT_STAGE' message=$*"
  log "Full log: $LOG_FILE"
  exit 1
}

stage() {
  if [ "$CURRENT_STAGE" != "bootstrap" ]; then
    log "DONE  $CURRENT_STAGE ($((SECONDS - STAGE_STARTED))s)"
  fi
  CURRENT_STAGE="$1"
  STAGE_STARTED=$SECONDS
  log "===== $CURRENT_STAGE ====="
}

on_error() {
  local status=$?
  log "FAILED stage='$CURRENT_STAGE' line=$1 status=$status command=$2"
  log "Full log: $LOG_FILE"
  return "$status"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

log "TSAD v2 AutoDL initialization"
log "run_id=$RUN_ID"
log "log_file=$LOG_FILE"
log "user=$(whoami) host=$(hostname) cwd=$(pwd)"
log "data_root=$DATA_ROOT repo_dir=$REPO_DIR"

stage "1/8 verify AutoDL context and storage"
if [ "$(uname -s)" != "Linux" ]; then
  fail "This script must run inside the AutoDL Linux SSH session, not on macOS."
fi
mkdir -p "$DATA_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
df -h "$DATA_ROOT"
nvidia-smi || fail "nvidia-smi failed; confirm the AutoDL GPU instance is running"

stage "2/8 clone or fast-forward the repository"
if [ ! -d "$REPO_DIR/.git" ]; then
  log "cloning $REPO_URL"
  git clone --progress "$REPO_URL" "$REPO_DIR"
else
  log "repository exists; checking for local source changes"
  if ! git -C "$REPO_DIR" diff --quiet || ! git -C "$REPO_DIR" diff --cached --quiet; then
    git -C "$REPO_DIR" status --short
    fail "repository has tracked local changes; refusing to overwrite them"
  fi
  git -C "$REPO_DIR" pull --ff-only --progress
fi
cd "$REPO_DIR"
log "commit=$(git rev-parse --short HEAD) branch=$(git branch --show-current)"

stage "3/8 create the environment and install dependencies"
if [ ! -x .venv/bin/python ]; then
  python3 -m venv --system-site-packages .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'

stage "4/8 verify Python, CUDA, and the CLI"
python - <<'PY'
import bitsandbytes
import peft
import torch
import transformers
import trl

print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("peft=", peft.__version__)
print("trl=", trl.__version__)
print("bitsandbytes=", bitsandbytes.__version__)
print("cuda=", torch.version.cuda)
print("cuda_available=", torch.cuda.is_available())
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY
python -m tsad_v2 --help

stage "5/8 run CPU protocol and source checks"
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests

stage "6/8 generate and validate the smoke synthetic dataset"
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml generate-synthetic
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml \
  validate-manifest data/processed/synthetic_smoke/train.jsonl
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml \
  validate-manifest data/processed/synthetic_smoke/val.jsonl

stage "7/8 download and collect the official UCR test data"
mkdir -p data/downloads/ucr/extracted data/raw/ucr
UCR_ZIP="data/downloads/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"
if [ -s "$UCR_ZIP" ]; then
  log "reusing $UCR_ZIP"
else
  log "downloading $UCR_URL"
  curl -fL --retry 3 --connect-timeout 30 --progress-bar "$UCR_URL" -o "$UCR_ZIP"
fi
python -m zipfile -t "$UCR_ZIP"
python -m zipfile -e "$UCR_ZIP" data/downloads/ucr/extracted
find data/downloads/ucr/extracted -type f -name '*_UCR_Anomaly_*.txt' \
  -exec cp -n {} data/raw/ucr/ \;
UCR_COUNT="$(find data/raw/ucr -maxdepth 1 -type f -name '*.txt' | wc -l | tr -d ' ')"
log "collected_ucr_files=$UCR_COUNT"
if [ "$UCR_COUNT" -eq 0 ]; then
  fail "no UCR files were extracted"
fi

stage "8/8 render the bounded UCR smoke dataset"
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml prepare-ucr
test -s data/processed/ucr_smoke/series.jsonl
test -s data/processed/ucr_smoke/windows.jsonl
SMOKE_IMAGES="$(find data/processed/ucr_smoke/images -type f -name '*.png' | wc -l | tr -d ' ')"
log "smoke_images=$SMOKE_IMAGES"

log "DONE  $CURRENT_STAGE ($((SECONDS - STAGE_STARTED))s)"
log "===== initialization complete ====="
log "repo=$REPO_DIR"
log "venv=$REPO_DIR/.venv"
log "log=$LOG_FILE"
log "next=bash scripts/run_pipeline.sh smoke"
