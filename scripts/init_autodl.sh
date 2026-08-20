#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare a fresh AutoDL instance through bounded one-series-one-sample smoke inputs.

DATA_ROOT="${TSAD_DATA_ROOT:-/root/autodl-tmp}"
REPO_URL="${TSAD_REPO_URL:-https://github.com/xiaty04/sft_anomaly.git}"
REPO_DIR="${TSAD_REPO_DIR:-$DATA_ROOT/sft_anomaly}"
SESSION_NAME="${TSAD_TMUX_SESSION:-tsad}"
UCR_URL="${TSAD_UCR_URL:-https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCR_TimeSeriesAnomalyDatasets2021.zip}"
RUN_ID="${TSAD_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
SCRIPT_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
IN_TMUX=0
NO_TMUX=0
RESUME_MODE=""

usage() {
  echo "Usage: bash $0 [--no-tmux] [--resume {smoke|full}]" >&2
  exit 2
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --inside-tmux)
      IN_TMUX=1
      shift
      ;;
    --no-tmux)
      NO_TMUX=1
      shift
      ;;
    --resume)
      [ "$#" -ge 2 ] || usage
      RESUME_MODE="$2"
      [ "$RESUME_MODE" = "smoke" ] || [ "$RESUME_MODE" = "full" ] || usage
      shift 2
      ;;
    *)
      usage
      ;;
  esac
done

if [ "$IN_TMUX" -eq 0 ] && [ "$NO_TMUX" -eq 0 ] && [ -z "${TMUX:-}" ]; then
  if ! command -v tmux >/dev/null 2>&1; then
    apt-get update
    apt-get install -y tmux
  fi
  WINDOW_NAME="init-${RUN_ID}"
  if [ -n "$RESUME_MODE" ]; then
    WINDOW_NAME="resume-${RESUME_MODE}-${RUN_ID}"
    printf -v TMUX_COMMAND \
      'TSAD_RUN_ID=%q bash %q --inside-tmux --resume %q; status=$?; echo; echo "[init] status=$status"; exec bash' \
      "$RUN_ID" "$SCRIPT_PATH" "$RESUME_MODE"
  else
    printf -v TMUX_COMMAND \
      'TSAD_RUN_ID=%q bash %q --inside-tmux; status=$?; echo; echo "[init] status=$status"; exec bash' \
      "$RUN_ID" "$SCRIPT_PATH"
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux new-window -d -t "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  else
    tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" "$TMUX_COMMAND"
  fi
  exec tmux attach-session -t "$SESSION_NAME"
fi

LOG_DIR="${TSAD_LOG_DIR:-$DATA_ROOT/logs/tsad-v2}"
LOG_FILE="$LOG_DIR/init_${RUN_ID}.log"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_FILE") 2>&1

export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-$DATA_ROOT/cache/huggingface}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$DATA_ROOT/cache/pip}"
export TMPDIR="${TMPDIR:-$DATA_ROOT/tmp}"
export TOKENIZERS_PARALLELISM=false

CURRENT_STAGE="bootstrap"
STAGE_STARTED=$SECONDS
log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"; }
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
  log "FAILED stage='$CURRENT_STAGE' line=$1 status=$status command=$2 log=$LOG_FILE"
  return "$status"
}
trap 'on_error "$LINENO" "$BASH_COMMAND"' ERR

stage "1/7 verify AutoDL context"
[ "$(uname -s)" = "Linux" ] || { echo "Run inside AutoDL Linux." >&2; exit 1; }
mkdir -p "$DATA_ROOT" "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
nvidia-smi
df -h "$DATA_ROOT"

stage "2/7 clone or fast-forward code"
if [ ! -d "$REPO_DIR/.git" ]; then
  git clone --progress "$REPO_URL" "$REPO_DIR"
else
  if ! git -C "$REPO_DIR" diff --quiet || ! git -C "$REPO_DIR" diff --cached --quiet; then
    git -C "$REPO_DIR" status --short
    echo "Tracked source changes exist; refusing to pull." >&2
    exit 1
  fi
  git -C "$REPO_DIR" pull --ff-only --progress
fi
cd "$REPO_DIR"
log "commit=$(git rev-parse --short HEAD)"

stage "3/7 create environment"
[ -x .venv/bin/python ] || python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[train,dev]'

stage "4/7 verify runtime and source"
python - <<'PY'
import bitsandbytes
import peft
import sklearn
import torch
import transformers

print("torch=", torch.__version__)
print("transformers=", transformers.__version__)
print("peft=", peft.__version__)
print("bitsandbytes=", bitsandbytes.__version__)
print("sklearn=", sklearn.__version__)
print("cuda=", torch.version.cuda)
print("gpu=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
PY
python -m tsad_v2 --help
python -m unittest discover -s tests -v
python -m compileall -q src tests
ruff check src tests

stage "5/7 download official UCR archive"
mkdir -p data/downloads/ucr/extracted data/raw/ucr
UCR_ZIP="data/downloads/ucr/UCR_TimeSeriesAnomalyDatasets2021.zip"
if [ ! -s "$UCR_ZIP" ]; then
  curl -fL --retry 3 --connect-timeout 30 --progress-bar "$UCR_URL" -o "$UCR_ZIP"
fi
python -m zipfile -t "$UCR_ZIP"
python -m zipfile -e "$UCR_ZIP" data/downloads/ucr/extracted
find data/downloads/ucr/extracted -type f -name '*_UCR_Anomaly_*.txt' \
  -exec cp -n {} data/raw/ucr/ \;
UCR_COUNT="$(find data/raw/ucr -maxdepth 1 -type f -name '*.txt' | wc -l | tr -d ' ')"
[ "$UCR_COUNT" -gt 0 ] || { echo "No UCR files extracted." >&2; exit 1; }
log "ucr_files=$UCR_COUNT"

stage "6/7 prepare four whole-series paired UCR smoke inputs"
python -m tsad_v2 --config configs/base.yaml --config configs/smoke.yaml prepare-ucr
test -s data/processed/ucr_smoke/series.jsonl
UCR_SMOKE_SAMPLES="$(wc -l < data/processed/ucr_smoke/series.jsonl | tr -d ' ')"
[ "$UCR_SMOKE_SAMPLES" -eq 4 ] || {
  echo "Expected 4 UCR smoke series samples, found $UCR_SMOKE_SAMPLES." >&2
  exit 1
}
log "ucr_smoke_series_samples=$UCR_SMOKE_SAMPLES"

stage "7/7 finish"
log "repo=$REPO_DIR venv=$REPO_DIR/.venv log=$LOG_FILE"
if [ -n "$RESUME_MODE" ]; then
  log "resume_mode=$RESUME_MODE launching=scripts/run_pipeline.sh"
  TSAD_RUN_ID="$RUN_ID" bash scripts/run_pipeline.sh "$RESUME_MODE"
else
  log "next=bash scripts/run_pipeline.sh smoke"
fi
