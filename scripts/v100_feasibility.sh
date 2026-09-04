#!/usr/bin/env bash
# V100 feasibility: download two small Qwen Instruct models and probe 100 AQUA samples each.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORK="${WORK_ROOT:-/root/autodl-tmp}"
export HF_HOME="${HF_HOME:-$WORK/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export MODEL_ROOT="${MODEL_ROOT:-$WORK/models}"
export MODEL_SOURCE="${MODEL_SOURCE:-modelscope}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$HF_HOME" "$MODEL_ROOT" "$ROOT/data/processed" "$ROOT/results"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
fi

PY="${PYTHON:-python}"
"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q -r "$ROOT/requirements-v100.txt"
"$PY" -m pip install -q -e "$ROOT"

"$PY" "$ROOT/scripts/prepare_data.py" --split test --sample 100 --seed 42 --output-name aqua_test_100
"$PY" "$ROOT/scripts/prepare_data.py" --split train --sample 100 --seed 42 --output-name aqua_train_100
"$PY" "$ROOT/scripts/download_models.py" --models 0.5b 1.5b --dest-root "$MODEL_ROOT"

run_one() {
  local name="$1"
  local layer="$2"
  local model_dir="$MODEL_ROOT/$name"
  echo "[probe] $name layer=$layer" >&2
  "$PY" -m option_mismatch.probe \
    --config "$ROOT/configs/qwen25_0.5b_2080.yaml" \
    --local-model-dir "$model_dir" \
    --probe-layer "$layer" \
    --test-jsonl "$ROOT/data/processed/aqua_test_100.jsonl" \
    --num-samples 100 \
    --max-new-tokens 128 \
    --output-json "$ROOT/results/${name}_h1_probe.json" \
    --output-npz "$ROOT/results/${name}_diligence_vectors.npz"
}

run_one "Qwen2.5-0.5B-Instruct" 16
run_one "Qwen2.5-1.5B-Instruct" 18

echo "[feasibility] done"
ls -lh "$MODEL_ROOT" "$ROOT/results"
