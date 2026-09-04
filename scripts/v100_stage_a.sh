#!/usr/bin/env bash
# Stage A: content-split H1 re-score + controls on AQUA-100 for 0.5B and 1.5B.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORK="${WORK_ROOT:-/root/autodl-tmp}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MODEL_ROOT="${MODEL_ROOT:-$WORK/models}"
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate base
fi
PY="${PYTHON:-python}"

run_one() {
  local name="$1"
  local layer="$2"
  local cfg="$3"
  "$PY" -m option_mismatch.probe --protocol \
    --config "$cfg" \
    --local-model-dir "$MODEL_ROOT/$name" \
    --probe-layer "$layer" \
    --test-jsonl "$ROOT/data/processed/aqua_test_100.jsonl" \
    --num-samples 100 \
    --max-new-tokens 256 \
    --output-json "$ROOT/reports/${name}_stage_a_h1.json" \
    --output-npz "$ROOT/results/${name}_stage_a_vectors.npz"
}

run_one "Qwen2.5-0.5B-Instruct" 16 "$ROOT/configs/qwen25_0.5b_2080.yaml"
run_one "Qwen2.5-1.5B-Instruct" 18 "$ROOT/configs/qwen25_1.5b_v100.yaml"
"$PY" "$ROOT/scripts/merge_stage_a.py"
echo "[stage-a] done"
