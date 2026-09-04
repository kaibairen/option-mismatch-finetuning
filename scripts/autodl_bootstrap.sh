#!/usr/bin/env bash
# Run inside an AutoDL RTX 2080 container after the repo is cloned.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export HF_HOME="${HF_HOME:-$ROOT/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
mkdir -p "$HF_HOME" "$ROOT/models" "$ROOT/results" "$ROOT/data/processed"

python3 -m pip install --upgrade pip
python3 -m pip install -r "$ROOT/requirements.txt"

python3 "$ROOT/scripts/prepare_data.py" --split test --limit 50

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[bootstrap] nvidia-smi missing; this is not a GPU boot" >&2
  exit 1
fi
nvidia-smi

python3 - <<'PY'
import torch
print("cuda", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available")
PY

echo "[bootstrap] environment ready"
echo "Next:"
echo "  python -m option_mismatch.probe --config configs/qwen25_0.5b_2080.yaml"
echo "  python -m option_mismatch.train_rep_dpo --config configs/qwen25_0.5b_2080.yaml"
echo "  python -m option_mismatch.eval_mcq --config configs/qwen25_0.5b_2080.yaml --adapter results/rep_dpo_lora"
