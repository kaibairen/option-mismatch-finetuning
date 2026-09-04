#!/usr/bin/env bash
# Idempotent Cloud Agent bootstrap: CPU wheels + AQUA-RAT probe split.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m pip install --upgrade pip
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
  echo "[install] CUDA GPU detected; installing GPU requirements"
  python3 -m pip install -r "$ROOT/requirements.txt"
else
  echo "[install] No GPU; installing CPU requirements"
  python3 -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
  python3 -m pip install -r "$ROOT/requirements-cpu.txt"
fi

python3 -m pip install -e "$ROOT"
python3 "$ROOT/scripts/prepare_data.py" --split test --limit 50
python3 "$ROOT/scripts/prepare_data.py" --split train --limit 256
python3 -m pytest "$ROOT/tests" -q
python3 -c "import torch, transformers, scipy, option_mismatch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "[install] ok"
