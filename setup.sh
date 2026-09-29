#!/usr/bin/env bash
# One-time environment setup: Python 3.12 venv + all dependencies.
set -euo pipefail
cd "$(dirname "$0")"

PY=${PYTHON:-$(command -v python3.12 || echo /opt/homebrew/bin/python3.12)}
if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null; then
  brew list python@3.12 >/dev/null 2>&1 || brew install python@3.12
  brew list libomp >/dev/null 2>&1 || brew install libomp   # OpenMP runtime for XGBoost
fi

[[ -d .venv ]] || "$PY" -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# macOS: XGBoost links Homebrew's libomp while PyTorch ships its own copy. Two OpenMP
# runtimes in one process segfault/deadlock, so point PyTorch at the Homebrew one.
TORCH_OMP=$(ls .venv/lib/python*/site-packages/torch/lib/libomp.dylib 2>/dev/null || true)
BREW_OMP=/opt/homebrew/opt/libomp/lib/libomp.dylib
if [[ -n "$TORCH_OMP" && -f "$BREW_OMP" && ! -L "$TORCH_OMP" ]]; then
  mv "$TORCH_OMP" "$TORCH_OMP.bundled"
  ln -s "$BREW_OMP" "$TORCH_OMP"
  echo "Linked PyTorch libomp -> $BREW_OMP"
fi

.venv/bin/python -c "import torch, xgboost, sklearn, shap, streamlit; print('Environment OK')"
