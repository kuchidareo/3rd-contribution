#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Keep Kaggle's CUDA-compatible torch installation. No data is downloaded here.
python - <<'PY'
import sys
import torch
print('Python:', sys.executable)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('Enable a Kaggle GPU accelerator first.')
PY
python -m pip install -r requirements-baseline.txt
python -m pip freeze > "${NCM_ENV_LOCK:-/kaggle/working/ncm-environment.txt}"
printf '%s\n' 'Setup complete. Dataset/checkpoint download is a separate explicit CLI step.'
