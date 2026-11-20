#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/wyk5/RGVNet/models/hbgnet_fhapd_js_baseline_seed73
export CUDA_VISIBLE_DEVICES=2
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$ROOT:$ROOT/_deps:/root/wyk5/RGVNet/.python_packages"
exec /root/miniconda3/envs/py3.11torch2.0.1cu118/bin/python -B -u "$ROOT/evaluate.py" "$@"
