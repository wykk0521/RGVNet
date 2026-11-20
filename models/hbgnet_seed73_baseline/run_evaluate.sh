#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/wyk5/RGVNet/models/hbgnet_seed73_baseline
export PYTHONPATH="$ROOT/_deps:/root/wyk5/RGVNet/.python_packages:/root/wyk5/RGVNet"
export PYTHONDONTWRITEBYTECODE=1
exec /root/miniconda3/envs/py3.11torch2.0.1cu118/bin/python -B -u "$ROOT/evaluate.py" --config "$ROOT/config_protocol_b.yaml" "$@"
