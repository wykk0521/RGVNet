#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/wyk5/RGVNet/models/hbgnet_seed17_sr_v2
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="$ROOT:/root/wyk5/RGVNet/models/hbgnet/_deps:/root/wyk5/RGVNet/.python_packages:/root/wyk5/RGVNet"
export PYTHONDONTWRITEBYTECODE=1
exec /root/miniconda3/envs/py3.11torch2.0.1cu118/bin/python -B -u "$ROOT/launch_sr_v2_audited.py"
