#!/usr/bin/env bash
# 教師データの収集、ずれの追加収集、学習、閉ループ評価。
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3.13}"
if [[ ! -x .venv/bin/python ]]; then
  "$PY" -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"
.venv/bin/python -m physical_pointnet.sim.geometry
.venv/bin/python -m physical_pointnet.collect --episodes 48 --seed 0 --out data/demos.npz
ENCODER="${ENCODER:-pointnet2}"
RUN="runs/${ENCODER}"
.venv/bin/python -m physical_pointnet.train --data data/demos.npz --epochs 30 --encoder "$ENCODER" --out "$RUN"
.venv/bin/python -m physical_pointnet.collect --episodes 36 --seed 1 --ckpt "$RUN/best.pt" \
  --append data/demos.npz --out data/demos_dagger.npz
.venv/bin/python -m physical_pointnet.train --data data/demos_dagger.npz --epochs 20 --encoder "$ENCODER" --out "${RUN}_dagger"
.venv/bin/python -m physical_pointnet.collect --episodes 24 --seed 2 --ckpt "${RUN}_dagger/best.pt" \
  --append data/demos_dagger.npz --out data/demos_dagger2.npz
.venv/bin/python -m physical_pointnet.train --data data/demos_dagger2.npz --epochs 25 --encoder "$ENCODER" \
  --out "${RUN}_final" --omega-weight 3
.venv/bin/python -m physical_pointnet.evaluate --ckpt "${RUN}_final/best_closed.pt" --episodes 8 --seed 1000 \
  --gif "${RUN}_final/rollout.gif" --metrics "${RUN}_final/metrics.json"
