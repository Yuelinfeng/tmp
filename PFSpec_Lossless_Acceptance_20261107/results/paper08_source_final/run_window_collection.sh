#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_WINDOW_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_WINDOW_PY=/root/miniconda3/envs/gaussiangpt/bin/python
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/collect_window.py" prepare --out "$PF_WINDOW_ROOT/paper07_aat_window"
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/collect_window.py" collect --out "$PF_WINDOW_ROOT/paper07_aat_window" --validate
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/collect_window.py" collect --out "$PF_WINDOW_ROOT/paper07_aat_window"
