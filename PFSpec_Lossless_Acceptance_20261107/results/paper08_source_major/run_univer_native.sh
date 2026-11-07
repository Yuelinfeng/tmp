#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_UNIVER_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
mkdir -p "$PF_UNIVER_ROOT/paper08_univer"
/root/miniconda3/envs/gaussiangpt/bin/python "$PF_UNIVER_ROOT/source/validate_univer_branches.py" --width 16 --eager-only --out "$PF_UNIVER_ROOT/paper08_univer/native_branches_01.json"
