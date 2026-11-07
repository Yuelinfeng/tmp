#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_SECOND_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_SECOND_PY=/root/miniconda3/envs/gaussiangpt/bin/python
"$PF_SECOND_PY" "$PF_SECOND_ROOT/source/offline_diffusion_second.py" --limit-per-source 2 --out "$PF_SECOND_ROOT/paper06_dflash/development_second_probe_02"
"$PF_SECOND_PY" "$PF_SECOND_ROOT/source/offline_diffusion_second.py"
