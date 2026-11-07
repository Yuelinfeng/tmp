#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_DIFFUSION_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_DIFFUSION_PY=/root/miniconda3/envs/gaussiangpt/bin/python
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/collect_diffusion.py" prepare --out "$PF_DIFFUSION_ROOT/paper06_dflash"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/collect_diffusion.py" collect --out "$PF_DIFFUSION_ROOT/paper06_dflash" --validate
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/collect_diffusion.py" collect --out "$PF_DIFFUSION_ROOT/paper06_dflash"
