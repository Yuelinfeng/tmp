#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
/root/miniconda3/envs/gaussiangpt/bin/python /root/autodl-tmp/pfspec_lossless_acceptance_20261106/source/diagnose_univer_numeric.py
