#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_UNIVER_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_UNIVER_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_UNIVER_OUT="$PF_UNIVER_ROOT/paper08_univer"
"$PF_UNIVER_PY" "$PF_UNIVER_ROOT/source/evaluate_univer.py" --root "$PF_UNIVER_OUT" --methods diffusion_univer_w16 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --validate --out "$PF_UNIVER_OUT/development_validation_03"
"$PF_UNIVER_PY" "$PF_UNIVER_ROOT/source/evaluate_univer.py" --root "$PF_UNIVER_OUT" --methods diffusion_univer_w16 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --out "$PF_UNIVER_OUT/development_online_01"
