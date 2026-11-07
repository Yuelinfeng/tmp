#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_MULTI_OUT="$PF_RESEARCH_ROOT/paper04_optimal_multi_draft"
mkdir "$PF_MULTI_OUT"
mkdir "$PF_MULTI_OUT/training"
ln -s "$PF_PREVIOUS_ROOT/data" "$PF_MULTI_OUT/data"
ln -s "$PF_PREVIOUS_ROOT/training/fusion_long" "$PF_MULTI_OUT/training/fusion_long"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/validate_greedy_coupling.py" "$PF_MULTI_OUT/coupling_numeric_validation.json"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/offline_multidraft.py" --root "$PF_MULTI_OUT" --out "$PF_MULTI_OUT/development_curve_01"
