#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_MULTI_OUT="$PF_RESEARCH_ROOT/paper04_optimal_multi_draft"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/validate_multi_branches.py" --width 16 --out "$PF_MULTI_OUT/branch_validation_02.json"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_multidraft.py" --root "$PF_MULTI_OUT" --methods fusion_long,multi_greedy_w16 --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_MULTI_OUT/development_validation_01"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_multidraft.py" --root "$PF_MULTI_OUT" --methods multi_greedy_w16 --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_MULTI_OUT/development_online_01"
