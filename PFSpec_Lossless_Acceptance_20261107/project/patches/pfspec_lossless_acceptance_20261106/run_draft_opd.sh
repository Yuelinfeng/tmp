#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_OPD_OUT="$PF_RESEARCH_ROOT/paper02_draft_opd"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/verification_feedback.py" collect
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/train_feedback.py" --feedback "$PF_OPD_OUT" --out "$PF_OPD_OUT" --name opd_ce_tv --mode ce_tv
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/train_feedback.py" --feedback "$PF_OPD_OUT" --out "$PF_OPD_OUT" --name opd_split --mode split_kl
ln -s "$PF_PREVIOUS_ROOT/training/fusion_long" "$PF_OPD_OUT/training/fusion_long"
"$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/evaluate.py" --root "$PF_OPD_OUT" --methods opd_ce_tv,opd_split --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_OPD_OUT/development_validation_01"
"$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/evaluate.py" --root "$PF_OPD_OUT" --methods w1,fusion_long,opd_ce_tv,opd_split --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_OPD_OUT/development_online_01"
