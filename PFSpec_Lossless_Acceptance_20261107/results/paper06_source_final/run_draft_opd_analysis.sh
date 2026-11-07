#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_OPD_OUT="$PF_RESEARCH_ROOT/paper02_draft_opd"
ln -s "$PF_PREVIOUS_ROOT/data" "$PF_OPD_OUT/data"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/summarize_paper.py" states --root "$PF_OPD_OUT"
"$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/offline_acceptance.py" --root "$PF_OPD_OUT" --data "$PF_OPD_OUT/development_states/data/val.pt" --selection "$PF_OPD_OUT/development_states/data/selection.json" --methods w1,fusion_long,opd_ce_tv,opd_split --batch-size 1 --out "$PF_OPD_OUT/development_offline_01"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/summarize_paper.py" report --root "$PF_OPD_OUT"
