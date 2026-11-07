#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_OPD_OUT="$PF_RESEARCH_ROOT/paper02_draft_opd"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/train_feedback.py" --feedback "$PF_OPD_OUT" --out "$PF_OPD_OUT" --name opd_forward --mode forward_kl
mkdir "$PF_OPD_OUT/training/opd_split_last"
ln -s "$PF_OPD_OUT/training/opd_split/last.pt" "$PF_OPD_OUT/training/opd_split_last/selected.pt"
"$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/evaluate.py" --root "$PF_OPD_OUT" --methods opd_forward,opd_split_last --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_OPD_OUT/development_validation_02"
"$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/evaluate.py" --root "$PF_OPD_OUT" --methods opd_forward,opd_split_last --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_OPD_OUT/development_online_02"
ln -s "$PF_PREVIOUS_ROOT/data" "$PF_OPD_OUT/data"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/summarize_paper.py" states --root "$PF_OPD_OUT"
"$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/offline_acceptance.py" --root "$PF_OPD_OUT" --data "$PF_OPD_OUT/development_states/data/val.pt" --selection "$PF_OPD_OUT/development_states/data/selection.json" --methods w1,fusion_long,opd_ce_tv,opd_split,opd_forward,opd_split_last --batch-size 1 --out "$PF_OPD_OUT/development_offline_01"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/summarize_paper.py" report --root "$PF_OPD_OUT" --additional-records "$PF_OPD_OUT/development_online_02/records.jsonl" --candidate-methods opd_ce_tv,opd_split,opd_forward,opd_split_last
