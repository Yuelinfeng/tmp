#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_OPD_OUT="$PF_RESEARCH_ROOT/paper02_draft_opd"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/train_feedback.py" --feedback "$PF_OPD_OUT" --out "$PF_OPD_OUT" --name opd_fullsplit --mode split_full_kl
mkdir "$PF_OPD_OUT/training/opd_fullsplit_last"
ln -s "$PF_OPD_OUT/training/opd_fullsplit/last.pt" "$PF_OPD_OUT/training/opd_fullsplit_last/selected.pt"
PF_SELECTED_STEP=$($PF_RESEARCH_PY -c "import json; print(json.load(open('$PF_OPD_OUT/training/opd_fullsplit/summary.json'))['selected_step'])")
if [ "$PF_SELECTED_STEP" -gt 0 ]; then
  "$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/evaluate.py" --root "$PF_OPD_OUT" --methods opd_fullsplit --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_OPD_OUT/development_validation_03"
  "$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/evaluate.py" --root "$PF_OPD_OUT" --methods opd_fullsplit --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_OPD_OUT/development_online_03"
fi
"$PF_RESEARCH_PY" "$PF_PREVIOUS_ROOT/source/offline_acceptance.py" --root "$PF_OPD_OUT" --data "$PF_OPD_OUT/development_states/data/val.pt" --selection "$PF_OPD_OUT/development_states/data/selection.json" --methods w1,fusion_long,opd_fullsplit,opd_fullsplit_last --batch-size 1 --out "$PF_OPD_OUT/development_offline_02"
PF_MORE_RECORDS=()
if [ -f "$PF_OPD_OUT/development_online_03/records.jsonl" ]; then
  PF_MORE_RECORDS=(--additional-records "$PF_OPD_OUT/development_online_03/records.jsonl")
fi
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/summarize_paper.py" report --root "$PF_OPD_OUT" --additional-records "$PF_OPD_OUT/development_online_02/records.jsonl" "${PF_MORE_RECORDS[@]}" --additional-offline "$PF_OPD_OUT/development_offline_02/state_metrics.jsonl" --candidate-methods opd_ce_tv,opd_split,opd_forward,opd_split_last,opd_fullsplit,opd_fullsplit_last
