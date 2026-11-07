#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_LK_OUT="$PF_RESEARCH_ROOT/paper03_lk"
PF_FEEDBACK="$PF_RESEARCH_ROOT/paper02_draft_opd"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/train_feedback.py" --feedback "$PF_FEEDBACK" --out "$PF_LK_OUT" --name lk_adaptive --mode lk_adaptive --draft-top-p 1 --eta 3
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_policies.py" --root "$PF_LK_OUT" --methods lk_adaptive --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_LK_OUT/development_validation_02"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_policies.py" --root "$PF_LK_OUT" --methods lk_adaptive --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_LK_OUT/development_online_02"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/offline_policies.py" --root "$PF_LK_OUT" --data "$PF_FEEDBACK/development_states/data/val.pt" --selection "$PF_FEEDBACK/development_states/data/selection.json" --methods w1,lk_adaptive --batch-size 1 --out "$PF_LK_OUT/development_offline_02"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/summarize_paper.py" report --root "$PF_LK_OUT" --additional-records "$PF_FEEDBACK/development_online_01/records.jsonl" --additional-records "$PF_LK_OUT/development_online_02/records.jsonl" --additional-offline "$PF_LK_OUT/development_offline_02/state_metrics.jsonl" --record-methods w1,fusion_long,lk_full_init,lk_ce_tv,lk_log,lk_adaptive --candidate-methods lk_full_init,lk_ce_tv,lk_log,lk_adaptive --reference-methods fusion_long,lk_ce_tv,lk_full_init,lk_log
