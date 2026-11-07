#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_LK_OUT="$PF_RESEARCH_ROOT/paper03_lk"
PF_FEEDBACK="$PF_RESEARCH_ROOT/paper02_draft_opd"
mkdir "$PF_LK_OUT"
mkdir "$PF_LK_OUT/training"
ln -s "$PF_PREVIOUS_ROOT/data" "$PF_LK_OUT/data"
ln -s "$PF_PREVIOUS_ROOT/training/fusion_long" "$PF_LK_OUT/training/fusion_long"
"$PF_RESEARCH_PY" - <<'PY'
from pathlib import Path
import json,torch
root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper03_lk/training/lk_full_init')
root.mkdir()
saved=torch.load('/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long/selected.pt',map_location='cpu',weights_only=False)
saved.update(draft_top_p=1.,step=0,mode='initial_weights_with_full_draft_policy',
             validation=dict(scope='Policy control; actual overlap measured in current offline evaluation'))
torch.save(saved,root/'selected.pt')
(root/'config.json').write_text(json.dumps(dict(initial='/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long/selected.pt',trained=False,target_sampling=dict(temperature=.9,top_p=.9),draft_sampling=dict(temperature=.9,top_p=1.)),indent=2))
PY
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/train_feedback.py" --feedback "$PF_FEEDBACK" --out "$PF_LK_OUT" --name lk_ce_tv --mode ce_tv --draft-top-p 1
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/train_feedback.py" --feedback "$PF_FEEDBACK" --out "$PF_LK_OUT" --name lk_log --mode lk_log --draft-top-p 1
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_policies.py" --root "$PF_LK_OUT" --methods lk_full_init,lk_ce_tv,lk_log --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_LK_OUT/development_validation_01"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_policies.py" --root "$PF_LK_OUT" --methods lk_full_init,lk_ce_tv,lk_log --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_LK_OUT/development_online_01"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/offline_policies.py" --root "$PF_LK_OUT" --data "$PF_FEEDBACK/development_states/data/val.pt" --selection "$PF_FEEDBACK/development_states/data/selection.json" --methods w1,fusion_long,lk_full_init,lk_ce_tv,lk_log --batch-size 1 --out "$PF_LK_OUT/development_offline_01"
