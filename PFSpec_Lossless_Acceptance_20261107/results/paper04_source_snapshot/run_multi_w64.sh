#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_MULTI_OUT="$PF_RESEARCH_ROOT/paper04_optimal_multi_draft"
"$PF_RESEARCH_PY" - <<'PY'
from pathlib import Path
import json,tarfile,torch
study=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
archive=study/'paper03_deliverables.tgz'
if archive.exists():raise FileExistsError(archive)
with tarfile.open(archive,'w:gz',compresslevel=3) as bundle:
    bundle.add(study/'paper03_lk',arcname='paper03_lk')
    bundle.add(study/'paper03_numeric_validation.json',arcname='paper03_numeric_validation.json')
    for name in ['train_feedback.py','policy_inference.py','evaluate_policies.py','offline_policies.py','summarize_paper.py','run_lk.sh','run_lk_adaptive.sh']:
        bundle.add(study/'source'/name,arcname='source/'+name)
print('Paper 3 archive complete',archive.stat().st_size,flush=True)
initial=Path('/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long/selected.pt')
root=study/'paper04_optimal_multi_draft/training/multi_greedy_w64'
root.mkdir()
saved=torch.load(initial,map_location='cpu',weights_only=False)
saved.update(feature_candidate_width=64,draft_top_p=.9,coupling='greedy_fixed_set_plus_one_random',initial=str(initial))
torch.save(saved,root/'selected.pt')
(root/'config.json').write_text(json.dumps(dict(initial=str(initial),trained=False,
    target_sampling=dict(temperature=.9,top_p=.9),draft_sampling=dict(temperature=.9,top_p=.9),
    requested_feature_candidates=64,target_nodes_per_full_round=130,
    mechanism='Same Section 5 coupling; first evidence-driven increase in candidate count',
    reason='Measured 16-candidate joint and round gains plus 11-13pp additional common-history set probability',
    cache_capacity=32768,capacity_reason='Complete prompt plus requested output and 130 temporary tree rows exceed 16384',
    feature_overlap_scope='Unchanged fusion_long single prediction'),indent=2)+'\n')
PY
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/validate_multi_branches.py" --width 64 --out "$PF_MULTI_OUT/branch_validation_w64.json"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_multidraft.py" --root "$PF_MULTI_OUT" --methods multi_greedy_w64 --capacity 32768 --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_MULTI_OUT/development_validation_02"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_multidraft.py" --root "$PF_MULTI_OUT" --methods multi_greedy_w64 --capacity 32768 --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_MULTI_OUT/development_online_02"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/report_multidraft.py" --root "$PF_MULTI_OUT" --candidates multi_greedy_w16,multi_greedy_w64 --additional-records "$PF_RESEARCH_ROOT/paper02_draft_opd/development_online_01/records.jsonl"
