#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_PREVIOUS_ROOT=/root/autodl-tmp/pfspec_feature_states_20261105
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_BLOCK_OUT="$PF_RESEARCH_ROOT/paper05_block_verification"
"$PF_RESEARCH_PY" - <<'PY'
from pathlib import Path
import json,torch
root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper05_block_verification')
initial=Path('/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long/selected.pt')
(root/'training').mkdir()
for width in (1,64):
    path=root/'training'/f'block_joint_w{width}';path.mkdir()
    saved=torch.load(initial,map_location='cpu',weights_only=False)
    saved.update(feature_candidate_width=width,draft_top_p=.9,block_pair_verification=True,
        coupling='Algorithm 2 two-token block with fixed Feature branch reuse',initial=str(initial))
    torch.save(saved,path/'selected.pt')
    (path/'config.json').write_text(json.dumps(dict(initial=str(initial),trained=False,
        target_sampling=dict(temperature=.9,top_p=.9),draft_sampling=dict(temperature=.9,top_p=.9),
        requested_feature_candidates=width,target_nodes_per_full_round=2+2*width,
        block_length=2,block_pair_verification=True,cache_capacity=32768,
        role='Single-candidate mechanism control' if width==1 else 'Primary block plus set implementation',
        scope='Unchanged Target and Draft; joint first P/F decisions and complete scaled residual'),indent=2)+'\n')
(root/'training/fusion_long').symlink_to(initial.parent,target_is_directory=True)
(root/'training/multi_greedy_w64').symlink_to(Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper04_optimal_multi_draft/training/multi_greedy_w64'),target_is_directory=True)
PY
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_block.py" --root "$PF_BLOCK_OUT" --methods block_joint_w1,block_joint_w64 --capacity 32768 --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --validate --out "$PF_BLOCK_OUT/development_validation_01"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_block.py" --root "$PF_BLOCK_OUT" --methods block_joint_w1,block_joint_w64 --capacity 32768 --inputs "$PF_PREVIOUS_ROOT/acceptance_study/development_inputs.json" --expected-gpu 5090 --out "$PF_BLOCK_OUT/development_online_01"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/report_multidraft.py" --root "$PF_BLOCK_OUT" --candidates multi_greedy_w64,block_joint_w1,block_joint_w64 --additional-records "$PF_RESEARCH_ROOT/paper02_draft_opd/development_online_01/records.jsonl" --additional-records "$PF_RESEARCH_ROOT/paper04_optimal_multi_draft/development_online_02/records.jsonl"
