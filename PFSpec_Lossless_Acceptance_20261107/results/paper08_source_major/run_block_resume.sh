#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_RESEARCH_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_RESEARCH_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_BLOCK_OUT="$PF_RESEARCH_ROOT/paper05_block_verification"
"$PF_RESEARCH_PY" - <<'PY'
from pathlib import Path
import json
root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper05_block_verification')
inputs=json.loads(Path('/root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json').read_text())
records=[json.loads(line) for line in (root/'development_online_01/records.jsonl').read_text().splitlines()]
for method in ('block_joint_w1','block_joint_w64'):
    completed={r['pair_id'] for r in records if r['method']==method}
    cases=[c for c in inputs['contexts'] if c['pair_id'] not in completed]
    path=root/(method+'_remaining_inputs.json')
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(dict(contexts=cases,scope='Remaining inputs after physical GPU memory failure; completed records retained'),indent=2)+'\n')
    print(method,'completed',len(completed),'remaining',len(cases),flush=True)
PY
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_block.py" --root "$PF_BLOCK_OUT" --methods block_joint_w1 --capacity 32768 --inputs "$PF_BLOCK_OUT/block_joint_w1_remaining_inputs.json" --expected-gpu 5090 --out "$PF_BLOCK_OUT/development_online_02"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/evaluate_block.py" --root "$PF_BLOCK_OUT" --methods block_joint_w64 --capacity 32768 --inputs "$PF_BLOCK_OUT/block_joint_w64_remaining_inputs.json" --expected-gpu 5090 --out "$PF_BLOCK_OUT/development_online_03"
"$PF_RESEARCH_PY" "$PF_RESEARCH_ROOT/source/report_multidraft.py" --root "$PF_BLOCK_OUT" --candidates multi_greedy_w64,block_joint_w1,block_joint_w64 --additional-records "$PF_RESEARCH_ROOT/paper02_draft_opd/development_online_01/records.jsonl" --additional-records "$PF_RESEARCH_ROOT/paper04_optimal_multi_draft/development_online_02/records.jsonl"
