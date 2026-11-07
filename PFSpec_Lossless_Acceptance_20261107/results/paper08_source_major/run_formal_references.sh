#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_FINAL_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_FINAL_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_FINAL_OUT="$PF_FINAL_ROOT/formal_acceptance_01"
"$PF_FINAL_PY" - <<'PYSETUP'
import json,time
from pathlib import Path
r=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
p=r/'paper08_common_process.json'
while json.loads(p.read_text())['status']=='running':time.sleep(10)
if json.loads(p.read_text())['status']!='complete':raise RuntimeError('Common-history computation is unfinished')
t=r/'formal_acceptance_01/training'
t.mkdir(parents=True,exist_ok=True)
dest=Path('/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long')
path=t/'fusion_long'
if path.exists() or path.is_symlink():
    if path.resolve()!=dest.resolve():raise ValueError('Existing model location differs')
else:path.symlink_to(dest,target_is_directory=True)
references=dict(methods=['w1','fusion_long'],
    w1='/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt',
    fusion_long=str(dest/'selected.pt'),
    natural_inputs=str(r/'evaluation_inputs/natural_complete_inputs.json'),
    capped_inputs=str(r/'evaluation_inputs/capped_inputs.json'),
    natural_requests_per_method=90,capped_requests_per_method=120,
    target_temperature=.9,target_top_p=.9,
    selection_scope='Reference models are fixed; new candidates use development data only. Formal scientific outputs remain unread until all method selection is complete.')
(r/'formal_acceptance_01/reference_settings.json').write_text(json.dumps(references,indent=2)+'\n')
PYSETUP
for PF_FINAL_METHOD in w1 fusion_long; do
    "$PF_FINAL_PY" "$PF_FINAL_ROOT/source/evaluate_final.py" --root "$PF_FINAL_OUT" --methods "$PF_FINAL_METHOD" --capacity 32768 --inputs "$PF_FINAL_ROOT/evaluation_inputs/natural_complete_inputs.json" --expected-gpu 5090 --out "$PF_FINAL_OUT/${PF_FINAL_METHOD}_natural"
    "$PF_FINAL_PY" "$PF_FINAL_ROOT/source/evaluate_final.py" --root "$PF_FINAL_OUT" --methods "$PF_FINAL_METHOD" --capacity 32768 --inputs "$PF_FINAL_ROOT/evaluation_inputs/capped_inputs.json" --expected-gpu 5090 --out "$PF_FINAL_OUT/${PF_FINAL_METHOD}_capped"
done
