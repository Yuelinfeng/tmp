#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_DIFFUSION_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_DIFFUSION_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_DIFFUSION_OUT="$PF_DIFFUSION_ROOT/paper06_dflash"
"$PF_DIFFUSION_PY" - <<'PY'
from pathlib import Path
import json
root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash')
out=root/'training/diffusion_full_w64'
out.mkdir()
(out/'selected.pt').symlink_to('../diffusion_context_w64/selected.pt')
(out/'inference.json').write_text(json.dumps(dict(draft_top_p=1.,feature_block_depth=1,
    target_sampling=dict(temperature=.9,top_p=.9),feature_candidate_width=64,
    checkpoint='../diffusion_context_w64/selected.pt',trained=False,
    scope='First targeted evaluation: actual full-support Draft with unchanged selected weights'),indent=2)+'\n')
PY
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/offline_diffusion_support.py"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/evaluate_diffusion.py" --root "$PF_DIFFUSION_OUT" --methods diffusion_full_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --validate --out "$PF_DIFFUSION_OUT/development_validation_03"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/evaluate_diffusion.py" --root "$PF_DIFFUSION_OUT" --methods diffusion_full_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --out "$PF_DIFFUSION_OUT/development_online_02"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/report_multidraft.py" --root "$PF_DIFFUSION_OUT" --candidates multi_greedy_w64,diffusion_context_w64,diffusion_full_w64 --additional-records "$PF_DIFFUSION_ROOT/paper02_draft_opd/development_online_01/records.jsonl" --additional-records "$PF_DIFFUSION_ROOT/paper04_optimal_multi_draft/development_online_02/records.jsonl" --additional-offline "$PF_DIFFUSION_OUT/development_support_01/state_metrics.jsonl"
