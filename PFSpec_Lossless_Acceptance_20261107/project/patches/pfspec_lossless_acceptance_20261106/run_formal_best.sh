#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_BEST_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_BEST_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_BEST_OUT="$PF_BEST_ROOT/formal_acceptance_01"
"$PF_BEST_PY" - <<'PYSETUP'
import json,time
from pathlib import Path
r=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
p=r/'formal_references_process.json'
while json.loads(p.read_text())['status']=='running':time.sleep(10)
if json.loads(p.read_text())['status']!='complete':raise RuntimeError('Reference generation did not complete')
selected=json.loads((r/'selection_before_final.json').read_text())
if selected['status']!='complete':raise RuntimeError('Method selection is unfinished')
t=r/'formal_acceptance_01/training'
t.mkdir(parents=True,exist_ok=True)
dest=Path(selected['inference_directory'])
path=t/selected['method']
if path.exists() or path.is_symlink():
    if path.resolve()!=dest.resolve():raise ValueError('Existing model location differs')
else:path.symlink_to(dest,target_is_directory=True)
PYSETUP
"$PF_BEST_PY" "$PF_BEST_ROOT/source/evaluate_final.py" --root "$PF_BEST_OUT" --methods diffusion_full_w64 --capacity 32768 --inputs "$PF_BEST_ROOT/evaluation_inputs/natural_complete_inputs.json" --expected-gpu 5090 --out "$PF_BEST_OUT/diffusion_full_w64_natural"
"$PF_BEST_PY" "$PF_BEST_ROOT/source/evaluate_final.py" --root "$PF_BEST_OUT" --methods diffusion_full_w64 --capacity 32768 --inputs "$PF_BEST_ROOT/evaluation_inputs/capped_inputs.json" --expected-gpu 5090 --out "$PF_BEST_OUT/diffusion_full_w64_capped"
"$PF_BEST_PY" "$PF_BEST_ROOT/source/offline_final_common.py" --selection "$PF_BEST_ROOT/selection_before_final.json" --records "$PF_BEST_OUT/w1_natural/records.jsonl" --inputs "$PF_BEST_ROOT/evaluation_inputs/natural_complete_inputs.json" --out "$PF_BEST_OUT/common_histories"
"$PF_BEST_PY" "$PF_BEST_ROOT/source/report_final.py" --records "$PF_BEST_OUT/w1_natural/records.jsonl" --records "$PF_BEST_OUT/fusion_long_natural/records.jsonl" --records "$PF_BEST_OUT/diffusion_full_w64_natural/records.jsonl" --inputs "$PF_BEST_ROOT/evaluation_inputs/natural_complete_inputs.json" --methods w1,fusion_long,diffusion_full_w64 --out "$PF_BEST_OUT/natural_comparison.json"
"$PF_BEST_PY" "$PF_BEST_ROOT/source/report_final.py" --records "$PF_BEST_OUT/w1_capped/records.jsonl" --records "$PF_BEST_OUT/fusion_long_capped/records.jsonl" --records "$PF_BEST_OUT/diffusion_full_w64_capped/records.jsonl" --inputs "$PF_BEST_ROOT/evaluation_inputs/capped_inputs.json" --methods w1,fusion_long,diffusion_full_w64 --out "$PF_BEST_OUT/capped_comparison.json"
