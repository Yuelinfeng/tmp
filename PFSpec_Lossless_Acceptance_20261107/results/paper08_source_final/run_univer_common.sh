#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_UNIVER_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_UNIVER_PY=/root/miniconda3/envs/gaussiangpt/bin/python
"$PF_UNIVER_PY" - <<'PYWAIT'
import json,time
from pathlib import Path
p=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper08_evaluation_resume_03_process.json')
while json.loads(p.read_text())['status']=='running':time.sleep(10)
if json.loads(p.read_text())['status']!='complete':raise RuntimeError('Natural evaluation is unfinished')
PYWAIT
"$PF_UNIVER_PY" "$PF_UNIVER_ROOT/source/offline_univer.py" --out "$PF_UNIVER_ROOT/paper08_univer/development_curve_01"
"$PF_UNIVER_PY" "$PF_UNIVER_ROOT/source/report_multidraft.py" --root "$PF_UNIVER_ROOT/paper08_univer" --candidates diffusion_full_w64,diffusion_univer_w16 --additional-records "$PF_UNIVER_ROOT/paper02_draft_opd/development_online_01/records.jsonl" --additional-records "$PF_UNIVER_ROOT/paper06_dflash/development_online_02/records.jsonl"
