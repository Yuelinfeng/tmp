#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_WINDOW_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_WINDOW_PY=/root/miniconda3/envs/gaussiangpt/bin/python
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/validate_window_objective.py" --out "$PF_WINDOW_ROOT/paper07_aat_window/window_numeric_validation.json"
"$PF_WINDOW_PY" - <<'PYWAIT'
import json,time
from pathlib import Path
p=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper07_collection_process.json')
while json.loads(p.read_text())['status']=='running':time.sleep(10)
if json.loads(p.read_text())['status']!='complete':raise RuntimeError('Observation data are unfinished')
PYWAIT
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/train_window.py" --loss ce_tv
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/train_window.py" --loss window
