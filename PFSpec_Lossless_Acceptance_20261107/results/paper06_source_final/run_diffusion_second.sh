#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_SECOND_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_SECOND_PY=/root/miniconda3/envs/gaussiangpt/bin/python
"$PF_SECOND_PY" - <<'PYWAIT'
import json,time
from pathlib import Path
p=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_long_decode_process.json')
while json.loads(p.read_text())['status']=='running':time.sleep(10)
PYWAIT
"$PF_SECOND_PY" "$PF_SECOND_ROOT/source/offline_diffusion_second.py" --limit-per-source 2 --out "$PF_SECOND_ROOT/paper06_dflash/development_second_probe"
"$PF_SECOND_PY" "$PF_SECOND_ROOT/source/offline_diffusion_second.py"
