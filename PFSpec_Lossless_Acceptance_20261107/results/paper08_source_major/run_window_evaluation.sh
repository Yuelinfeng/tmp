#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_WINDOW_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_WINDOW_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_WINDOW_OUT="$PF_WINDOW_ROOT/paper07_aat_window"
"$PF_WINDOW_PY" - <<'PYWAIT'
import json,time
from pathlib import Path
p=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper07_training_process.json')
while json.loads(p.read_text())['status']=='running':time.sleep(10)
if json.loads(p.read_text())['status']!='complete':raise RuntimeError('Matched training is unfinished')
PYWAIT
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/offline_window.py"
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/evaluate_diffusion.py" --root "$PF_WINDOW_OUT" --methods diffusion_window_ce_tv_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --validate --out "$PF_WINDOW_OUT/development_validation_01"
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/evaluate_diffusion.py" --root "$PF_WINDOW_OUT" --methods diffusion_window_ce_tv_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --out "$PF_WINDOW_OUT/development_online_01"
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/evaluate_diffusion.py" --root "$PF_WINDOW_OUT" --methods diffusion_window_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --validate --out "$PF_WINDOW_OUT/development_validation_02"
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/evaluate_diffusion.py" --root "$PF_WINDOW_OUT" --methods diffusion_window_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --out "$PF_WINDOW_OUT/development_online_02"
"$PF_WINDOW_PY" "$PF_WINDOW_ROOT/source/report_multidraft.py" --root "$PF_WINDOW_OUT" --candidates diffusion_full_w64,diffusion_window_ce_tv_w64,diffusion_window_w64 --additional-records "$PF_WINDOW_ROOT/paper02_draft_opd/development_online_01/records.jsonl" --additional-records "$PF_WINDOW_ROOT/paper06_dflash/development_online_02/records.jsonl"
