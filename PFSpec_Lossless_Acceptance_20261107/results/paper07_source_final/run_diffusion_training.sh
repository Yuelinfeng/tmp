#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_DIFFUSION_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_DIFFUSION_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_DIFFUSION_OUT="$PF_DIFFUSION_ROOT/paper06_dflash"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/train_diffusion.py" --root "$PF_DIFFUSION_OUT"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/offline_diffusion.py"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/evaluate_diffusion.py" --root "$PF_DIFFUSION_OUT" --methods diffusion_context_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --validate --out "$PF_DIFFUSION_OUT/development_validation_01"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/evaluate_diffusion.py" --root "$PF_DIFFUSION_OUT" --methods diffusion_context_w64 --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --out "$PF_DIFFUSION_OUT/development_online_01"
"$PF_DIFFUSION_PY" "$PF_DIFFUSION_ROOT/source/report_multidraft.py" --root "$PF_DIFFUSION_OUT" --candidates multi_greedy_w64,diffusion_context_w64 --additional-records "$PF_DIFFUSION_ROOT/paper02_draft_opd/development_online_01/records.jsonl" --additional-records "$PF_DIFFUSION_ROOT/paper04_optimal_multi_draft/development_online_02/records.jsonl"
