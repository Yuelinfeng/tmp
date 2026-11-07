#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
PF_WRAP_ROOT=/root/autodl-tmp/pfspec_lossless_acceptance_20261106
PF_WRAP_PY=/root/miniconda3/envs/gaussiangpt/bin/python
PF_WRAP_OUT="$PF_WRAP_ROOT/acceptance_wrap_validation"
"$PF_WRAP_PY" - <<'PYSETUP'
from pathlib import Path
r=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
t=r/'acceptance_wrap_validation'/'training'
t.mkdir(parents=True,exist_ok=True)
for name,dest in [('fusion_long',Path('/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long')),('diffusion_full_w64',r/'paper06_dflash/training/diffusion_full_w64')]:
    p=t/name
    if p.exists() or p.is_symlink():
        if p.resolve()!=dest.resolve():raise ValueError('Existing training path differs')
    else:p.symlink_to(dest,target_is_directory=True)
PYSETUP
for PF_WRAP_METHOD in w1 fusion_long diffusion_full_w64; do
    "$PF_WRAP_PY" "$PF_WRAP_ROOT/source/evaluate_final.py" --root "$PF_WRAP_OUT" --methods "$PF_WRAP_METHOD" --capacity 32768 --inputs /root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json --expected-gpu 5090 --validate --out "$PF_WRAP_OUT/$PF_WRAP_METHOD"
done
