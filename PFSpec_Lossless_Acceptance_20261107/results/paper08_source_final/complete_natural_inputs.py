"""Allow every legal Position/Feature pair and the final EOS in natural runs."""
import json
from pathlib import Path
import sys


def main():
    sys.path.insert(0, '/root/workspace/GaussianGPT')
    from draft_a.pf_feature_unified import P_VOCAB
    root = Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/evaluation_inputs')
    original = root/'natural_inputs.json'
    out = root/'natural_complete_inputs.json'
    if out.exists():
        raise FileExistsError(out)
    data = json.loads(original.read_text())
    maximum = 2*P_VOCAB+1
    budget = maximum + maximum % 2
    for case in data['contexts']:
        case['budget'] = budget
    data['generation_bound'] = dict(position_vocabulary=P_VOCAB,
        maximum_bos_generated_tokens=maximum, even_decoder_budget=budget,
        reason='Strictly increasing Position ranks allow at most one occurrence per Position; each has one Feature, followed by EOS',
        original_inputs=str(original), scope='Prompts, sources and seeds remain the original reserved inputs; capped inputs retain their 512-token budget')
    out.write_text(json.dumps(data, indent=2)+chr(10))
    print('Complete natural inputs', len(data['contexts']), 'budget', budget, flush=True)


if __name__ == '__main__':
    main()
