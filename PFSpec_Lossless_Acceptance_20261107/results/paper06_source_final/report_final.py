"""Report all prespecified natural and capped inputs with source-level intervals."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys


METRICS = [
    'feature_conditional_acceptance', 'pair_joint_acceptance',
    'mean_emitted_per_round', 'generated_tokens_per_second',
    'second_feature_conditional_acceptance', 'second_feature_joint_acceptance',
    'total_target_tokens_per_call',
]
COUNTS = [
    'eligible_full_rounds', 'first_position_accepted', 'first_feature_accepted',
    'first_pair_accepted', 'total_draft_tokens', 'total_accepted_draft_tokens',
    'rounds', 'round_emitted_tokens', 'target_verification_calls',
    'target_verification_tokens', 'total_target_calls', 'total_target_input_tokens',
    'candidate_count', 'second_feature_eligible_rounds', 'second_feature_reached',
    'second_feature_hits', 'second_feature_distinct_candidates',
    'second_feature_branch_nodes', 'total_feature_candidate_nodes',
]


def report(raw, contexts, methods):
    import acceptance_metrics
    from multi_metrics import configure
    from long_metrics import configure as configure_long
    from report_multidraft import mean_interval
    configure(acceptance_metrics)
    configure_long(acceptance_metrics)
    cases = {r['pair_id']: r for r in contexts}
    if len(cases) != len(contexts):
        raise ValueError('Input identities repeat')
    expected = {(case, method) for case in cases for method in methods}
    identities = [(r['pair_id'], r['method']) for r in raw]
    if len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError('Saved requests differ from the complete prespecified input set')
    for row in raw:
        case = cases[row['pair_id']]
        for field in ['source', 'seed', 'prefix', 'cohort']:
            if row[field] != case[field]:
                raise ValueError(f'Saved input differs: {row["pair_id"]}, {field}')
        if row['requested_tokens'] != int(case.get('budget', 16000)):
            raise ValueError('Saved generation budget differs from the input')
    rows = [acceptance_metrics.normalize(r) for r in raw]
    result = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        own = [r for r in rows if r['cohort'] == cohort]
        by_method = {}
        for method in methods:
            selected = [r for r in own if r['method'] == method]
            originals = [r for r in raw if r['cohort'] == cohort and r['method'] == method]
            totals = {key: sum(r.get(key, 0) or 0 for r in selected)
                      for key in COUNTS if any(key in r for r in selected)}
            intervals = {key: mean_interval(selected, key) for key in METRICS
                         if any(r.get(key) is not None for r in selected)}
            for key, interval in intervals.items():
                interval['undefined_requests'] = sum(r.get(key) is None for r in selected)
            comparisons = {}
            for reference in methods:
                if reference == method:
                    continue
                differences = {}
                for key in intervals:
                    reference_rows = [r for r in own if r['method'] == reference]
                    if any(r.get(key) is not None for r in reference_rows):
                        differences[key] = acceptance_metrics.paired_difference(
                            own, method, reference, key)
                comparisons[reference] = differences
            by_method[method] = dict(
                requests=len(selected), sources=len({r['source'] for r in selected}),
                eos_requests=sum(bool(r['eos']) for r in selected),
                terminations=dict(Counter(r['termination'] for r in originals)),
                generated_tokens=sum(r['emitted_tokens'] for r in selected),
                generation_seconds=sum(r['total_generation_seconds'] for r in selected),
                peak_gpu_gib=max(r['peak_gpu_gib'] for r in originals),
                counts=totals, source_mean_intervals=intervals,
                paired_differences=comparisons,
                count_scope='Actual sums across all requests; intervals average seeds within each source',
            )
        result[cohort] = by_method
    return dict(status='complete', requests=len(raw), methods=methods, cohorts=result,
                uncertainty_scope='5000 paired source resamples; repeated Completion seeds remain within their source',
                feature_scope='First Feature set membership includes residual hits; second Feature counters stay separate',
                selection_scope='All methods are selected before formal output is read; formal outcomes do not select weights',
                cost_scope='Generation includes prefix and first-use runtime creation; process setup and discarded warmup are separate')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, action='append', required=True)
    p.add_argument('--inputs', type=Path, action='append', required=True)
    p.add_argument('--methods', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    old = Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    sys.path.insert(0, str(old))
    contexts = [r for path in a.inputs for r in json.loads(path.read_text())['contexts']]
    import torch
    cases = {r['pair_id']: r for r in contexts}
    raw = []
    token_files = 0
    for path in a.records:
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            saved = torch.load(path.parent / 'tokens' / f'{row["pair_id"]}_{row["method"]}.pt',
                               map_location='cpu', weights_only=False)
            if saved['prompt'] != cases[row['pair_id']]['prompt']:
                raise ValueError('Saved token prefix differs from the input')
            suffix = saved['suffix']
            if len(suffix) != row['generated_tokens'] or bool(suffix and suffix[-1] == 12096) != row['eos']:
                raise ValueError('Saved output tokens differ from the raw request')
            raw.append(row)
            token_files += 1
    result = report(raw, contexts, a.methods.split(','))
    result['verified_output_token_files'] = token_files
    result['raw_record_paths'] = [str(p) for p in a.records]
    result['input_paths'] = [str(p) for p in a.inputs]
    a.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print('Complete paired report saved', len(raw), flush=True)


if __name__ == '__main__':
    main()
