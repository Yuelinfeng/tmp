"""Acceptance counts for the existing single-Feature W1 path."""
import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import random


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def source_average(rows, field):
    groups = defaultdict(list)
    omitted = 0
    for row in rows:
        value = row.get(field)
        if value is None:
            omitted += 1
        else:
            groups[row['source']].append(value)
    values = [sum(items)/len(items) for items in groups.values()]
    return dict(mean=ratio(sum(values), len(values)), sources=len(values), omitted_rows=omitted)


def online_counts(stats):
    """A full W1 round reaches F exactly when its first P was accepted."""
    if stats.get('tree_width') != 1:
        raise ValueError('These first-pair counts require one Feature candidate')
    eligible = stats['tree_verify_calls']
    first_p = stats['feature_verified']
    first_f = stats['feature_accepted']
    if stats['feature_attempts'] != first_p:
        raise ValueError('Feature attempt and verification counts differ')
    if stats['feature_proposed'] != eligible or not 0 <= first_f <= first_p <= eligible:
        raise ValueError('Counts do not describe the single-Feature full-round path')
    if stats['feature_branch_hits'] != first_f:
        raise ValueError('Feature hits and accepted candidates differ')
    lengths = stats.get('round_generated_lengths')
    if lengths is not None and len(lengths) != stats['rounds']:
        raise ValueError('Generated lengths do not cover all rounds')
    return dict(
        eligible_full_rounds=eligible,
        first_position_accepted=first_p,
        first_feature_accepted=first_f,
        first_pair_accepted=first_f,
        feature_rejected=first_p-first_f,
        feature_not_reached=eligible-first_p,
        total_draft_tokens=stats['proposed'],
        total_accepted_draft_tokens=stats['accepted'],
        target_verification_calls=eligible,
        target_verification_tokens=stats['target_tree_nodes'],
        total_target_calls=stats.get('target_calls'),
        total_target_input_tokens=stats.get('target_input_tokens'),
        rounds=stats['rounds'],
        round_emitted_tokens=sum(lengths) if lengths is not None else None,
        position_acceptance=ratio(first_p, eligible),
        feature_conditional_acceptance=ratio(first_f, first_p),
        pair_joint_acceptance=ratio(first_f, eligible),
        mean_emitted_per_round=ratio(sum(lengths), stats['rounds']) if lengths is not None else None,
    )


def normalize(row):
    result = {k: row[k] for k in ('cohort', 'pair_id', 'source', 'method', 'seed', 'prefix')}
    result.update(emitted_tokens=row['generated_tokens'], total_generation_seconds=row['seconds'],
                  generated_tokens_per_second=ratio(row['generated_tokens'], row['seconds']),
                  eos=row['eos'], draft_proposal_seconds=None)
    if row['method'] in ('opt_target', 'ranker_only'):
        result['feature_metrics_available'] = False
        return result
    result.update(online_counts(row['counts']))
    if result['round_emitted_tokens'] is None:
        bootstrap = min(2, row['generated_tokens']) if row['prefix'] == 0 else 0
        result['round_emitted_tokens'] = row['generated_tokens']-bootstrap
        result['mean_emitted_per_round'] = ratio(result['round_emitted_tokens'], result['rounds'])
        result['round_emission_scope'] = 'Output length excluding BOS bootstrap; older record omitted per-round lengths'
    else:
        result['round_emission_scope'] = 'Sum of saved per-round generated lengths'
    result['feature_metrics_available'] = True
    return result


def paired_difference(rows, method, reference, metric, *, repeats=5000, seed=20261106):
    indexed = {(r['pair_id'], r['method']): r for r in rows}
    cases = {r['pair_id'] for r in rows if r['method'] == method}
    reference_cases = {r['pair_id'] for r in rows if r['method'] == reference}
    if cases != reference_cases:
        raise ValueError('Paired methods must cover the same input cases')
    differences = defaultdict(list)
    undefined = []
    for case in sorted(cases):
        left, right = indexed[(case, method)], indexed[(case, reference)]
        if left['source'] != right['source']:
            raise ValueError('Paired inputs have different source ids')
        x, y = left.get(metric), right.get(metric)
        if x is None or y is None:
            undefined.append(case)
            continue
        differences[left['source']].append(x-y)
    values = [sum(v)/len(v) for _, v in sorted(differences.items())]
    if not values:
        return dict(difference=None, ci95=None, sources=0, undefined_cases=undefined)
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(values, k=len(values)))/len(values) for _ in range(repeats))
    return dict(difference=sum(values)/len(values),
                ci95=[samples[int(.025*repeats)], samples[min(repeats-1, int(.975*repeats))]],
                sources=len(values), requests=len(cases), undefined_cases=undefined,
                bootstrap_repeats=repeats, bootstrap_seed=seed)


def summarize(rows):
    result = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        own = [r for r in rows if r['cohort'] == cohort]
        methods = sorted({r['method'] for r in own})
        groups = {}
        for method in methods:
            selected = [r for r in own if r['method'] == method]
            group = dict(requests=len(selected), sources=len({r['source'] for r in selected}),
                         eos=sum(r['eos'] for r in selected),
                         emitted_tokens=sum(r['emitted_tokens'] for r in selected),
                         generation_seconds=sum(r['total_generation_seconds'] for r in selected))
            if all(r['feature_metrics_available'] for r in selected):
                count_keys = ('eligible_full_rounds', 'first_position_accepted', 'first_feature_accepted',
                              'first_pair_accepted', 'total_draft_tokens', 'total_accepted_draft_tokens',
                              'target_verification_calls', 'target_verification_tokens', 'rounds')
                counts = {k: sum(r[k] for r in selected) for k in count_keys}
                for key in ('total_target_calls', 'total_target_input_tokens'):
                    if all(r.get(key) is not None for r in selected):
                        counts[key] = sum(r[key] for r in selected)
                group['counts'] = counts
                group['pooled_rates'] = dict(
                    position_acceptance=ratio(counts['first_position_accepted'], counts['eligible_full_rounds']),
                    feature_conditional_acceptance=ratio(counts['first_feature_accepted'], counts['first_position_accepted']),
                    pair_joint_acceptance=ratio(counts['first_pair_accepted'], counts['eligible_full_rounds']))
                group['source_mean_metrics'] = {
                    k: source_average(selected, k)
                    for k in ('position_acceptance', 'feature_conditional_acceptance',
                              'pair_joint_acceptance', 'mean_emitted_per_round',
                              'generated_tokens_per_second', 'total_generation_seconds')}
                if method != 'w1' and 'w1' in methods:
                    group['paired_differences_vs_w1'] = {
                        k: paired_difference(own, method, 'w1', k)
                        for k in ('position_acceptance', 'feature_conditional_acceptance',
                                  'pair_joint_acceptance', 'mean_emitted_per_round')}
            groups[method] = group
        result[cohort] = groups
    return dict(cohorts=result,
                rate_scope='Pooled rates retain raw counts; paired differences average within each source first',
                feature_scope='Original draft candidates only; residual and bonus samples are emitted tokens',
                verification_scope='Full-round tree verification, including its pending input; total Target counts also include prefix, bootstrap, K1 and recovery',
                draft_time_scope='Not measured by this count normalizer')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = [normalize(json.loads(line)) for line in args.records.read_text().splitlines() if line.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (args.out/'online_counts.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize(rows)
    (args.out/'online_summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    table = []
    for cohort, groups in summary['cohorts'].items():
        for method, group in groups.items():
            item = dict(cohort=cohort, method=method, **{k:group[k] for k in
                ('requests','sources','eos','emitted_tokens','generation_seconds')})
            item.update(group.get('counts', {}))
            for metric, result in group.get('source_mean_metrics', {}).items():
                item['source_mean_'+metric] = result['mean']
            for metric, result in group.get('paired_differences_vs_w1', {}).items():
                item[metric+'_difference_vs_w1'] = result['difference']
                interval = result['ci95'] or [None, None]
                item[metric+'_difference_lower95'] = interval[0]
                item[metric+'_difference_upper95'] = interval[1]
            table.append(item)
    fields = list(dict.fromkeys(k for row in table for k in row))
    with (args.out/'cohort_summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(dict(requests=len(rows), output=str(args.out))))


if __name__ == '__main__':
    main()
