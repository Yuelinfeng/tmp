"""Paired source-level comparisons with explicit single and set acceptance scopes."""
import argparse,json,random,sys
from collections import defaultdict
from pathlib import Path


def mean_interval(rows,metric):
    groups=defaultdict(list)
    for row in rows:
        if row.get(metric) is not None:groups[row['source']].append(row[metric])
    values=[sum(v)/len(v) for _,v in sorted(groups.items())]
    if not values:return dict(mean=None,ci95=None,sources=0)
    rng=random.Random(20261106)
    samples=sorted(sum(rng.choices(values,k=len(values)))/len(values) for _ in range(5000))
    return dict(mean=sum(values)/len(values),ci95=[samples[125],samples[4875]],sources=len(values))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--candidates',default='multi_greedy_w16')
    p.add_argument('--additional-records',type=Path,action='append',default=[])
    p.add_argument('--additional-offline',type=Path,action='append',default=[])
    p.add_argument('--out',type=Path)
    a=p.parse_args()
    sys.path.insert(0,'/root/autodl-tmp/pfspec_feature_states_20261105/source')
    import acceptance_metrics
    from multi_metrics import configure
    configure(acceptance_metrics)
    from long_metrics import configure as configure_second_feature
    configure_second_feature(acceptance_metrics)
    methods=a.candidates.split(',');included=set(methods+['w1','fusion_long'])
    paths=list(a.root.glob('development_online_*/records.jsonl'))+a.additional_records
    raw=[json.loads(line) for path in paths for line in path.read_text().splitlines() if line.strip()]
    raw=[r for r in raw if r['method'] in included]
    rows=[acceptance_metrics.normalize(r) for r in raw]
    identities=[(r['pair_id'],r['method']) for r in rows]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate paired input and method')
    result=acceptance_metrics.summarize(rows)
    fields=['feature_conditional_acceptance','pair_joint_acceptance','mean_emitted_per_round']
    comparisons={};intervals={}
    for cohort in sorted({r['cohort'] for r in rows}):
        own=[r for r in rows if r['cohort']==cohort]
        present={r['method'] for r in own};comparisons[cohort]={};intervals[cohort]={}
        for method in sorted(present):
            selected=[r for r in own if r['method']==method]
            intervals[cohort][method]={k:mean_interval(selected,k) for k in fields}
            if method in methods:
                for ref in ['fusion_long','w1']+methods[:methods.index(method)]:
                    if ref in present:comparisons[cohort][method+'_minus_'+ref]={k:acceptance_metrics.paired_difference(own,method,ref,k) for k in fields}
    offline=[json.loads(line) for line in (a.root/'development_curve_01/state_metrics.jsonl').read_text().splitlines()]
    existing={(r['pair_id'],r.get('state_index'),r['method']) for r in offline}
    for path in a.additional_offline:
        for line in path.read_text().splitlines():
            row=json.loads(line)
            key=(row['pair_id'],row.get('state_index'),row['method'])
            if key not in existing:offline.append(row);existing.add(key)
    common={}
    for cohort in sorted({r['cohort'] for r in offline}):
        own=[r for r in offline if r['cohort']==cohort];present={r['method'] for r in own};common[cohort]={}
        for method in methods:
            if method in present:
                common[cohort][method+'_minus_fusion_long']={k:acceptance_metrics.paired_difference(own,method,'fusion_long',k) for k in ['feature_overlap','feature_set_acceptance','pair_joint_probability']}
    out=a.out or a.root/'development_comparison.json'
    out.write_text(json.dumps(dict(scope='Development only; formal reserve remains unused',online=result,
        source_mean_intervals=intervals,online_paired_comparisons=comparisons,
        common_history_paired_comparisons=common,raw_record_paths=[str(p) for p in paths],
        model_scope='Per-method checkpoints and actual Draft policies are retained in each evaluation plan; trained weights may differ',
        acceptance_scope='Multi-candidate output membership includes residual hits; single prediction overlap remains separate',
        cost_scope='Generation includes first-use runtime creation; merged controls retain the costs recorded in their original development trials',
        peak_gpu_gib={method:max(r['peak_gpu_gib'] for r in raw if r['method']==method) for method in included if any(r['method']==method for r in raw)}),indent=2,allow_nan=False)+'\n')
    print('Paired multi-draft comparisons saved',len(rows),flush=True)


if __name__=='__main__':main()
