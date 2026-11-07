"""Build common development histories and paired paper-level comparisons."""
import argparse
import json
from pathlib import Path
import sys


def common_states(args):
    import torch
    out = args.root / 'development_states'
    if out.exists():
        raise FileExistsError(out)
    rows, files, metadata = [], [], {}
    for shard in sorted((args.root/'feedback_shards').glob('*.pt')):
        value=torch.load(shard,map_location='cpu',weights_only=False)
        record=value['record']
        if record['split']!='val':
            continue
        index=len(files)
        files.append(str(shard))
        metadata[str(shard)]={k:record[k] for k in ('source','pair_id','cohort','prefix')}
        rows.extend(dict(row,source_index=index) for row in value['rows'])
    if not rows:
        raise ValueError('No development common histories')
    data={k:torch.stack([r[k] for r in rows]) for k in ('state','logits')}
    for key in ('previous_feature','position','weight'):
        data[key]=torch.tensor([r[key] for r in rows])
    data['source']=torch.tensor([r['source_index'] for r in rows])
    data['pair_index']=torch.tensor([r['anchor']//2-1 for r in rows])
    (out/'data').mkdir(parents=True)
    torch.save(data,out/'data/val.pt')
    (out/'data/selection.json').write_text(json.dumps(dict(files={'val':files},
        source_metadata=metadata,scope='Prior development inputs; all sampled full-round starts, '
            'including unreached Feature branches; never a formal holdout'),indent=2)+'\n')
    print('Common development states',len(rows),'sources',len(files),flush=True)


def report(args):
    from acceptance_metrics import normalize, summarize, paired_difference
    records=[json.loads(line) for line in (args.root/'development_online_01/records.jsonl').read_text().splitlines()]
    for path in args.additional_records:
        records.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    if args.record_methods:
        included=set(args.record_methods.split(','))
        records=[r for r in records if r['method'] in included]
    rows=[normalize(r) for r in records]
    identities=[(r['pair_id'],r['method']) for r in rows]
    if len(identities)!=len(set(identities)):
        raise ValueError('A request and method are duplicated across result files')
    candidates=args.candidate_methods.split(',')
    pairs=[(method,reference) for method in candidates
           for reference in args.reference_methods.split(',') if method!=reference]
    comparisons={}
    for cohort in sorted({r['cohort'] for r in rows}):
        selected=[r for r in rows if r['cohort']==cohort]
        methods={r['method'] for r in selected}
        comparisons[cohort]={}
        for method,reference in pairs:
            if method not in methods or reference not in methods:
                continue
            comparisons[cohort][method+'_minus_'+reference]={
                key:paired_difference(selected,method,reference,key)
                for key in ('feature_conditional_acceptance','pair_joint_acceptance','mean_emitted_per_round')}
    offline=[]
    path=args.root/'development_offline_01/state_metrics.jsonl'
    if path.exists():
        offline=[json.loads(line) for line in path.read_text().splitlines()]
    known_methods={r['method'] for r in offline}
    for path in args.additional_offline:
        extra=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        offline.extend(r for r in extra if r['method'] not in known_methods)
        known_methods.update(r['method'] for r in extra)
    offline_comparisons={}
    for cohort in sorted({r['cohort'] for r in offline}):
        selected=[r for r in offline if r['cohort']==cohort]
        methods={r['method'] for r in selected}
        offline_comparisons[cohort]={}
        for method,reference in pairs:
            if method not in methods or reference not in methods:
                continue
            offline_comparisons[cohort][method+'_minus_'+reference]={
                key:paired_difference(selected,method,reference,key)
                for key in ('feature_overlap','pair_joint_probability')}
    out=args.root/'development_comparison.json'
    out.write_text(json.dumps(dict(scope='Development only; weights selected using these feedback sources',
        online=summarize(rows),online_paired_comparisons=comparisons,
        common_history_paired_comparisons=offline_comparisons,
        candidate_count_per_feature_slot=1,
        target_verification_nodes_per_full_round=4),indent=2,allow_nan=False)+'\n')
    print('Saved paired development comparisons',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['states','report'])
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--previous-source',type=Path,default=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source'))
    parser.add_argument('--additional-records',type=Path,action='append',default=[])
    parser.add_argument('--additional-offline',type=Path,action='append',default=[])
    parser.add_argument('--candidate-methods',default='opd_ce_tv,opd_split')
    parser.add_argument('--reference-methods',default='fusion_long,opd_ce_tv')
    parser.add_argument('--record-methods',default='')
    args=parser.parse_args()
    sys.path.insert(0,str(args.previous_source))
    if args.phase=='states':
        common_states(args)
    else:
        report(args)


if __name__=='__main__':
    main()
