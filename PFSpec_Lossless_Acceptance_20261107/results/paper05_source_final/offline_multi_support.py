"""Compare the effect of truncated Draft support at the same requested width."""
import argparse,json,sys,time
from pathlib import Path

sys.path[:0]=['/root/autodl-tmp/pfspec_feature_states_20261105/source',
 '/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04','/root/workspace/GaussianGPT']


def main():
    from research import setup,write
    from acceptance_metrics import source_average,paired_difference
    from greedy_coupling import expected_acceptance
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/root/workspace/GaussianGPT'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--width',type=int,default=64)
    a=p.parse_args();setup(a)
    if a.out.exists():raise FileExistsError(a.out)
    import torch
    from state_draft import make_draft
    from collect_feature_teacher_from_s3 import policy_batch
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    feedback=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper02_draft_opd/development_states/data')
    data=torch.load(feedback/'val.pt',map_location='cpu',weights_only=True)
    plan=json.loads((feedback/'selection.json').read_text())
    weight=torch.load(previous/'data/feature_weight.pt',map_location='cpu',weights_only=True)['weight']
    checkpoint=previous/'training/fusion_long/selected.pt'
    model=make_draft(weight,'fusion',checkpoint).cuda().float().eval().requires_grad_(False)
    a.out.mkdir(parents=True);rows=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    methods=[f'multi_greedy_w{a.width}',f'multi_greedy_full_w{a.width}']
    with torch.inference_mode(),(a.out/'state_metrics.jsonl').open('w') as handle:
        for i in range(len(data['state'])):
            _,logits=model(data['state'][i:i+1].cuda(),data['previous_feature'][i:i+1].cuda(),data['position'][i:i+1].cuda())
            policies=[policy_batch(logits.float(),.9,.9)[0],(logits.float()/.9).softmax(-1)[0]]
            target=policy_batch(data['logits'][i:i+1].cuda().float(),.9,.9)[0]
            name=plan['files']['val'][int(data['source'][i])];meta=plan['source_metadata'][name]
            position=float(data['weight'][i])
            for method,q in zip(methods,policies):
                value=float(expected_acceptance(target,q,a.width))
                row=dict(method=method,source=meta['source'],cohort=meta['cohort'],
                    pair_id=f'{meta["pair_id"]}:{int(data["pair_index"][i])}',
                    feature_overlap=float(torch.minimum(target,q).sum()),feature_set_acceptance=value,
                    pair_joint_probability=position*value,position_probability=position,
                    requested_feature_candidates=a.width,actual_feature_candidates=min(a.width,int((q>0).sum())))
                handle.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    fields=['feature_overlap','feature_set_acceptance','pair_joint_probability','actual_feature_candidates']
    result={}
    for cohort in sorted({r['cohort'] for r in rows}):
        own=[r for r in rows if r['cohort']==cohort]
        result[cohort]=dict(methods={m:{k:source_average([r for r in own if r['method']==m],k) for k in fields} for m in methods},
            differences={k:paired_difference(own,methods[1],methods[0],k) for k in fields})
    write(a.out/'summary.json',dict(status='complete',cohorts=result,seconds=seconds,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,checkpoint=str(checkpoint),
        target_sampling=dict(temperature=.9,top_p=.9),draft_top_p=[.9,1.],
        scope='Development states only; unchanged weights, actual support and candidate count explicitly compared'))
    print('Draft support comparison complete',len(rows),flush=True)


if __name__=='__main__':main()
