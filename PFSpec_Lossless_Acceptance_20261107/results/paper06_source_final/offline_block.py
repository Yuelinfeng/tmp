"""Measure single overlap and exact greedy set probabilities on shared histories."""
import argparse,json,sys,time
from pathlib import Path

sys.path[:0]=['/root/autodl-tmp/pfspec_feature_states_20261105/source',
 '/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04','/root/workspace/GaussianGPT']


def main():
    from research import setup,write
    from acceptance_metrics import source_average,paired_difference
    from greedy_coupling import expected_acceptance
    from block_coupling import expected_joint
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/root/workspace/GaussianGPT'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--widths',default='1,4,8,16,32,64')
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
    widths=[1,64]
    a.out.mkdir(parents=True);rows=[];start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),(a.out/'state_metrics.jsonl').open('w') as handle:
        for i in range(len(data['state'])):
            _,logits=model(data['state'][i:i+1].cuda(),data['previous_feature'][i:i+1].cuda(),data['position'][i:i+1].cuda())
            q=policy_batch(logits.float(),.9,.9)[0]
            target=policy_batch(data['logits'][i:i+1].cuda().float(),.9,.9)[0]
            overlap=float(torch.minimum(target,q).sum())
            number=int(data['source'][i]);name=plan['files']['val'][number]
            meta=plan['source_metadata'][name];position=float(data['weight'][i])
            for method,width,block in [('fusion_long',1,False),('multi_greedy_w64',64,False),('block_joint_w1',1,True),('block_joint_w64',64,True)]:
                joint=float(expected_joint(position,target,q,width)) if block else position*float(expected_acceptance(target,q,width))
                value=joint/position if position>0 else None
                row=dict(method=method,source=meta['source'],
                    cohort=meta['cohort'],pair_id=f'{meta["pair_id"]}:{int(data["pair_index"][i])}',
                    feature_overlap=overlap,feature_set_acceptance=value,pair_joint_probability=joint,
                    single_pair_joint_probability=position*overlap,position_probability=position,
                    requested_feature_candidates=width,actual_feature_candidates=min(width,int((q>0).sum())),
                    target_verification_nodes=2+2*min(width,int((q>0).sum())))
                handle.write(json.dumps(row,allow_nan=False)+'\n');rows.append(row)
    torch.cuda.synchronize();seconds=time.perf_counter()-start
    result={}
    fields=['feature_overlap','feature_set_acceptance','pair_joint_probability']
    for cohort in sorted({r['cohort'] for r in rows}):
        own=[r for r in rows if r['cohort']==cohort];result[cohort]={}
        for method in sorted({r['method'] for r in own}):
            selected=[r for r in own if r['method']==method]
            result[cohort][method]=dict(states=len(selected),sources=len({r['source'] for r in selected}),
                source_means={k:source_average(selected,k) for k in fields},
                differences_vs_fusion_long={k:paired_difference(own,method,'fusion_long',k) for k in fields})
    write(a.out/'summary.json',dict(status='complete',cohorts=result,seconds=seconds,
        peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,checkpoint=str(checkpoint),
        parameters=sum(p.numel() for p in model.parameters()),target_sampling=dict(temperature=.9,top_p=.9),
        draft_sampling=dict(temperature=.9,top_p=.9),widths=widths,rows=len(rows),
        scope='Development only; Algorithm 2 first P/F pair plus reuse of fixed verified branches; unchanged predictions'))
    print('Block joint common histories complete',len(rows),flush=True)


if __name__=='__main__':main()
