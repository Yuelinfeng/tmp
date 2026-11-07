"""Measure both Feature hits under complete first-Feature Target conditions."""
import argparse
import json
from pathlib import Path
import sys
import time


def main():
    root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash')
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=root/'development_second_01')
    parser.add_argument('--project',type=Path,default=Path('/root/workspace/GaussianGPT'))
    parser.add_argument('--source',type=Path,default=shared)
    parser.add_argument('--limit-per-source',type=int)
    args=parser.parse_args()
    sys.path[:0]=[str(previous/'source'),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),str(args.project)]
    from research import setup,context,write
    setup(args)
    import torch
    from train_diffusion import load_data,batch
    from diffusion_draft import make_model
    from state_draft import LayerTap,StateTargetGraph,state_capture
    from draft_a.pf_feature_unified import _shared
    from draft_a.pf_ranker_eval import cache_for,typed_kinds
    from multi_runtime import MultiTargetTreeRuntime
    from greedy_coupling import candidates,expected_acceptance
    from acceptance_metrics import source_average
    from collect_feature_teacher_from_s3 import policy_batch
    import bench_gaussiangpt_pf_block2_cow as capture_module
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ctx=context(args);target=ctx['target'];policy=_shared().policy
    data=load_data(root,'val')
    saved=torch.load(root/'training/diffusion_context_w64/selected.pt',map_location='cpu',weights_only=False)
    model=make_model(saved['initial'],target.transformer.wte.weight.detach(),target.lm_head.weight[8000:12096].detach(),
                     layers=saved['config']['layers'],memory_tokens=saved['config']['memory_tokens'])
    model.load_state_dict(saved['model_state']);model=model.cuda().float().eval().requires_grad_(False)
    args.out.mkdir()
    rows=[];work=dict(prefix_target_calls=0,prefix_target_tokens=0,verification_calls=0,verification_nodes=0)
    begun=time.perf_counter();tap=LayerTap(target)
    with torch.inference_mode(),tap.context(target),state_capture(tap),(args.out/'state_metrics.jsonl').open('w') as stream:
        for source_index,origin in enumerate(data['origins']):
            indexes=sorted((i for i,r in enumerate(data['rows']) if r['source_index']==source_index),key=lambda i:data['rows'][i]['anchor'])
            if args.limit_per_source:
                indexes=indexes[:args.limit_per_source]
            collected=json.loads((root/'collection_tokens'/(Path(origin['shard']).stem+'.json')).read_text())
            known=collected['prompt']+collected['suffix']
            distributions={}
            for start in range(0,len(indexes),32):
                chosen=indexes[start:start+32];inputs,_,_=batch(data,chosen,model.memory_tokens)
                q=(model(**inputs).float()/.9).softmax(-1)
                distributions.update({i:q[j].clone() for j,i in enumerate(chosen)})
            linear=StateTargetGraph(target,tap,cache_capacity_buckets=(32768,))
            tree=MultiTargetTreeRuntime(target,linear,graph=False,max_width=64)
            cache=cache_for(target,32768);capture=capture_module.CaptureLastBlock(target)
            with linear.attention_context(enabled=False):
                for index in indexes:
                    example=data['rows'][index];anchor=int(example['anchor']);before=int(cache.get_pos())
                    if example['memory_end']!=anchor or known[anchor]-8000!=example['previous_feature'] or known[anchor-1]!=example['previous_position']:
                        raise ArithmeticError('Recorded model inputs differ from the committed token history')
                    while before<anchor:
                        tokens=known[before:min(anchor,before+64)]
                        capture.run(target,cache,tokens,typed_kinds(before,len(tokens)),'cuda')
                        work['prefix_target_calls']+=1;work['prefix_target_tokens']+=len(tokens);before=int(cache.get_pos())
                    q1,q2=distributions[index]
                    rng=torch.Generator(device='cuda').manual_seed(20261106+index)
                    ids,rest=candidates(q1,64,rng);p1=int(example['position']);p2=int(example['second_position'])
                    logits,_=tree.verify(cache,known[anchor],p1,[i+8000 for i in ids],[p2]*len(ids))
                    c1=policy(logits[0])[p1];p=policy(logits[1])[8000:12096]
                    random=ids[-1];a=torch.minimum(p[random]/rest[random],p.new_tensor(1.))
                    residual=(p-rest).clamp_min(0);mass=residual.sum()
                    residual=residual/mass if bool(mass>0) else torch.zeros_like(residual)
                    branch=(1-a)*residual[ids];branch[-1]+=a
                    conditional2=torch.stack([policy(logits[3+2*i])[8000:12096] for i in range(len(ids))])
                    c2=torch.stack([policy(logits[2+2*i])[p2] for i in range(len(ids))])
                    fixed2=torch.argsort(q2,descending=True,stable=True)[:3]
                    rest2=q2.clone();rest2[fixed2]=0;rest2/=rest2.sum()
                    alpha2=conditional2[:,fixed2].sum(-1)+torch.minimum(conditional2,rest2[None]).sum(-1)
                    j1_exact=c1*expected_acceptance(p,q1,64)
                    j1_draw=c1*branch.sum()
                    j2_draw=c1*(branch*c2*alpha2).sum()
                    if not bool(torch.isfinite(j2_draw)) or not 0<=float(j2_draw)<=float(j1_draw)+1e-6:
                        raise ArithmeticError('Both-Feature mass exceeds the first pair mass')
                    teacher=policy_batch(example['logits'][None].cuda().float(),.9,.9)[0]
                    row=dict(source=origin['source'],cohort=origin['cohort'],pair_id=f'{origin["source"]}:{anchor}',state_index=anchor,
                        first_feature_overlap=float(torch.minimum(p,q1).sum()),first_set_acceptance=float(expected_acceptance(p,q1,64)),
                        first_pair_joint_exact=float(j1_exact),first_pair_joint_conditional_draw=float(j1_draw),
                        both_features_joint_conditional_draw=float(j2_draw),
                        second_feature_conditional_on_first_hit=float(j2_draw/j1_draw) if bool(j1_draw>0) else None,
                        stored_teacher_feature_tv=float((teacher-p).abs().sum()/2),first_candidates=len(ids),second_fixed_candidates=3,
                        target_verification_nodes=2+2*len(ids))
                    rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                    work['verification_calls']+=1;work['verification_nodes']+=2+2*len(ids)
                    tree.commit(cache,[])
                    if len(rows)%64==0:write(args.out/'progress.json',dict(status='running',rows=len(rows),seconds=time.perf_counter()-begun,work=work,last=row))
            capture.close();del tree,linear,cache,distributions;torch.cuda.empty_cache()
            print('Second Feature common history source complete',origin['source'],len(indexes),flush=True)
    fields=['first_feature_overlap','first_set_acceptance','first_pair_joint_exact','first_pair_joint_conditional_draw',
            'both_features_joint_conditional_draw','second_feature_conditional_on_first_hit','stored_teacher_feature_tv']
    result=dict(status='complete',rows=len(rows),seconds=time.perf_counter()-begun,work=work,
        cohorts={cohort:{field:source_average([r for r in rows if r['cohort']==cohort],field) for field in fields} for cohort in ['bos','completion']},
        target_sampling=dict(temperature=.9,top_p=.9),draft_sampling=dict(temperature=.9,top_p=1.),
        scope='Same committed development histories and stored model inputs; full Target conditions for every first Feature and second Position',
        integration_scope='First joint probability is exact. Both-Feature probability integrates the second random candidate exactly and conditions on one independent first random-candidate draw per state. This is a Monte Carlo estimate over the first remaining-candidate law.',
        advance_scope='EOS and output budgets are measured by the separate natural generator; no six-token advance is inferred from these probabilities',
        information_scope='Additional ordinary Target prefix and four-token branch work is recorded; no future Target labels are Draft inputs')
    write(args.out/'summary.json',result);write(args.out/'progress.json',result)


if __name__=='__main__':main()
