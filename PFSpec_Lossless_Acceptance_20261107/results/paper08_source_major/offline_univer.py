"""Compare four methods on common histories with full second-Feature conditions."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from univer_coupling import allocation


def main():
    root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--limit-per-source',type=int,default=32)
    parser.add_argument('--project',type=Path,default=Path('/root/workspace/GaussianGPT'))
    parser.add_argument('--source',type=Path,default=shared)
    args=parser.parse_args()
    sys.path[:0]=[str(previous/'source'),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),str(args.project)]
    from research import setup,context,write
    setup(args)
    import torch
    from train_diffusion import load_data,batch
    from diffusion_draft import make_model
    from state_draft import LayerTap,StateTargetGraph,state_capture,make_draft
    from draft_a.pf_feature_unified import _shared
    from draft_a.pf_ranker_eval import cache_for,typed_kinds
    from gpt_position_spec_decode_smoke import choose_position
    from greedy_coupling import candidates,expected_acceptance
    from block_coupling import expected_joint
    from long_hybrid_runtime import HybridTargetTreeRuntime
    import long_hybrid_runtime
    from univer_tree_runtime import UniVerTargetTreeRuntime
    long_hybrid_runtime.LongTargetTreeRuntime=UniVerTargetTreeRuntime
    from acceptance_metrics import source_average,paired_difference
    from collect_feature_teacher_from_s3 import policy_batch
    import bench_gaussiangpt_pf_block2_cow as capture_module
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    ctx=context(args);target=ctx['target'];policy=_shared().policy
    data=load_data(root/'paper06_dflash','val')
    saved=torch.load(root/'paper06_dflash/training/diffusion_context_w64/selected.pt',map_location='cpu',weights_only=False)
    weight=target.lm_head.weight[8000:12096].detach()
    model=make_model(saved['initial'],target.transformer.wte.weight.detach(),weight,layers=5,memory_tokens=64)
    model.load_state_dict(saved['model_state']);model=model.cuda().float().eval().requires_grad_(False)
    w1=make_draft(weight,'top',Path('/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt')).cuda().float().eval().requires_grad_(False)
    fusion=make_draft(weight,'fusion',previous/'training/fusion_long/selected.pt').cuda().float().eval().requires_grad_(False)
    args.out.mkdir()
    rows=[];work=dict(prefix_target_calls=0,prefix_target_tokens=0,verification_calls=0,verification_nodes=0,ranker_calls=0)
    started=time.perf_counter();tap=LayerTap(target)
    with torch.inference_mode(),tap.context(target),state_capture(tap),(args.out/'state_metrics.jsonl').open('w') as stream:
        for source_index,origin in enumerate(data['origins']):
            indexes=sorted((i for i,x in enumerate(data['rows']) if x['source_index']==source_index),key=lambda i:data['rows'][i]['anchor'])[:args.limit_per_source]
            collected=json.loads((root/'paper06_dflash/collection_tokens'/(Path(origin['shard']).stem+'.json')).read_text())
            known=collected['prompt']+collected['suffix'];distributions={}
            for start in range(0,len(indexes),32):
                chosen=indexes[start:start+32];inputs,_,_=batch(data,chosen,64)
                q=(model(**inputs).float()/.9).softmax(-1)
                _,wl=w1(inputs['state'],inputs['previous_feature'],inputs['p1'])
                _,fl=fusion(inputs['state'],inputs['previous_feature'],inputs['p1'])
                qw=policy_batch(wl.float(),.9,.9);qf=policy_batch(fl.float(),.9,.9)
                for j,i in enumerate(chosen):distributions[i]=(q[j].clone(),qw[j].clone(),qf[j].clone())
            linear=StateTargetGraph(target,tap,cache_capacity_buckets=(32768,))
            tree=HybridTargetTreeRuntime(target,linear,graph=False,max_width=16)
            cache=cache_for(target,32768);capture=capture_module.CaptureLastBlock(target)
            with linear.attention_context(enabled=False):
                for index in indexes:
                    example=data['rows'][index];anchor=int(example['anchor']);before=int(cache.get_pos())
                    while before<anchor:
                        tokens=known[before:min(anchor,before+64)]
                        capture.run(target,cache,tokens,typed_kinds(before,len(tokens)),'cuda')
                        work['prefix_target_calls']+=1;work['prefix_target_tokens']+=len(tokens);before=int(cache.get_pos())
                    state=_shared().position_state(known[:anchor+1],ctx['ranks'])
                    p1=int(example['position']);p2=int(example['second_position'])
                    first=choose_position(ctx['ranker'],state,known[anchor-1]);work['ranker_calls']+=1
                    if first is None or int(first[0])!=p1:raise ArithmeticError('First Position differs from saved history')
                    state.finalize(p1)
                    second=choose_position(ctx['ranker'],state,p1);work['ranker_calls']+=1
                    if second is None or int(second[0])!=p2:raise ArithmeticError('Second Position differs from saved history')
                    state.finalize(p2)
                    third=choose_position(ctx['ranker'],state,p2);work['ranker_calls']+=1
                    qblock,qw,qf=distributions[index];q1,q2=qblock
                    rng=torch.Generator(device='cuda').manual_seed(20261110+index)
                    ids,rest=candidates(q1,16,rng)
                    ids2=[candidates(q2,4,rng)[0] for _ in ids]
                    kwargs={} if third is None else dict(second_features=[[x+8000 for x in row] for row in ids2],third_position=int(third[0]))
                    logits,_=tree.verify(cache,known[anchor],p1,[x+8000 for x in ids],[p2]*len(ids),**kwargs)
                    p=policy(logits[1])[8000:12096];c1=float(policy(logits[0])[p1])
                    pd=p.double().cpu().numpy();pd/=pd.sum()
                    rd=rest.double().cpu().numpy();rd/=rd.sum()
                    a=allocation(c1,pd,rd,ids[:-1],ids[-1])
                    second_joint=0.;reach=1.
                    if third is not None:
                        for branch,c in enumerate(a['conditional']):
                            node=2+10*branch
                            c2=float(policy(logits[node])[p2])
                            p2f=policy(logits[node+1])[8000:12096]
                            second_joint+=reach*float(expected_joint(c*c2,p2f,q2,4))
                            reach*=1-c
                    for method,q,width in [('w1',qw,1),('fusion_long',qf,1),('diffusion_full_w64',q1,64),('diffusion_univer_w16',q1,16)]:
                        overlap=float(torch.minimum(p,q).sum())
                        joint=float(expected_joint(c1,p,q,width)) if method=='diffusion_univer_w16' else c1*float(expected_acceptance(p,q,width))
                        row=dict(method=method,source=origin['source'],cohort=origin['cohort'],pair_id=f'{origin["source"]}:{anchor}',state_index=anchor,
                            feature_overlap=overlap,pair_joint_probability=joint,feature_conditional_set_acceptance=joint/c1 if c1>0 else None,
                            feature_set_acceptance=joint/c1 if c1>0 else None,
                            position_probability=c1,feature_candidates=width,second_feature_joint_sample=second_joint if method=='diffusion_univer_w16' else None,
                            actual_shared_target_nodes=162 if third is not None else 34,third_position_available=third is not None)
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    stream.flush();work['verification_calls']+=1;work['verification_nodes']+=162 if third is not None else 34
                    tree.commit(cache,[])
                    write(args.out/'progress.json',dict(status='running',rows=len(rows),work=work,seconds=time.perf_counter()-started,last_source=origin['source']))
            capture.close();del tree,linear,cache,distributions;torch.cuda.empty_cache()
    fields=['feature_overlap','pair_joint_probability','feature_set_acceptance','feature_conditional_set_acceptance','second_feature_joint_sample']
    cohorts={}
    for cohort in ['bos','completion']:
        own=[x for x in rows if x['cohort']==cohort]
        cohorts[cohort]={m:dict(source_means={f:source_average([x for x in own if x['method']==m],f) for f in fields},
            differences_vs_full={f:paired_difference(own,m,'diffusion_full_w64',f) for f in fields if f!='second_feature_joint_sample'})
            for m in ['w1','fusion_long','diffusion_full_w64','diffusion_univer_w16']}
    write(args.out/'summary.json',dict(status='complete',rows=len(rows),cohorts=cohorts,work=work,seconds=time.perf_counter()-started,
        scope='First 32 predeclared states per development source; saved common Draft inputs and one identical fresh Target distribution for every method',
        integration_scope='First pair is exact; second Feature joint integrates its remaining candidate and samples one first remaining candidate per state',
        advancement_scope='Actual emitted advancement is measured by natural generation',target_sampling=dict(temperature=.9,top_p=.9)))
    print('UniVer common histories complete',len(rows),flush=True)


if __name__=='__main__':main()
