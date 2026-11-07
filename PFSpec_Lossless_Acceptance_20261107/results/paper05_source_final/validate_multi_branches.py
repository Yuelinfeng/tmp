"""Compare every Feature branch and selected KV path with ordinary Target calls."""
import argparse,json,sys
from pathlib import Path

sys.path[:0]=['/root/autodl-tmp/pfspec_feature_states_20261105/source',
 '/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04',
 '/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04/reference',
 '/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04/reference/runtime',
 '/root/workspace/GaussianGPT']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/root/workspace/GaussianGPT'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    p.add_argument('--width',type=int,default=16)
    p.add_argument('--draft-top-p',type=float,default=.9)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    from research import setup,context,write
    setup(a)
    import torch
    from policy_inference import configure
    configure()
    from state_draft import LayerTap,StateTargetGraph,state_capture,make_draft
    from draft_a.pf_feature_unified import MLPAdapter,_shared
    from draft_a.pf_ranker_eval import cache_for,typed_kinds
    from gpt_position_spec_decode_smoke import choose_position
    from bench_gaussiangpt_pf_block2_cow import fork_position_state
    from multi_runtime import MultiTargetTreeRuntime
    from greedy_coupling import candidates
    ctx=context(a);target=ctx['target'];shared=_shared();tap=LayerTap(target)
    checkpoint=Path('/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long/selected.pt')
    model=make_draft(target.lm_head.weight[8000:12096].detach(),'fusion',checkpoint).cuda().float().eval().requires_grad_(False)
    model._pfspec_draft_top_p=a.draft_top_p
    cases=json.loads(Path('/root/autodl-tmp/pfspec_feature_states_20261105/acceptance_study/development_inputs.json').read_text())['contexts']
    complete=next(r for r in cases if r['prefix']>0)['prompt']
    prompts=[complete[:3],complete[:257]]
    rows=[]

    def compare(label,left,right,probability=False):
        if left.shape!=right.shape:raise ArithmeticError('Branch output dimensions differ')
        mask=torch.isfinite(left);same=bool(torch.equal(mask,torch.isfinite(right)))
        difference=float((left[mask]-right[mask]).abs().max()) if bool(mask.any()) else 0.
        equal=bool(torch.allclose(left[mask],right[mask],atol=2e-5,rtol=2e-5))
        record=dict(label=label,finite_support_equal=same,maximum_error=difference,allclose=equal)
        if not probability:
            parts=[(left[begin:begin+1024],right[begin:begin+1024]) for begin in range(0,3072,1024)] if left.ndim==1 and left.numel()==3072 else [(left,right)]
            relative=[]
            for lpart,rpart in parts:
                lv,rv=lpart.double(),rpart.double()
                relative.append(dict(l2=float(torch.linalg.vector_norm(lv-rv)/torch.linalg.vector_norm(rv).clamp_min(1e-12)),
                    maximum=float((lv-rv).abs().max()/rv.abs().max().clamp_min(1e-12))))
            record['relative_errors']=relative
            equal=bool(torch.isfinite(left).all() and torch.isfinite(right).all()) and all(x['l2']<=2e-6 and x['maximum']<=2e-6 for x in relative)
        if probability:
            lp=shared.policy(left.reshape(-1));rp=shared.policy(right.reshape(-1))
            record.update(policy_tv=float((lp-rp).abs().sum()/2),policy_support_equal=bool(torch.equal(lp>0,rp>0)))
            equal=equal and record['policy_tv']<=1e-5 and record['policy_support_equal']
        rows.append(record)
        if not same or not equal:
            write(a.out,dict(status='failed',width=a.width,rows=rows,failure=label))
            raise ArithmeticError('Target branch differs: '+label)

    with torch.inference_mode(),tap.context(target),state_capture(tap):
        import bench_gaussiangpt_pf_block2_cow as capture_module
        for case,prompt in enumerate(prompts):
            for graph in (False,True):
                linear=StateTargetGraph(target,tap,cache_capacity_buckets=(1024,))
                MultiTargetTreeRuntime.maximum_width=a.width
                tree=MultiTargetTreeRuntime(target,linear,graph=graph)
                factory=lambda:linear.cache_for(target,1024,factory=cache_for)
                with linear.attention_context(enabled=graph):
                    cache=factory();capture=capture_module.CaptureLastBlock(target)
                    _,initial=capture.run(target,cache,prompt[:-1],typed_kinds(0,len(prompt)-1),'cuda')
                    previous=initial[-1].detach().clone();capture.close()
                    online=shared.position_state(prompt,ctx['ranks'])
                    first=choose_position(ctx['ranker'],online,prompt[-2]);shadow=fork_position_state(online);shadow.finalize(first[0])
                    second=choose_position(ctx['ranker'],shadow,first[0]);p1,p2=int(first[0]),int(second[0])
                    adapter=MLPAdapter(model,bf16=False);adapter.prepare(prompt,1024)
                    rng=torch.Generator(device='cuda').manual_seed(73301+case)
                    q=adapter.propose(prompt,p1,previous,rng);ids,_=candidates(q,a.width,rng)
                    features=[x+8000 for x in ids];before=int(cache.get_pos())
                    logits,states=tree.verify(cache,prompt[-1],p1,features,[p2]*len(features))
                    logits,states=logits.clone(),states.clone()
                    last_reference=None
                    for branch,feature in enumerate(features):
                        reference=cache_for(target,1024);cap=capture_module.CaptureLastBlock(target)
                        if reference is cache:raise RuntimeError('Reference must own separate KV storage')
                        cap.run(target,reference,prompt[:-1],typed_kinds(0,len(prompt)-1),'cuda')
                        path=[prompt[-1],p1,feature,p2]
                        ref_logits,ref_states=cap.run(target,reference,path,typed_kinds(before,4),'cuda')
                        for depth,node in enumerate([0,1,2+2*branch,3+2*branch]):
                            label=f'prefix{len(prompt)-1}_graph{int(graph)}_branch{branch}_depth{depth}'
                            compare(label+'_logits',logits[node],ref_logits[depth],True)
                            compare(label+'_states',states[node],ref_states[depth])
                        bonus=int(ref_logits[3,8000:12096].argmax())
                        next_state=fork_position_state(shadow);next_state.finalize(p2)
                        third=choose_position(ctx['ranker'],next_state,p2)
                        if third is not None:
                            prev=torch.tensor([bonus],device='cuda');position=torch.tensor([int(third[0])],device='cuda')
                            _,lq=model(states[3+2*branch].unsqueeze(0),prev,position)
                            _,rq=model(ref_states[3].unsqueeze(0),prev,position)
                            from collect_feature_teacher_from_s3 import policy_batch
                            if a.draft_top_p==1:
                                lp,rp=(lq.float()/.9).softmax(-1),(rq.float()/.9).softmax(-1)
                            else:
                                lp,rp=policy_batch(lq.float(),.9,a.draft_top_p),policy_batch(rq.float(),.9,a.draft_top_p)
                            tv=float((lp-rp).abs().sum()/2);support=bool(torch.equal(lp>0,rp>0))
                            rows.append(dict(label=label+'_next_draft',policy_tv=tv,policy_support_equal=support,maximum_error=float((lq-rq).abs().max())))
                            if tv>1e-5 or not support:
                                write(a.out,dict(status='failed',width=a.width,rows=rows,failure='Next draft probabilities differ'))
                                raise ArithmeticError('Next draft probabilities differ')
                        cap.close()
                        last_reference=reference
                    branch=len(features)-1;path=[0,1,2+2*branch,3+2*branch]
                    write(a.out,dict(status='running',width=a.width,rows=rows,completed_branch_context=case,
                                    graph=graph,phase='selected KV and next Target output'))
                    tree.commit(cache,path)
                    end=before+4
                    compare('selected_K',cache.k_cache[:,:,:end],last_reference.k_cache[:,:,:end])
                    compare('selected_V',cache.v_cache[:,:,:end],last_reference.v_cache[:,:,:end])
                    if int(cache.get_pos())!=int(last_reference.get_pos()):raise ArithmeticError('Selected cache length differs')
                    if not torch.equal(cache.token_cache[:,:end],last_reference.token_cache[:,:end]):raise ArithmeticError('Selected token path differs')
                    cap=capture_module.CaptureLastBlock(target)
                    follow=prompt[-1:]
                    ll,ls=cap.run(target,cache,follow,typed_kinds(end,1),'cuda');ll,ls=ll.clone(),ls.clone()
                    rl,rs=cap.run(target,last_reference,follow,typed_kinds(end,1),'cuda')
                    compare('next_logits',ll,rl,True);compare('next_states',ls,rs)
                    cap.close();adapter.close()
                del tree,linear,cache,last_reference
                torch.cuda.empty_cache()
    write(a.out,dict(status='complete',width=a.width,draft_top_p=a.draft_top_p,cases=len(prompts),rows=rows,
        maximum_error=max(r['maximum_error'] for r in rows),
        maximum_policy_tv=max(r.get('policy_tv',0) for r in rows),
        scope='Every branch at two prefix lengths, ordinary and graph computation; selected KV and next Target call'))
    print('Multi-branch Target validation complete',len(rows),flush=True)


if __name__=='__main__':main()
