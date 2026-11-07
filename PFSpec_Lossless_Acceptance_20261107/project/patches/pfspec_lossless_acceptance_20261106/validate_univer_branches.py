"""Compare independent two-Feature candidates, states and selected KV paths."""
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
    p.add_argument('--draft-top-p',type=float,default=1.)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--eager-only',action='store_true')
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
    from univer_tree_runtime import UniVerTargetTreeRuntime as LongTargetTreeRuntime
    from diffusion_draft import make_model
    from greedy_coupling import candidates
    ctx=context(a);target=ctx['target'];shared=_shared();tap=LayerTap(target)
    checkpoint=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash/training/diffusion_context_w64/selected.pt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=make_model(saved['initial'],target.transformer.wte.weight.detach(),target.lm_head.weight[8000:12096].detach().clone(),
                     layers=saved['config']['layers'],memory_tokens=saved['config']['memory_tokens'])
    model.load_state_dict(saved['model_state'])
    model=model.cuda().float().eval().requires_grad_(False)
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
            active=(lp>0)|(rp>0)
            active_equal=bool(torch.allclose(left.reshape(-1)[active],right.reshape(-1)[active],atol=2e-5,rtol=2e-5))
            record.update(policy_tv=float((lp-rp).abs().sum()/2),policy_support_equal=bool(torch.equal(lp>0,rp>0)),
                active_logit_allclose=active_equal,
                active_logit_maximum_error=float((left.reshape(-1)[active]-right.reshape(-1)[active]).abs().max()) if bool(active.any()) else 0.)
            # Raw scores remain recorded; effective Target scores retain the same tolerance.
            equal=active_equal and record['policy_tv']<=1e-5 and record['policy_support_equal']
        rows.append(record)
        if not same or not equal:
            write(a.out,dict(status='failed',width=a.width,rows=rows,failure=label))
            raise ArithmeticError('Target branch differs: '+label)

    with torch.inference_mode(),tap.context(target),state_capture(tap):
        import bench_gaussiangpt_pf_block2_cow as capture_module
        for case,prompt in enumerate(prompts):
            for graph in ((False,) if a.eager_only else (False,True)):
                linear=StateTargetGraph(target,tap,cache_capacity_buckets=(1024,))
                tree=LongTargetTreeRuntime(target,linear,graph=graph,max_width=a.width)
                factory=lambda:linear.cache_for(target,1024,factory=cache_for)
                with linear.attention_context(enabled=graph):
                    cache=factory();capture=capture_module.CaptureLastBlock(target)
                    _,initial=capture.run(target,cache,prompt[:-1],typed_kinds(0,len(prompt)-1),'cuda')
                    previous=initial[-1].detach().clone();capture.close()
                    online=shared.position_state(prompt,ctx['ranks'])
                    first=choose_position(ctx['ranker'],online,prompt[-2]);shadow=fork_position_state(online);shadow.finalize(first[0])
                    second=choose_position(ctx['ranker'],shadow,first[0]);p1,p2=int(first[0]),int(second[0])
                    third_state=fork_position_state(shadow);third_state.finalize(p2)
                    third=choose_position(ctx['ranker'],third_state,p2)
                    if third is None:raise ValueError('Native branch input has no third Position')
                    p3=int(third[0]);before=int(cache.get_pos())
                    length=min(before,64)
                    memory=torch.zeros(1,64,3072,device='cuda');memory[0,-length:]=initial[-length:]
                    logits_draft=model(memory=memory,memory_positions=torch.arange(before-64,before,device='cuda').clamp_min(0)[None],
                        valid=(torch.arange(64,device='cuda')>=64-length)[None],ends=torch.tensor([before],device='cuda'),
                        previous_feature=torch.tensor([prompt[-1]-8000],device='cuda'),p1=torch.tensor([p1],device='cuda'),
                        p2=torch.tensor([p2],device='cuda'),state=previous[None])
                    q=(logits_draft.float()/.9).softmax(-1)[0]
                    rng=torch.Generator(device='cuda').manual_seed(73301+case)
                    ids,_=candidates(q[0],a.width,rng)
                    ids2=[candidates(q[1],4,rng)[0] for _ in ids]
                    features=[x+8000 for x in ids];features2=[[x+8000 for x in row] for row in ids2]
                    logits,states=tree.verify(cache,prompt[-1],p1,features,[p2]*len(features),
                        second_features=features2,third_position=p3)
                    logits,states=logits.clone(),states.clone()
                    last_reference=None
                    combinations=[(i,j,feature,feature2) for i,feature in enumerate(features) for j,feature2 in enumerate(features2[i])]
                    for branch,second_branch,feature,feature2 in combinations:
                        feature_node=2+10*branch
                        feature2_node=feature_node+2+2*second_branch
                        nodes=[0,1,feature_node,feature_node+1,feature2_node,feature2_node+1]
                        reference=cache_for(target,1024);cap=capture_module.CaptureLastBlock(target)
                        if reference is cache:raise RuntimeError('Reference must own separate KV storage')
                        cap.run(target,reference,prompt[:-1],typed_kinds(0,len(prompt)-1),'cuda')
                        path=[prompt[-1],p1,feature,p2,feature2,p3]
                        ref_logits,ref_states=cap.run(target,reference,path,typed_kinds(before,6),'cuda')
                        for depth,node in enumerate(nodes):
                            label=f'prefix{len(prompt)-1}_graph{int(graph)}_branch{branch}_second{second_branch}_depth{depth}'
                            compare(label+'_logits',logits[node],ref_logits[depth],True)
                            compare(label+'_states',states[node],ref_states[depth])
                        cap.close()
                        last_reference=reference
                    branch=len(features)-1;feature_node=2+10*branch;feature2_node=feature_node+2+2*(len(features2[branch])-1);path=[0,1,feature_node,feature_node+1,feature2_node,feature2_node+1]
                    write(a.out,dict(status='running',width=a.width,rows=rows,completed_branch_context=case,
                                    graph=graph,phase='selected KV and next Target output'))
                    tree.commit(cache,path)
                    end=before+6
                    compare('selected_K',cache.k_cache[:,:,:end],last_reference.k_cache[:,:,:end])
                    compare('selected_V',cache.v_cache[:,:,:end],last_reference.v_cache[:,:,:end])
                    if int(cache.get_pos())!=int(last_reference.get_pos()):raise ArithmeticError('Selected cache length differs')
                    if not torch.equal(cache.token_cache[:,:end],last_reference.token_cache[:,:end]):raise ArithmeticError('Selected token path differs')
                    cap=capture_module.CaptureLastBlock(target)
                    follow=prompt[-1:]
                    ll,ls=cap.run(target,cache,follow,typed_kinds(end,1),'cuda');ll,ls=ll.clone(),ls.clone()
                    rl,rs=cap.run(target,last_reference,follow,typed_kinds(end,1),'cuda')
                    compare('next_logits',ll,rl,True);compare('next_states',ls,rs)
                    cap.close()
                del tree,linear,cache,last_reference
                torch.cuda.empty_cache()
    write(a.out,dict(status='complete',width=a.width,draft_top_p=a.draft_top_p,cases=len(prompts),rows=rows,
        maximum_error=max(r['maximum_error'] for r in rows),
        maximum_policy_tv=max(r.get('policy_tv',0) for r in rows),
        graph_modes=[False] if a.eager_only else [False,True],
        scope='Every first/second Feature combination at two prefix lengths, explicit requested execution modes; all six full Target conditions, selected KV and next Target call'))
    print('Multi-branch Target validation complete',len(rows),flush=True)


if __name__=='__main__':main()
