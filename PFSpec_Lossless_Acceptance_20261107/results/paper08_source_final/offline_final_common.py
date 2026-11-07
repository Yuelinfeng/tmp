"""Compare selected methods on the same formal W1 histories and Target policies."""
import argparse,json,random,sys,time
from pathlib import Path


def main():
    root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
    old=Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--records',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--development-validation',action='store_true')
    parser.add_argument('--rows-per-request',type=int,default=128)
    args=parser.parse_args()
    selected=json.loads(args.selection.read_text())
    if selected.get('status')!='complete':raise ValueError('Method selection is unfinished')
    best=selected['method']
    if best not in ['diffusion_full_w64','diffusion_univer_w16','diffusion_univer_w32','diffusion_univer_w64']:
        raise ValueError('Unsupported selected context-Draft method')
    candidate_width=int(selected.get('first_feature_candidates',best.rsplit('_w',1)[1]))
    conditional_tree='univer' in best
    args.project=Path('/root/workspace/GaussianGPT');args.source=shared
    sys.path[:0]=[str(old/'source'),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),str(args.project)]
    from research import setup,context,write
    setup(args)
    import torch
    from state_draft import LayerTap,StateTargetGraph,state_capture,make_draft
    from diffusion_draft import make_model
    from diffusion_inference import LiveMemory
    from types import SimpleNamespace
    from draft_a.pf_feature_unified import _shared
    from draft_a.pf_ranker_eval import cache_for,typed_kinds
    from gpt_position_spec_decode_smoke import choose_position
    from bench_gaussiangpt_pf_block2_cow import fork_position_state,truncate
    import bench_gaussiangpt_pf_block2_cow as capture_module
    from collect_feature_teacher_from_s3 import policy_batch
    from greedy_coupling import expected_acceptance
    from block_coupling import expected_joint
    from acceptance_metrics import source_average,paired_difference
    from report_multidraft import mean_interval
    torch.set_num_threads(1)
    ctx=context(args);target=ctx['target'];api=_shared()
    inputs=json.loads(args.inputs.read_text())['contexts']
    cases={c['pair_id']:c for c in inputs}
    records=[json.loads(line) for line in args.records.read_text().splitlines() if line.strip()]
    if len(records)!=len(cases) or {r['pair_id'] for r in records}!=set(cases):raise ValueError('W1 histories differ from prescribed inputs')
    if any(r['method']!='w1' for r in records):raise ValueError('Expected W1 histories only')
    for row in records:
        case=cases[row['pair_id']]
        if any(row[k]!=case[k] for k in ['source','seed','cohort','prefix']):raise ValueError('History identity differs')
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    weight=target.lm_head.weight[8000:12096].detach()
    saved=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    draft=make_model(saved['initial'],target.transformer.wte.weight.detach(),weight,
        layers=saved['config']['layers'],memory_tokens=saved['config']['memory_tokens'])
    draft.load_state_dict(saved['model_state']);draft=draft.cuda().float().eval().requires_grad_(False)
    w1=make_draft(weight,'top',args.project/'draft_a/runs/feature_target_continue_20261014/selected.pt').cuda().float().eval().requires_grad_(False)
    fusion=make_draft(weight,'fusion',old/'training/fusion_long/selected.pt').cuda().float().eval().requires_grad_(False)
    tap=LayerTap(target);rows=[];sources=[]
    work=dict(prefix_target_calls=0,prefix_target_tokens=0,conditional_target_calls=0,conditional_target_tokens=0,ranker_calls=0,draft_calls=0,draft_input_tokens=0,draft_context_tokens=0)
    started=time.perf_counter()
    with torch.inference_mode(),tap.context(target),state_capture(tap),(args.out/'state_metrics.jsonl').open('w') as stream:
        for request_index,record in enumerate(records):
            case=cases[record['pair_id']]
            token_path=args.records.parent/'tokens'/f'{record["pair_id"]}_w1.pt'
            history=torch.load(token_path,map_location='cpu',weights_only=False)
            prompt=[int(t) for t in history['prompt']];suffix=[int(t) for t in history['suffix']]
            if prompt!=case['prompt'] or len(suffix)!=record['generated_tokens']:raise ValueError('Saved history differs from raw record')
            if bool(suffix and suffix[-1]==12096)!=record['eos']:raise ValueError('EOS differs from raw record')
            api.validate_sequence(prompt,suffix,ctx['ranks'])
            known=prompt+suffix
            body_length=len(known)-1-int(bool(known and known[-1]==12096))
            if body_length%2:raise ValueError('Incomplete saved P/F sequence')
            first_pair=max(0,case['prefix']//2-1)
            usable=body_length//2-1
            rng=random.Random(20261107+request_index)
            indexes=sorted(rng.sample(range(first_pair,usable),min(args.rows_per_request,max(0,usable-first_pair))))
            source=dict(pair_id=record['pair_id'],source=case['source'],cohort=case['cohort'],prescribed_states=len(indexes),evaluated_states=0,unavailable=[])
            linear=StateTargetGraph(target,tap,cache_capacity_buckets=(32768,))
            cache=cache_for(target,32768)
            capture=capture_module.CaptureLastBlock(target)
            adapter=SimpleNamespace(feature=draft,context_tokens=0)
            memory=LiveMemory(adapter,None)
            with linear.attention_context(enabled=False):
                for pair_index in indexes:
                    anchor=2*pair_index+2
                    while int(cache.get_pos())<anchor:
                        start=int(cache.get_pos());tokens=known[start:min(anchor,start+64)]
                        _,states=capture.run(target,cache,tokens,typed_kinds(start,len(tokens)),'cuda')
                        memory.store(start,states)
                        work['prefix_target_calls']+=1;work['prefix_target_tokens']+=len(tokens)
                    prefix=known[:anchor+1]
                    online=api.position_state(prefix,ctx['ranks'])
                    first=choose_position(ctx['ranker'],online,prefix[-2]);work['ranker_calls']+=1
                    if first is None:
                        source['unavailable'].append(dict(history_pair_index=pair_index,reason='No first Position proposal'))
                        continue
                    shadow=fork_position_state(online);shadow.finalize(first[0])
                    second=choose_position(ctx['ranker'],shadow,first[0]);work['ranker_calls']+=1
                    if second is None:
                        source['unavailable'].append(dict(history_pair_index=pair_index,reason='No full-round second Position proposal'))
                        continue
                    p1,p2=int(first[0]),int(second[0]);state=memory.values[anchor-1]
                    block=memory.inputs(prefix,p1,p2,state)
                    prediction=draft(**block).float()
                    qbest=(prediction/.9).softmax(-1)[0,0]
                    _,wl=w1(state[None],torch.tensor([prefix[-1]-8000],device='cuda'),torch.tensor([p1],device='cuda'))
                    _,fl=fusion(state[None],torch.tensor([prefix[-1]-8000],device='cuda'),torch.tensor([p1],device='cuda'))
                    qw=policy_batch(wl.float(),.9,.9)[0];qf=policy_batch(fl.float(),.9,.9)[0]
                    logits,_=capture.run(target,cache,[prefix[-1],p1],typed_kinds(anchor,2),'cuda')
                    work['conditional_target_calls']+=1;work['conditional_target_tokens']+=2
                    raw_teacher=logits[1,8000:12096].clone()
                    p=api.policy(logits[1])[8000:12096];c1=float(api.policy(logits[0])[p1])
                    for method,q,raw,width in [('w1',qw,wl[0],1),('fusion_long',qf,fl[0],1),(best,qbest,prediction[0,0],candidate_width)]:
                        overlap=float(torch.minimum(p,q).sum())
                        joint=float(expected_joint(c1,p,q,width)) if method==best and conditional_tree else c1*float(expected_acceptance(p,q,width))
                        mass=joint/c1 if c1>0 else None
                        row=dict(method=method,source=case['source'],cohort=case['cohort'],pair_id=f'{case["pair_id"]}:{pair_index}',request_pair_id=case['pair_id'],seed=case['seed'],prefix=case['prefix'],history_pair_index=pair_index,
                            position_candidate=p1,second_position_candidate=p2,position_probability=c1,feature_overlap=overlap,feature_tv=1-overlap,
                            feature_set_acceptance=mass,pair_joint_probability=joint,
                            raw_teacher_cross_entropy=float(-(raw_teacher.softmax(-1)*raw.log_softmax(-1)).sum()),
                            target_top1_agreement=int(p.argmax()==q.argmax()),feature_candidates=width,
                            target_feature_support=int((p>0).sum()),draft_feature_support=int((q>0).sum()))
                        rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n')
                    truncate(cache,anchor,anchor+2)
                    work['draft_calls']+=3;work['draft_input_tokens']+=7
                    source['evaluated_states']+=1
                    stream.flush()
            work['draft_context_tokens']+=adapter.context_tokens
            sources.append(source)
            capture.close();del cache,linear,memory
            torch.cuda.empty_cache()
            write(args.out/'progress.json',dict(status='running',completed_requests=request_index+1,total_requests=len(records),rows=len(rows),work=work,seconds=time.perf_counter()-started))
    fields=['feature_overlap','feature_tv','feature_set_acceptance','pair_joint_probability','position_probability','raw_teacher_cross_entropy','target_top1_agreement']
    cohorts={}
    for cohort in ['bos','completion']:
        own=[r for r in rows if r['cohort']==cohort]
        cohorts[cohort]={}
        for method in ['w1','fusion_long',best]:
            chosen=[r for r in own if r['method']==method]
            cohorts[cohort][method]=dict(states=len(chosen),source_means={f:source_average(chosen,f) for f in fields},source_mean_intervals={f:mean_interval(chosen,f) for f in fields},
                paired_differences={other:{f:paired_difference(own,method,other,f) for f in fields} for other in ['w1','fusion_long',best] if other!=method})
    write(args.out/'summary.json',dict(status='complete',methods=['w1','fusion_long',best],rows=len(rows),requests=len(records),sources=sources,cohorts=cohorts,work=work,seconds=time.perf_counter()-started,
        scope='Development validation' if args.development_validation else 'Formal W1 continuation histories, used only after complete method selection',
        sampling=dict(rows_per_request=args.rows_per_request,seed=20261107,prefix_chunk_tokens=64,terminal_pair_excluded=True),
        target_sampling=dict(temperature=.9,top_p=.9),draft_sampling={best:dict(temperature=.9,top_p=1.)},
        history_scope='All methods use identical committed states, actual causal Position proposals, and one ordinary full conditional Target policy; unavailable full-round states remain listed',
        feature_scope='First Feature acceptance from the actual coupling; zero first-Position probability remains in joint rows and is undefined for Feature conditional acceptance',
        cross_entropy_scope='Raw teacher soft-label CE over all Feature tokens; overlap uses each actual inference policy',
        cost_scope='Additional Target prefix and conditional work is measured separately from online generation'))
    print('Common-history evaluation complete',len(rows),flush=True)


if __name__=='__main__':main()
