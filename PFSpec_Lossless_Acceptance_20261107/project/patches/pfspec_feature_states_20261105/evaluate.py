"""Natural-generation comparison using the saved W1 sampling and Target backend."""
import argparse
from contextlib import ExitStack, nullcontext
import gc
import json
import math
from pathlib import Path
import random
import sys
import time

from research import write, setup, context


def summaries(rows):
    from collections import defaultdict
    result = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        own = [r for r in rows if r['cohort']==cohort]
        indexed={(r['pair_id'],r['method']):r for r in own}
        methods=sorted({r['method'] for r in own})
        cases=sorted({r['pair_id'] for r in own})
        groups={}
        for method in methods:
            a=[indexed[(case,method)] for case in cases]
            item=dict(requests=len(a),tokens=sum(r['generated_tokens'] for r in a),
                      seconds=sum(r['seconds'] for r in a),eos=sum(r['eos'] for r in a),comparisons={})
            item['pooled_tokens_per_second']=item['tokens']/item['seconds']
            for ref in ['opt_target','ranker_only','w1']:
                if ref not in methods or ref==method:
                    continue
                sources=defaultdict(list)
                times=defaultdict(list)
                for case in cases:
                    x,y=indexed[(case,method)],indexed[(case,ref)]
                    sources[x['source']].append(math.log((x['generated_tokens']/x['seconds'])/(y['generated_tokens']/y['seconds'])))
                    times[x['source']].append(math.log(y['seconds']/x['seconds']))
                def interval(values):
                    per_source=[sum(v)/len(v) for v in values.values()]
                    rng=random.Random(20261105)
                    boot=sorted(math.exp(sum(rng.choices(per_source,k=len(per_source)))/len(per_source)) for _ in range(4000))
                    return dict(ratio=math.exp(sum(per_source)/len(per_source)),ci95=[boot[99],boot[3899]],sources=len(per_source))
                item['comparisons'][ref]=dict(throughput=interval(sources),generation_request_time=interval(times))
            groups[method]=item
        result[cohort]=groups
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path('/root/workspace/GaussianGPT'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    p.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/pfspec_feature_states_20261105'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path)
    p.add_argument('--methods',default='ranker_only,w1,top,fusion')
    p.add_argument('--budget',type=int,default=16000)
    p.add_argument('--capacity',type=int,default=16384)
    p.add_argument('--validate',action='store_true')
    p.add_argument('--bos',type=int,default=6)
    p.add_argument('--completion',type=int,default=8)
    p.add_argument('--expected-gpu',help='Device name fragment for a specified evaluation device')
    args=p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    setup(args)
    import torch
    if args.expected_gpu:
        requested=''.join(c.lower() for c in args.expected_gpu if c.isalnum())
        actual=''.join(c.lower() for c in torch.cuda.get_device_name() if c.isalnum())
        if requested not in actual:
            raise ValueError(f'Requested GPU {args.expected_gpu}; found {torch.cuda.get_device_name()}')
    from acceptance_metrics import normalize, summarize as acceptance_summary
    from method_adapters import load_method, decode_method
    from dynamic_target_graph import DynamicTargetGraph
    from tree_runtime import TargetTreeRuntime
    from tree_decoder import decode
    from proposal_adapters import build_proposal
    from draft_a.pf_feature_unified import MLPAdapter
    from draft_a.pf_ranker_eval import cache_for, validate_sequence
    from state_draft import LayerTap, StateTargetGraph, state_capture, make_draft
    ctx=context(args)
    target=ctx['target']
    methods=args.methods.split(',')
    adapters={}
    taps={}
    checkpoints={}
    for method in methods:
        if method in ('opt_target','ranker_only'):
            adapters[method]=load_method('target' if method=='opt_target' else method,ctx)
        else:
            adapter=load_method('pf_mlp',ctx)
            if method!='w1':
                checkpoint=args.root/'training'/method/'selected.pt'
                saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
                model=make_draft(target.lm_head.weight[8000:12096].detach(),saved['kind'],checkpoint,
                                 target.transformer.wte.weight.detach()).cuda().float().eval().requires_grad_(False)
                adapter.model=MLPAdapter(model,bf16=False)
                checkpoints[method]=dict(path=str(checkpoint),kind=saved['kind'],step=saved['step'],validation=saved['validation'],
                                         parameters=sum(p.numel() for p in model.parameters()))
                if saved['kind'] in ('fusion','semantic'):
                    taps[method]=LayerTap(target)
            adapters[method]=adapter
    if args.inputs:
        cases=json.loads(args.inputs.read_text())['contexts']
    else:
        selection=json.loads((args.root/'data/selection.json').read_text())
        cases=[dict(pair_id=f'dev-bos-{69401+i}',source=f'dev-bos-{69401+i}',prefix=0,
                    seed=69401+i,prompt=[12096]) for i in range(args.bos)]
        for i,name in enumerate(selection['files']['val'][:args.completion]):
            path=args.project/'draft_a/data/teacher_sequences'/name
            if not path.exists():
                matches=list((args.project/'draft_a/runs/rollout500_20261013').rglob(name))
                if len(matches)!=1:raise FileNotFoundError(name)
                path=matches[0]
            record=torch.load(path,map_location='cpu',weights_only=False)
            body=record['tokens'].long().tolist()
            if body[0]==12096:body=body[1:]
            if body[-1]==12096:body=body[:-1]
            length=min(256,(len(body)//4)*2)
            cases.append(dict(pair_id='dev-completion-'+name[:-3],source=name,prefix=length,
                              seed=69501+i,prompt=[12096]+body[:length]))
    write(args.out/'plan.json',dict(methods=methods,contexts=cases,budget=args.budget,
        capacity=args.capacity,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        checkpoints=checkpoints,fp32=True,tf32=False,temperature=.9,top_p=.9,
        scope='development' if not args.inputs else 'supplied evaluation inputs',
        time_scope='generation, including prefix, first-use graphs, proposals, acceptance, recovery and cache operations; excludes VQ/render',
        target=str(args.project/'checkpoints/gpt_vfront.ckpt')))

    def runtime_for(method,graph=True):
        tap=taps.get(method)
        cls=StateTargetGraph if tap else DynamicTargetGraph
        runtime=cls(target,tap,cache_capacity_buckets=(args.capacity,)) if tap else cls(target,cache_capacity_buckets=(args.capacity,))
        if method in ('opt_target','ranker_only'):
            return dict(runtime=runtime,tree=None,proposal=None,graph=graph)
        return dict(runtime=runtime,tree=TargetTreeRuntime(target,runtime,graph=graph,max_width=1),
                    proposal=build_proposal(adapters[method],verify=False) if graph else None,graph=graph)

    def generate(method,case,budget,arm):
        adapter=adapters[method]
        if method in ('opt_target','ranker_only'):
            return decode_method(adapter,target,case['prompt'],budget,case['seed'],runtime=arm['runtime'])
        runtime=arm['runtime']
        factory=lambda core,length:runtime.cache_for(core,length,factory=cache_for)
        with ExitStack() as stack:
            if method in taps:
                stack.enter_context(taps[method].context(target))
                stack.enter_context(state_capture(taps[method]))
            if arm['proposal']:
                stack.enter_context(arm['proposal'].context())
            stack.enter_context(runtime.attention_context(enabled=arm['graph']))
            return decode(target,adapter.model,ctx['ranker'],ctx['ranks'],case['prompt'],budget,
                          case['seed'],tree_runtime=arm['tree'],width=1,cache_factory=factory)

    def counts(stats):
        return {k:v for k,v in stats.items() if isinstance(v,(int,bool,list,dict))
                and not k.lower().endswith('_rate')
                and not any(x in k.lower() for x in ['seconds','time','throughput','precision'])}

    started=time.perf_counter()
    with torch.inference_mode():
        if args.validate:
            reports=[]
            for method in methods:
                if method in ('opt_target','ranker_only'):
                    continue
                for case in [cases[0],cases[-1]]:
                    eager_arm=runtime_for(method,False)
                    eager_stats,eager_tokens=generate(method,case,128,eager_arm)
                    graph_arm=runtime_for(method,True)
                    graph_stats,graph_tokens=generate(method,case,128,graph_arm)
                    good=eager_tokens==graph_tokens and counts(eager_stats)==counts(graph_stats)
                    report=dict(method=method,pair_id=case['pair_id'],output_equal=eager_tokens==graph_tokens,
                                actions_equal=counts(eager_stats)==counts(graph_stats),tokens=len(graph_tokens),
                                runtime=graph_arm['runtime'].stats(),tree=graph_arm['tree'].stats())
                    reports.append(report)
                    write(args.out/'validation.json',dict(status='running',rows=reports))
                    graph_arm['proposal'].close()
                    del eager_arm,graph_arm
                    gc.collect();torch.cuda.empty_cache()
                    if not good:
                        raise ArithmeticError(f'Decode differs for {method}, {case["pair_id"]}')
            write(args.out/'validation.json',dict(status='complete',rows=reports))
            return
        # Discard short warmup runtimes; measured methods start with fresh ones.
        for method in methods:
            arm=runtime_for(method)
            generate(method,cases[0],16,arm)
            if arm['proposal']:arm['proposal'].close()
            del arm
            gc.collect();torch.cuda.empty_cache()
        rows=[]
        arms={}
        for i,case in enumerate(cases):
            shift=i%len(methods)
            order=methods[shift:]+methods[:shift]
            if (i//len(methods))%2:order=list(reversed(order))
            for method in order:
                created=method not in arms
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin=time.perf_counter()
                if created:arms[method]=runtime_for(method)
                budget=int(case.get('budget',args.budget))
                stats,suffix=generate(method,case,budget,arms[method])
                torch.cuda.synchronize()
                seconds=time.perf_counter()-begin
                validate_sequence(case['prompt'],suffix,ctx['ranks'])
                row=dict(cohort=case.get('cohort','bos' if case['prefix']==0 else 'completion'),
                         pair_id=case['pair_id'],source=case['source'],method=method,
                         seed=case['seed'],prefix=case['prefix'],seconds=seconds,
                         generated_tokens=len(suffix),eos=bool(suffix and suffix[-1]==12096),
                         runtime_created=created,counts=counts(stats),peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                         requested_tokens=budget)
                row['tokens_per_second']=len(suffix)/seconds
                row['termination']='eos' if row['eos'] else 'budget'
                row['acceptance']=normalize(row)
                with (args.out/'records.jsonl').open('a') as f:
                    f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
                (args.out/'tokens').mkdir(exist_ok=True)
                torch.save(dict(prompt=case['prompt'],suffix=suffix),args.out/'tokens'/f'{case["pair_id"]}_{method}.pt')
                rows.append(row)
                write(args.out/'progress.json',dict(status='running',completed=len(rows),expected=len(methods)*len(cases),last=row))
                print(case['pair_id'],method,len(suffix),round(seconds,3),flush=True)
        write(args.out/'summary.json',summaries(rows))
        write(args.out/'acceptance_summary.json',acceptance_summary([r['acceptance'] for r in rows]))
        write(args.out/'runtime.json',{m:dict(runtime=a['runtime'].stats(),
              tree=a['tree'].stats() if a['tree'] else None,
              proposal=a['proposal'].stats() if a['proposal'] else None) for m,a in arms.items()})
        for arm in arms.values():
            if arm['proposal']:arm['proposal'].close()
        write(args.out/'progress.json',dict(status='complete',completed=len(rows),expected=len(rows),seconds=time.perf_counter()-started))


if __name__=='__main__':
    main()
