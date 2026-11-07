"""Collect actual Position-candidate supervision and train small Feature drafts."""
import argparse
from contextlib import nullcontext
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def setup(args):
    os.chdir(args.project)
    sys.path[:0] = [str(args.source), str(args.source/'reference'),
                    str(args.source/'reference/runtime'), str(args.project)]
    import torch
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def context(args):
    from method_adapters import build_context
    return build_context(args.project, device='cuda')


def collect(args):
    import torch
    from state_draft import LayerTap, StateTargetGraph, state_capture
    from draft_a.pf_ranker_eval import cache_for, typed_kinds, position_state, policy
    from gpt_position_spec_decode_smoke import choose_position
    from bench_gaussiangpt_pf_block2_cow import truncate
    from collect_feature_teacher_from_s3 import policy_batch
    from method_adapters import load_method
    if (args.out/'data/metadata.json').exists():
        raise FileExistsError('Supervision already exists')
    ctx = context(args)
    target = ctx['target']
    tap = LayerTap(target)
    runtime = StateTargetGraph(target, tap, cache_capacity_buckets=(8192,16384))
    plan_path=args.selection_plan or args.project/'draft_a/runs/feature_target_full_v2_20261014/training_plan.json'
    plan = json.loads(plan_path.read_text())
    corpus = args.project/'draft_a/runs/rollout500_20261013'
    source_dir = args.project/'draft_a/data/teacher_sequences'
    old = load_method('pf_mlp', ctx).model.feature
    if not torch.equal(old.frozen_head, target.lm_head.weight[8000:12096]):
        raise ValueError('Feature output weights differ from Target')
    (args.out/'data').mkdir(parents=True,exist_ok=True)
    torch.save({'weight':old.frozen_head.detach().cpu()},args.out/'data/feature_weight.pt')
    rng = random.Random(20261105)
    chosen = {}
    split_counts=[('train', args.trajectories),
                  ('val', args.validation_trajectories or max(8, args.trajectories//8))]
    if args.evaluation_only:
        split_counts=[item for item in split_counts if item[0]=='val']
    for split, count in split_counts:
        names = list(plan[split+'_files'])
        rng.shuffle(names)
        chosen[split] = names[:count]
    selection=dict(files=chosen, original_plan=str(plan_path), seed=20261105)
    if 'source_metadata' in plan:
        selection['source_metadata']=plan['source_metadata']
    write(args.out/'data/selection.json', selection)
    collected = {}
    started = time.perf_counter()
    with torch.inference_mode(), tap.context(target), state_capture(tap), runtime.attention_context():
        import bench_gaussiangpt_pf_block2_cow as capture_module
        capture = capture_module.CaptureLastBlock(target)
        for split, names in chosen.items():
            chunks = []
            diagnostic = []
            unavailable = []
            for number, name in enumerate(names):
                path = source_dir/name
                if not path.exists():
                    matches = list(corpus.rglob(name))
                    if len(matches) != 1:
                        raise FileNotFoundError(name)
                    path = matches[0]
                record = torch.load(path, map_location='cpu', weights_only=False)
                if 'tokens' in record:
                    body = torch.as_tensor(record['tokens']).long().tolist()
                elif 'prompt' in record and 'suffix' in record:
                    body = list(record['prompt'])+list(record['suffix'])
                else:
                    raise ValueError('Saved history has no token sequence')
                if body and body[0] == 12096:
                    body = body[1:]
                if body and body[-1] == 12096:
                    body = body[:-1]
                if not body and args.evaluation_only:
                    unavailable.append(dict(path=str(path),reason='No complete pair before EOS'))
                    continue
                if len(body)%2 or not body:
                    raise ValueError('Expected complete scene pairs')
                tokens = [12096]+body
                usable = (len(body)//2)-1
                metadata=plan.get('source_metadata',{}).get(name,{})
                first_pair=max(0,int(metadata.get('prefix',0))//2-1)
                if usable <= first_pair or (not args.evaluation_only and usable < 2):
                    unavailable.append(dict(path=str(path),reason='No evaluable continuation state'))
                    continue
                selected = sorted(rng.sample(range(first_pair,usable),
                                            min(args.rows_per_trajectory,usable-first_pair)))
                if len(tokens)+8>16384:
                    raise ValueError('Training history exceeds Target capacity')
                cache = runtime.cache_for(target,len(tokens)+8,factory=cache_for)
                cursor = 0
                trajectory = []
                for pair_index in selected:
                    end = 2*pair_index+2
                    while cursor < end:
                        stop = min(end, cursor+128)
                        _, states = capture.run(target,cache,tokens[cursor:stop],
                                                typed_kinds(cursor, stop-cursor),'cuda')
                        cursor = stop
                    state = states[-1].clone()
                    prefix = tokens[:end+1]
                    online = position_state(prefix, ctx['ranks'])
                    candidate = choose_position(ctx['ranker'],online,prefix[-2])
                    if candidate is None:
                        continue
                    candidate = int(candidate[0])
                    pending = prefix[-1]
                    logits, _ = capture.run(target,cache,[pending,candidate],
                                            ['position','feature'],'cuda')
                    # Graph and ordinary paths share this publish interface.
                    if (pending < 8000 or pending >= 12096 or candidate >= 8000):
                        raise ValueError('Invalid typed inputs')
                    p_position = float(policy(logits[0])[candidate].item())
                    raw = logits[1,8000:12096].clone()
                    p = policy_batch(raw[None],.9,.9)[0]
                    _, previous_logits = old(state[-1024:][None],
                                             torch.tensor([pending-8000],device='cuda'),
                                             torch.tensor([candidate],device='cuda'))
                    q = policy_batch(previous_logits,.9,.9)[0]
                    overlap = float(torch.minimum(p,q).sum().item())
                    trajectory.append(dict(state=state.cpu(),previous_feature=pending-8000,
                        position=candidate,previous_position=prefix[-2],
                        logits=raw.cpu(),weight=p_position,source=number,pair_index=pair_index))
                    diagnostic.append((p_position,overlap,pair_index))
                    truncate(cache,end,end+2)
                if trajectory:
                    chunks.extend(trajectory)
                write(args.out/'progress.json',dict(phase='collect',split=split,
                      trajectories=number+1,total=len(names),rows=len(chunks),seconds=time.perf_counter()-started))
                print('collect',split,number+1,len(names),len(chunks),flush=True)
            if not chunks:
                raise RuntimeError('No Feature supervision')
            data = {key:torch.stack([r[key] for r in chunks]) if torch.is_tensor(chunks[0][key])
                    else torch.tensor([r[key] for r in chunks]) for key in chunks[0]}
            (args.out/'data').mkdir(parents=True,exist_ok=True)
            torch.save(data,args.out/'data'/f'{split}.pt')
            collected[split]=dict(rows=len(chunks),trajectories=len(names),
                unavailable_sources=unavailable,
                mlp_overlap=sum(x[1] for x in diagnostic)/len(diagnostic),
                mean_position_mass=sum(x[0] for x in diagnostic)/len(diagnostic),
                mean_joint_mass=sum(x[0]*x[1] for x in diagnostic)/len(diagnostic),
                ideal_advance_ratio=(2+2*sum(x[0] for x in diagnostic)/len(diagnostic))/(2+2*sum(x[0]*x[1] for x in diagnostic)/len(diagnostic)))
            collected[split]['history_ranges']={}
            for label,lower,upper in [('tokens_0_2048',0,1024),('tokens_2048_4096',1024,2048),('tokens_above_4096',2048,8192)]:
                subset=[x for x in diagnostic if lower<=x[2]<upper]
                if subset:
                    collected[split]['history_ranges'][label]=dict(rows=len(subset),
                        mean_position_mass=sum(x[0] for x in subset)/len(subset),
                        mlp_overlap=sum(x[1] for x in subset)/len(subset),
                        reachable_overlap=sum(x[0]*x[1] for x in subset)/max(sum(x[0] for x in subset),1e-12))
            del data, chunks
            gc.collect()
        capture.close()
    write(args.out/'data/metadata.json',dict(status='complete',layers=tap.layers,
          target=str(args.project/'checkpoints/gpt_vfront.ckpt'),
          feature=str(args.initial),gpu=torch.cuda.get_device_name(),torch=torch.__version__,
          policy=dict(temperature=.9,top_p=.9,fp32=True,tf32=False),
          history=args.history_label or ('Existing natural scene outputs; Ranker candidate paths' if args.selection_plan else 'Teacher-generated histories; Ranker candidate paths; all positions sampled across trajectory'),
          meaning='ideal_advance_ratio excludes EOS, changes of future histories, and all extra time; not a speed prediction',
          splits=collected,seconds=time.perf_counter()-started))


def statistics(model, data, *, ablate=False):
    import torch
    from collect_feature_teacher_from_s3 import policy_batch
    total = torch.zeros(5,device='cuda',dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for start in range(0,len(data['state']),256):
            sl=slice(start,start+256)
            state=data['state'][sl].cuda()
            if ablate and state.shape[-1] == 3072:
                state[:,:2048]=0
            _,logits=model(state,data['previous_feature'][sl].cuda(),data['position'][sl].cuda())
            teacher=data['logits'][sl].cuda().float()
            p=policy_batch(teacher,.9,.9)
            q=policy_batch(logits.float(),.9,.9)
            overlap=torch.minimum(p,q).sum(-1)
            w=data['weight'][sl].cuda()
            ce=-(teacher.softmax(-1)*logits.float().log_softmax(-1)).sum(-1)
            total+=torch.stack([overlap.sum(),(overlap*w).sum(),w.sum(),ce.sum(),torch.tensor(len(overlap),device='cuda')]).double()
    a,b,w,ce,n=total.cpu().tolist()
    return dict(overlap=a/n,reachable_overlap=b/max(w,1e-12),joint=b/n,raw_ce=ce/n,rows=int(n))


def train(args):
    import torch
    import torch.nn.functional as F
    from collect_feature_teacher_from_s3 import policy_batch
    from state_draft import make_draft
    from mlp_medusa_graph import ProposalForwardGraph
    torch.manual_seed(20261105)
    directory=args.out/'training'/(args.name or args.kind)
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    train_data=torch.load(args.out/'data/train.pt',map_location='cpu',weights_only=True)
    val=torch.load(args.out/'data/val.pt',map_location='cpu',weights_only=True)
    original_rows=len(train_data['state'])
    original_val_rows=len(val['state'])
    if args.extra_data:
        extra_train=torch.load(args.extra_data/'data/train.pt',map_location='cpu',weights_only=True)
        extra_val=torch.load(args.extra_data/'data/val.pt',map_location='cpu',weights_only=True)
        train_data={k:torch.cat((v,extra_train[k])) for k,v in train_data.items()}
        val={k:torch.cat((v,extra_val[k])) for k,v in val.items()}
        del extra_train,extra_val
    weight=torch.load(args.out/'data/feature_weight.pt',map_location='cpu',weights_only=True)['weight']
    token_weight=None
    if args.kind=='semantic':
        token_path=args.out/'data/token_embeddings.pt'
        if not token_path.exists():
            state=torch.load(args.project/'checkpoints/gpt_vfront.ckpt',map_location='cpu',weights_only=False)['state_dict']
            key=next(k for k in state if k.endswith('transformer.wte.weight'))
            torch.save(state[key].float().contiguous(),token_path)
            del state
        token_weight=torch.load(token_path,map_location='cpu',weights_only=True)
    model=make_draft(weight,args.kind,args.initial,token_weight).cuda().float()
    inputs=(val['state'][:8].cuda(),val['previous_feature'][:8].cuda(),val['position'][:8].cuda())
    if args.kind=='top':
        inputs=(inputs[0][:,-1024:],*inputs[1:])
    if args.initial == args.project/'draft_a/runs/feature_target_continue_20261014/selected.pt':
        from train_feature_state_distill import FeatureStateDraft
        original=FeatureStateDraft(weight).cuda().float().eval()
        original.load_state_dict(torch.load(args.initial,map_location='cpu',weights_only=False)['model_state'])
        with torch.no_grad():
            expected=original(inputs[0][:,-1024:],*inputs[1:])
            actual=model(*inputs)
            errors=[float((a-b).abs().max()) for a,b in zip(expected,actual)]
        if max(errors)>1e-6:
            raise ArithmeticError(f'Initial model differs: {errors}')
        del original
        write(directory/'initial_equivalence.json',dict(max_errors=errors))
    initial=statistics(model,val)
    best=initial['reachable_overlap']
    history=[dict(step=0,validation=initial)]
    torch.save(dict(kind=args.kind,model_state=model.state_dict(),validation=initial,step=0),directory/'selected.pt')
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    n=len(train_data['state'])
    device_data={k:v.cuda() for k,v in train_data.items()}
    started=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train()
        if args.extra_data:
            indices=torch.cat((torch.randint(original_rows,(128,),device='cuda'),
                               torch.randint(original_rows,n,(128,),device='cuda')))
        else:
            indices=torch.randint(n,(256,),device='cuda')
        s=device_data['state'][indices]
        previous=device_data['previous_feature'][indices]
        position=device_data['position'][indices]
        teacher=device_data['logits'][indices].float()
        w=device_data['weight'][indices].float()
        if not args.weighted:
            w=torch.ones_like(w)
        w=w/w.mean().clamp_min(1e-8)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            _,logits=model(s,previous,position)
        logits=logits.float()
        raw=-(teacher.softmax(-1)*logits.log_softmax(-1)).sum(-1)
        p=policy_batch(teacher,.9,.9)
        q=policy_batch(logits,.9,.9)
        tv=.5*(p-q).abs().sum(-1)
        loss=(w*(raw+2*tv)).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in optimizer.param_groups:
            group['lr']=args.lr*(.1+.9*(1+math.cos(math.pi*step/args.steps))/2)
        optimizer.step()
        if step%200==0 or step==args.steps:
            measured=statistics(model,val)
            item=dict(step=step,loss=float(loss),validation=measured,seconds=time.perf_counter()-started)
            history.append(item)
            if measured['reachable_overlap']>best:
                best=measured['reachable_overlap']
                torch.save(dict(kind=args.kind,model_state=model.state_dict(),validation=measured,
                                step=step,weighted=args.weighted,initial=str(args.initial)),directory/'selected.pt')
            write(directory/'history.json',history)
            write(args.out/'progress.json',dict(phase='train',kind=args.kind,**item))
            print(args.kind,item,flush=True)
    saved=torch.load(directory/'selected.pt',map_location='cpu',weights_only=False)
    model.load_state_dict(saved['model_state'])
    model.eval().requires_grad_(False)
    if args.extra_data:
        write(directory/'validation_domains.json',dict(
            original=statistics(model,{k:v[:original_val_rows] for k,v in val.items()}),
            natural=statistics(model,{k:v[original_val_rows:] for k,v in val.items()})))
    if args.kind in ('fusion','semantic'):
        write(directory/'input_ablation.json',dict(full=statistics(model,val),without_lower_states=statistics(model,val,ablate=True)))
    with torch.inference_mode():
        one=tuple(x[:1] for x in inputs)
        expected=tuple(x.clone() for x in model(*one))
        graph=ProposalForwardGraph(model,'forward',method_kind='mlp')
        with graph.context():
            actual=model(*one)
            errors=[float((a-b).abs().max()) for a,b in zip(expected,actual)]
            for _ in range(20):
                model(*one)
            torch.cuda.synchronize()
            begin=time.perf_counter()
            for _ in range(1000):
                model(*one)
            torch.cuda.synchronize()
            latency_ms=1000*(time.perf_counter()-begin)/1000
        if max(errors)>1e-6:
            raise ArithmeticError(f'Draft graph differs: {errors}')
        write(directory/'graph_equivalence.json',dict(max_errors=errors,stats=graph.stats(),
              batch1_graph_wall_ms=latency_ms,measured_calls=1000,
              scope='Draft graph call only, including Python invocation; not full generation'))
        graph.close()
    write(directory/'summary.json',dict(status='complete',kind=args.kind,steps=args.steps,
        selected_step=saved['step'],validation=saved['validation'],initial=initial,
        parameters=sum(p.numel() for p in model.parameters()),seconds=time.perf_counter()-started,
        seed=20261105,weighted=args.weighted,lr=args.lr,initial_checkpoint=str(args.initial),
        extra_data=str(args.extra_data) if args.extra_data else None,
        sampling='half original and half additional rows per update' if args.extra_data else 'uniform row sampling'))


def analyze(args):
    import torch
    from state_draft import make_draft
    weight=torch.load(args.out/'data/feature_weight.pt',map_location='cpu',weights_only=True)['weight']
    tokens=torch.load(args.out/'data/token_embeddings.pt',map_location='cpu',weights_only=True)
    domains={'original':args.out/'data/val.pt','natural':args.extra_data/'data/val.pt'}
    result={}
    for name in ['top','fusion','top_long','fusion_long','semantic','top_capacity']:
        path=args.out/'training'/name/'selected.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=make_draft(weight,saved['kind'],path,tokens).cuda().float().eval().requires_grad_(False)
        own={}
        for domain,path in domains.items():
            data=torch.load(path,map_location='cpu',weights_only=True)
            own[domain]=statistics(model,data)
            own[domain]['sources']={}
            for source in data['source'].unique().tolist():
                mask=data['source']==source
                own[domain]['sources'][str(source)]=statistics(model,{k:v[mask] for k,v in data.items()})
        result[name]=own
        del model
    comparisons={}
    for domain in domains:
        comparisons[domain]={}
        for name,reference in [('fusion_long','top_capacity'),('fusion_long','top_long'),
                               ('fusion_long','fusion'),('semantic','fusion_long')]:
            left,right=result[name][domain]['sources'],result[reference][domain]['sources']
            differences=[left[k]['overlap']-right[k]['overlap'] for k in left]
            rng=random.Random(20261105)
            boot=sorted(sum(rng.choices(differences,k=len(differences)))/len(differences) for _ in range(4000))
            comparisons[domain][name+'_minus_'+reference]=dict(
                overlap_difference=sum(differences)/len(differences),ci95=[boot[99],boot[3899]],sources=len(differences))
    write(args.out/'matched_prediction_analysis.json',dict(models=result,comparisons=comparisons,
          scope='Same held-out histories and Ranker candidates; source-level prediction differences, not speed estimates'))
    from collections import defaultdict
    accounting={}
    for directory in sorted(args.out.glob('natural_dev_*')):
        if not directory.is_dir():
            continue
        status=json.loads((directory/'progress.json').read_text())
        if status['status']!='complete':
            raise RuntimeError('Natural comparison is still running')
        rows=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
        own={}
        for cohort in sorted({r['cohort'] for r in rows}):
            subset=[r for r in rows if r['cohort']==cohort]
            indexed={(r['pair_id'],r['method']):r for r in subset}
            own[cohort]={}
            for name in sorted({r['method'] for r in subset}-{'opt_target','ranker_only'}):
                sources=defaultdict(list)
                for row in subset:
                    if row['method']!=name:
                        continue
                    ref=indexed[(row['pair_id'],'ranker_only')]
                    a=row['generated_tokens']/row['counts']['rounds']
                    b=ref['generated_tokens']/ref['counts']['rounds']
                    c=row['seconds']/row['counts']['rounds']
                    d=ref['seconds']/ref['counts']['rounds']
                    joint=row['counts']['feature_accepted']/max(row['counts']['tree_verify_calls'],1)
                    sources[row['source']].append((math.log(a/b),math.log(c/d),joint))
                per_source=[tuple(sum(x[i] for x in values)/len(values) for i in range(3)) for values in sources.values()]
                means=[sum(x[i] for x in per_source)/len(per_source) for i in range(3)]
                own[cohort][name]=dict(advance_ratio=math.exp(means[0]),cost_per_round_ratio=math.exp(means[1]),
                    throughput_ratio=math.exp(means[0]-means[1]),joint_event_frequency=means[2],sources=len(sources))
        accounting[directory.name]=own
    write(args.out/'execution_accounting.json',dict(comparisons=accounting,
          scope='Source-equal request amortization against Position-only. Histories can differ; no isolated operation attribution'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['collect','train','analyze'])
    p.add_argument('--project',type=Path,default=Path('/root/workspace/GaussianGPT'))
    p.add_argument('--source',type=Path,default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    p.add_argument('--out',type=Path,default=Path('/root/autodl-tmp/pfspec_feature_states_20261105'))
    p.add_argument('--initial',type=Path)
    p.add_argument('--selection-plan',type=Path)
    p.add_argument('--extra-data',type=Path)
    p.add_argument('--name')
    p.add_argument('--trajectories',type=int,default=96)
    p.add_argument('--validation-trajectories',type=int)
    p.add_argument('--evaluation-only',action='store_true')
    p.add_argument('--history-label')
    p.add_argument('--rows-per-trajectory',type=int,default=128)
    p.add_argument('--kind',choices=['top','fusion','semantic','capacity'],default='fusion')
    p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--lr',type=float,default=3e-5)
    p.add_argument('--weighted',action='store_true')
    args=p.parse_args()
    if args.initial is None:
        args.initial=args.project/'draft_a/runs/feature_target_continue_20261014/selected.pt'
    args.out.mkdir(parents=True,exist_ok=True)
    setup(args)
    if args.phase=='collect':collect(args)
    elif args.phase=='train':train(args)
    else:analyze(args)


if __name__=='__main__':
    main()
