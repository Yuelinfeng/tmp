"""Train a context-reading masked block from authentic development-separated data."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
import torch
from diffusion_draft import make_model


def load_data(directory, split):
    memories,rows,origins = [],[],[]
    for path in sorted((directory/'feedback_shards').glob('*.pt')):
        saved = torch.load(path,map_location='cpu',weights_only=False)
        record = saved['record']
        if record['split']!=split:
            continue
        index = len(memories)
        memories.append(saved['memory'])
        rows.extend(dict(row,source_index=index) for row in saved['rows'])
        origins.append(dict(source=record['source'],cohort=record['cohort'],rows=len(saved['rows']),shard=str(path)))
    if not rows:
        raise ValueError('No masked-block examples')
    return dict(memories=memories,rows=rows,origins=origins)


def batch(data, indexes, length, device='cuda'):
    rows = [data['rows'][int(index)] for index in indexes]
    contexts,positions,valid = [],[],[]
    for row in rows:
        end = row['memory_end']
        start = max(0,end-length)
        context = data['memories'][row['source_index']][start:end]
        padding = length-len(context)
        contexts.append(torch.cat((torch.zeros(padding,3072),context),0))
        positions.append(torch.arange(end-length,end).clamp_min(0))
        valid.append(torch.arange(length)>=padding)
    inputs = dict(memory=torch.stack(contexts),memory_positions=torch.stack(positions),
        valid=torch.stack(valid),ends=torch.tensor([r['memory_end'] for r in rows]),
        previous_feature=torch.tensor([r['previous_feature'] for r in rows]),
        p1=torch.tensor([r['position'] for r in rows]),p2=torch.tensor([r['second_position'] for r in rows]),
        state=torch.stack([r['state'] for r in rows]))
    teacher = torch.stack([torch.stack((r['logits'],r['second_logits'])) for r in rows])
    return {k:v.to(device) for k,v in inputs.items()},teacher.to(device),rows


def set_acceptance(p,q,width=64):
    indexes = q.argsort(-1,descending=True)
    support = (q>0).sum(-1)
    fixed_count = torch.minimum(support-1,torch.full_like(support,width-1))
    selected = torch.arange(q.shape[-1],device=q.device)[None,:]<fixed_count[:,None]
    fixed = torch.zeros_like(q,dtype=torch.bool).scatter(1,indexes,selected)
    rest = q.masked_fill(fixed,0)
    rest = rest/rest.sum(-1,keepdim=True)
    return p.masked_fill(~fixed,0).sum(-1)+torch.minimum(p,rest).sum(-1)


def measure(model,data):
    from collect_feature_teacher_from_s3 import policy_batch
    values=[]
    model.eval()
    with torch.no_grad():
        for start in range(0,len(data['rows']),32):
            inputs,teacher,rows = batch(data,range(start,min(start+32,len(data['rows']))),model.memory_tokens)
            logits = model(**inputs)
            p = policy_batch(teacher.flatten(0,1),.9,.9).reshape_as(teacher)
            q = policy_batch(logits.flatten(0,1),.9,.9).reshape_as(logits)
            overlap = torch.minimum(p,q).sum(-1)
            first_set = set_acceptance(p[:,0],q[:,0])
            for i,row in enumerate(rows):
                values.append(dict(source=row['source_index'],cohort=data['origins'][row['source_index']]['cohort'],
                    first_overlap=float(overlap[i,0]),second_overlap=float(overlap[i,1]),
                    first_set=float(first_set[i]),first_joint=float(first_set[i])*row['weight']))
    per_source={}
    for source in sorted({v['source'] for v in values}):
        local=[v for v in values if v['source']==source]
        per_source[str(source)]={k:sum(v[k] for v in local)/len(local)
                               for k in ['first_overlap','second_overlap','first_set','first_joint']}
        per_source[str(source)]['cohort']=local[0]['cohort']
    cohorts={}
    for cohort in ['bos','completion']:
        local=[v for v in per_source.values() if v['cohort']==cohort]
        cohorts[cohort]={k:sum(v[k] for v in local)/len(local)
                         for k in ['first_overlap','second_overlap','first_set','first_joint']}
    return dict(per_source=per_source,cohorts=cohorts,
                selection_score=sum(v['first_joint'] for v in cohorts.values())/2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash'))
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--base-lr',type=float,default=3e-5)
    parser.add_argument('--memory-tokens',type=int,default=64)
    parser.add_argument('--layers',type=int,default=5)
    parser.add_argument('--evaluate-every',type=int,default=400)
    parser.add_argument('--seed',type=int,default=20261106)
    args=parser.parse_args()
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0]=[str(previous/'source'),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    from collect_feature_teacher_from_s3 import policy_batch
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    train=load_data(args.root,'train')
    validation=load_data(args.root,'val')
    if {v['source'] for v in train['origins']}&{v['source'] for v in validation['origins']}:
        raise ValueError('Training and development source identities overlap')
    tokens=torch.load(previous/'data/token_embeddings.pt',map_location='cpu',weights_only=True)
    if isinstance(tokens,dict):
        tokens=tokens['weight']
    weight=torch.load(previous/'data/feature_weight.pt',map_location='cpu',weights_only=True)['weight']
    initial=previous/'training/fusion_long/selected.pt'
    model=make_model(initial,tokens,weight,layers=args.layers,memory_tokens=args.memory_tokens).cuda().float()
    output=args.root/'training/diffusion_context_w64'
    output.mkdir(parents=True)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(initial=str(initial),kind='diffusion_context',feature_candidate_width=64,draft_top_p=.9,
        train_sources=train['origins'],development_sources=validation['origins'],
        train_rows=len(train['rows']),development_rows=len(validation['rows']),
        parameters=sum(p.numel() for p in model.parameters()),
        target_sampling=dict(temperature=.9,top_p=.9),draft_sampling=dict(temperature=.9,top_p=.9),
        prediction_scope='Two masked Features conditioned on committed Target context and two known Ranker Positions; online consumes the first Feature distribution',
        teacher_scope='Second Feature teacher conditions on the original fusion_long first Feature proposal; no future Target states enter Draft inputs',
        shared_target_embedding_elements=tokens.numel(),shared_feature_head_elements=weight.numel(),
        second_feature_loss_weight=math.exp(-1.),formal_sources_used=False)
    (output/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    core=[p for name,p in model.named_parameters() if not name.startswith('fusion.')]
    optimizer=torch.optim.AdamW([dict(params=core,lr=args.lr),dict(params=model.fusion.parameters(),lr=args.base_lr)],weight_decay=.01)
    counts=Counter(r['source_index'] for r in train['rows'])
    sampling=torch.tensor([1/counts[r['source_index']] for r in train['rows']])
    rng=torch.Generator().manual_seed(args.seed+1)
    best=-1.
    history=[]
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps+1):
        if step%args.evaluate_every==0 or step==args.steps:
            measured=measure(model,validation)
            entry=dict(step=step,seconds=time.perf_counter()-started,**measured)
            history.append(entry)
            (output/'curve.json').write_text(json.dumps(history,indent=2)+'\n')
            if measured['selection_score']>best:
                best=measured['selection_score']
                torch.save(dict(model_state=model.state_dict(),kind='fusion',draft_architecture='diffusion_context',initial=str(initial),
                    config=config,feature_candidate_width=64,draft_top_p=.9,step=step,validation=measured),output/'selected.pt')
            print(json.dumps(entry),flush=True)
        if step==args.steps:
            break
        model.train()
        chosen=torch.multinomial(sampling,args.batch_size,replacement=True,generator=rng)
        inputs,teacher,_=batch(train,chosen,args.memory_tokens)
        logits=model(**inputs)
        p=policy_batch(teacher.flatten(0,1),.9,.9).reshape_as(teacher).detach()
        q=policy_batch(logits.flatten(0,1),.9,.9).reshape_as(logits)
        ce=-(teacher.softmax(-1)*logits.log_softmax(-1)).sum(-1)
        tv=.5*(p-q).abs().sum(-1)
        losses=ce+2*tv
        loss=(losses[:,0]+math.exp(-1.)*losses[:,1]).mean()/(1+math.exp(-1.))
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError('Nonfinite masked-block loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not bool(torch.isfinite(gradient_norm)):
            raise ArithmeticError('Nonfinite masked-block gradient')
        schedule=min(1.,(step+1)/(args.steps*.04))*.5*(1+math.cos(math.pi*step/args.steps))
        optimizer.param_groups[0]['lr']=args.lr*schedule
        optimizer.param_groups[1]['lr']=args.base_lr*schedule
        optimizer.step()
    torch.cuda.synchronize()
    torch.save(dict(model_state=model.state_dict(),config=config,kind='fusion',draft_architecture='diffusion_context',
        feature_candidate_width=64,draft_top_p=.9,step=args.steps),output/'last.pt')
    (output/'result.json').write_text(json.dumps(dict(status='complete',steps=args.steps,
        seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        parameters=config['parameters'],selected_score=best),indent=2)+'\n')


if __name__=='__main__':
    main()
