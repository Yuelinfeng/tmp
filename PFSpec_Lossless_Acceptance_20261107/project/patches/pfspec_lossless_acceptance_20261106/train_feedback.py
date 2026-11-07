"""Replay true verification starts with accepted/rejected distillation objectives."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time


def load_feedback(directory, split):
    import torch
    rows, origins = [], []
    for path in sorted((directory / 'feedback_shards').glob('*.pt')):
        value = torch.load(path, map_location='cpu', weights_only=False)
        record = value['record']
        if record['split'] != split:
            continue
        origin = len(origins)
        selected = [r for r in value['rows'] if r['position_accepted']]
        if not selected:
            raise ValueError(f'No reached Feature states for {record["source"]}')
        origins.append(dict(source=record['source'], cohort=record['cohort'], rows=len(selected),
                            sampling_checkpoint='fusion_long', shard=str(path)))
        rows.extend(dict(r, source_index=origin) for r in selected)
    if not rows:
        raise ValueError(f'Empty {split} feedback')
    data = {}
    for key in ('state','logits'):
        data[key] = torch.stack([r[key] for r in rows])
    for key in ('previous_feature','position','source_index','feature_accepted'):
        data[key] = torch.tensor([r[key] for r in rows])
    data['weight'] = torch.tensor([r['weight'] for r in rows], dtype=torch.float32)
    return data, origins


def source_mean(values, sources):
    import torch
    return torch.stack([values[sources == source].mean() for source in sources.unique()]).mean()


def measure(model, data, draft_top_p=.9):
    import torch
    from collect_feature_teacher_from_s3 import policy_batch
    parts = []
    model.eval()
    with torch.no_grad():
        for begin in range(0, len(data['state']), 128):
            sl = slice(begin,begin+128)
            _, logits = model(data['state'][sl].cuda(), data['previous_feature'][sl].cuda(),
                              data['position'][sl].cuda())
            p = policy_batch(data['logits'][sl].cuda().float(), .9, .9)
            q = (logits.float()/.9).softmax(-1) if draft_top_p==1 else policy_batch(logits.float(), .9, draft_top_p)
            overlap = torch.minimum(p,q).sum(-1).cpu()
            parts.append(overlap)
    values = torch.cat(parts)
    sources = data['source_index']
    return dict(overlap=float(source_mean(values,sources)),
                joint=float(source_mean(values*data['weight'],sources)),
                reached_rows=len(values), sources=int(sources.unique().numel()),
                per_source={str(int(s)):dict(overlap=float(values[sources==s].mean()),
                    joint=float((values*data['weight'])[sources==s].mean()))
                    for s in sources.unique()})


def feedback_loss(logits, teacher, accepted, mode, smoothing, draft_top_p=.9, eta=3.):
    import torch
    from collect_feature_teacher_from_s3 import policy_batch
    p = policy_batch(teacher.float(), .9, .9).detach()
    q = (logits.float()/.9).softmax(-1) if draft_top_p==1 else policy_batch(logits.float(), .9, draft_top_p)
    if mode in ('lk_log','lk_adaptive'):
        if draft_top_p!=1:
            raise ValueError('This LK implementation uses the actual full-support draft policy')
        log_q=(logits.float()/.9).log_softmax(-1)
        log_alpha=torch.logsumexp(torch.minimum(p.log(),log_q),-1)
        if mode=='lk_log':
            return -log_alpha.mean()
        alpha=log_alpha.exp()
        mixing=torch.exp(-eta*alpha.detach().mean())
        forward=torch.special.xlogy(p,p).sum(-1)-(p*log_q).sum(-1)
        return (mixing*forward+(1-mixing)*(1-alpha)).mean()
    if mode == 'ce_tv':
        ce = -(teacher.float().softmax(-1) * logits.float().log_softmax(-1)).sum(-1)
        tv = .5*(p-q).abs().sum(-1)
        return (ce+2*tv).mean()
    if mode == 'split_full_kl':
        log_p=(teacher.float()/.9).log_softmax(-1).detach()
        log_q=(logits.float()/.9).log_softmax(-1)
        forward=(log_p.exp()*(log_p-log_q)).sum(-1)
        reverse=(log_q.exp()*(log_q-log_p)).sum(-1)
        groups=[]
        if bool(accepted.any()):
            groups.append(forward[accepted].mean())
        if bool((~accepted).any()):
            groups.append(reverse[~accepted].mean())
        return torch.stack(groups).mean()
    # The two auxiliary policies have full support for finite KL values.
    # Actual decoding keeps the original temperature/top-p policy unchanged.
    full_p = (teacher.float()/.9).softmax(-1).detach()
    full_q = (logits.float()/.9).softmax(-1)
    p_train = (1-smoothing)*p+smoothing*full_p
    q_train = (1-smoothing)*q+smoothing*full_q
    log_p = p_train.clamp_min(1e-30).log()
    log_q = q_train.clamp_min(1e-30).log()
    forward = (p_train*(log_p-log_q)).sum(-1)
    reverse = (q_train*(log_q-log_p)).sum(-1)
    if mode == 'forward_kl':
        return forward.mean()
    groups = []
    if bool(accepted.any()):
        groups.append(forward[accepted].mean())
    if bool((~accepted).any()):
        groups.append(reverse[~accepted].mean())
    return torch.stack(groups).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--feedback', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--mode', choices=['split_kl','ce_tv','forward_kl','split_full_kl','lk_log','lk_adaptive'], required=True)
    parser.add_argument('--initial', type=Path, default=Path('/root/autodl-tmp/pfspec_feature_states_20261105/training/fusion_long/selected.pt'))
    parser.add_argument('--project', type=Path, default=Path('/root/workspace/GaussianGPT'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    parser.add_argument('--previous-source', type=Path, default=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source'))
    parser.add_argument('--steps', type=int, default=1600)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--smoothing', type=float, default=1e-6)
    parser.add_argument('--draft-top-p',type=float,default=.9)
    parser.add_argument('--eta',type=float,default=3.)
    args = parser.parse_args()
    sys.path[:0] = [str(args.previous_source),str(args.source),str(args.source/'reference'),
                    str(args.source/'reference/runtime'),str(args.project)]
    from research import setup, write
    setup(args)
    import torch
    from state_draft import make_draft
    torch.manual_seed(args.seed)
    output = args.out / 'training' / args.name
    if output.exists():
        raise FileExistsError(output)
    if not 0 < args.smoothing < 1:
        raise ValueError('KL training smoothing must lie strictly between zero and one')
    if not 0 < args.draft_top_p <= 1 or args.eta <= 0:
        raise ValueError('Invalid draft policy or adaptive loss coefficient')
    train, train_origins = load_feedback(args.feedback,'train')
    validation, val_origins = load_feedback(args.feedback,'val')
    if {r['source'] for r in train_origins} & {r['source'] for r in val_origins}:
        raise ValueError('Training and development source identities overlap')
    weight = torch.load(Path('/root/autodl-tmp/pfspec_feature_states_20261105/data/feature_weight.pt'),
                        map_location='cpu',weights_only=True)['weight']
    model = make_draft(weight,'fusion',args.initial).cuda().float()
    device_data = {k:v.cuda() for k,v in train.items()}
    source_counts = Counter(train['source_index'].tolist())
    sampling = torch.tensor([1/source_counts[int(s)] for s in train['source_index']],device='cuda')
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    output.mkdir(parents=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(kind='fusion', train_sources=train_origins, development_sources=val_origins,
                  train_rows=len(train['state']), development_rows=len(validation['state']),
                  policy=dict(temperature=.9,top_p=.9,fp32=True,tf32=False),
                  inference_sampling_changed=args.draft_top_p!=.9, full_target_forward_added=False,
                  draft_policy=dict(temperature=.9,top_p=args.draft_top_p),
                  training_support_smoothing_only=args.smoothing if args.mode in ('split_kl','forward_kl') else None,
                  training_distribution='Actual Target top-p and actual full-softmax Draft' if args.mode in ('lk_log','lk_adaptive') else
                                        'Full softmax at temperature 0.9' if args.mode=='split_full_kl' else
                                        'Actual top-p with auxiliary full support' if args.mode!='ce_tv' else
                                        'Raw teacher CE and actual-policy TV',
                  input_exposure='Actual reached single-Feature starts sampled using fusion_long; '
                                 'no recursive draft-only hidden-state inputs',
                  feedback_partition='Original sampled candidate acceptance; residual samples are excluded',
                  selection='Source-equal joint probability on development feedback only')
    write(output/'config.json',config)
    initial = measure(model,validation,args.draft_top_p)
    best = initial['joint']
    history = [dict(step=0,validation=initial)]
    torch.save(dict(kind='fusion',model_state=model.state_dict(),validation=initial,step=0,
                    mode=args.mode,initial=str(args.initial),draft_top_p=args.draft_top_p),output/'selected.pt')
    torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train()
        ids=torch.multinomial(sampling,args.batch_size,replacement=True)
        optimizer.zero_grad(set_to_none=True)
        _,logits=model(device_data['state'][ids],device_data['previous_feature'][ids],
                        device_data['position'][ids])
        loss=feedback_loss(logits,device_data['logits'][ids],
                           device_data['feature_accepted'][ids].bool(),args.mode,args.smoothing,args.draft_top_p,args.eta)
        if not bool(torch.isfinite(loss)):
            raise ArithmeticError('Training loss is nonfinite')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        for group in optimizer.param_groups:
            group['lr']=args.lr*(.1+.9*(1+math.cos(math.pi*step/args.steps))/2)
        optimizer.step()
        if step%200==0 or step==args.steps:
            measured=measure(model,validation,args.draft_top_p)
            item=dict(step=step,loss=float(loss.detach()),validation=measured,
                      seconds=time.perf_counter()-started)
            history.append(item)
            if measured['joint'] > best:
                best=measured['joint']
                torch.save(dict(kind='fusion',model_state=model.state_dict(),validation=measured,
                                step=step,mode=args.mode,initial=str(args.initial),draft_top_p=args.draft_top_p),output/'selected.pt')
            write(output/'history.json',history)
            print(args.name,step,item,flush=True)
    torch.cuda.synchronize()
    torch.save(dict(kind='fusion',model_state=model.state_dict(),validation=history[-1]['validation'],
                    step=args.steps,mode=args.mode,initial=str(args.initial),draft_top_p=args.draft_top_p),output/'last.pt')
    write(output/'summary.json',dict(status='complete',initial=initial,best_joint=best,
        selected_step=torch.load(output/'selected.pt',map_location='cpu',weights_only=False)['step'],
        last_step=args.steps,
        train_seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        gpu=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
        inference_sampling_changed=args.draft_top_p!=.9,formal_data_used=False))


if __name__=='__main__':
    main()
