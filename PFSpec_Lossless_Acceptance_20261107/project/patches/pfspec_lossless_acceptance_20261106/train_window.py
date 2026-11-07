"""Compare matched first-Feature CE/TV and complete-conditional P/F/P training."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time
import torch


def measure(model, data):
    from train_diffusion import batch
    from collect_feature_teacher_from_s3 import policy_batch
    from window_objective import window_values
    values = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(data['rows']), 32):
            inputs, teacher, rows = batch(data, range(start, min(start+32, len(data['rows']))), 64)
            logits = model(**inputs)[:, 0]
            p = policy_batch(teacher[:, 0].float(), .9, .9)
            q = (logits.float()/.9).softmax(-1)
            measured = window_values(p, q, rows)
            for i, row in enumerate(rows):
                origin = data['origins'][row['source_index']]
                values.append(dict(source=origin['source'], cohort=origin['cohort'],
                                   **{k: float(v[i]) for k, v in measured.items()}))
    fields = list(measured)
    sources = {}
    for source in sorted({v['source'] for v in values}):
        own = [v for v in values if v['source'] == source]
        sources[source] = dict(cohort=own[0]['cohort'], rows=len(own),
                              **{k: sum(v[k] for v in own)/len(own) for k in fields})
    cohorts = {}
    for cohort in ['bos', 'completion']:
        own = [v for v in sources.values() if v['cohort'] == cohort]
        cohorts[cohort] = {k: sum(v[k] for v in own)/len(own) for k in fields}
    return dict(per_source=sources, cohorts=cohorts,
                selection_score=sum(v['accepted_draft_length_sample'] for v in cohorts.values())/2,
                selection_scope='Exact first pair and one full-support sample for the conditional second Position; source equal')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper07_aat_window'))
    parser.add_argument('--loss', choices=['ce_tv', 'window'], required=True)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base-lr', type=float, default=1e-5)
    parser.add_argument('--evaluate-every', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261107)
    args = parser.parse_args()
    previous = Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared = Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0] = [str(previous/'source'), str(shared), str(shared/'reference'),
                    str(shared/'reference/runtime'), '/root/workspace/GaussianGPT']
    from train_diffusion import load_data, batch
    from diffusion_draft import make_model
    from collect_feature_teacher_from_s3 import policy_batch
    from window_objective import loss as window_loss
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, development = load_data(args.root, 'train'), load_data(args.root, 'val')
    if {v['source'] for v in train['origins']} & {v['source'] for v in development['origins']}:
        raise ValueError('Training and development sources overlap')
    tokens = torch.load(previous/'data/token_embeddings.pt', map_location='cpu', weights_only=True)
    if isinstance(tokens, dict):
        tokens = tokens['weight']
    weight = torch.load(previous/'data/feature_weight.pt', map_location='cpu', weights_only=True)['weight']
    starting = Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash/training/diffusion_full_w64/selected.pt')
    saved = torch.load(starting, map_location='cpu', weights_only=False)
    initial = Path(saved['initial'])
    model = make_model(initial, tokens, weight, layers=5, memory_tokens=64)
    model.load_state_dict(saved['model_state'])
    model = model.cuda().float()
    method = 'diffusion_window_ce_tv_w64' if args.loss == 'ce_tv' else 'diffusion_window_w64'
    out = args.root/'training'/method
    out.mkdir(parents=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(initial=str(initial), starting_checkpoint=str(starting), starting_step=saved['step'],
        layers=5, memory_tokens=64, kind='diffusion_context', feature_candidate_width=64,
        feature_block_depth=1, draft_top_p=1., parameters=sum(p.numel() for p in model.parameters()),
        target_sampling=dict(temperature=.9, top_p=.9), draft_sampling=dict(temperature=.9, top_p=1.),
        train_sources=train['origins'], development_sources=development['origins'],
        train_rows=len(train['rows']), development_rows=len(development['rows']), formal_sources_used=False,
        window_scope='P1/F1/P2 actual Draft tokens; second Feature prediction is unused',
        teacher_scope='Full-support old Feature sample and its full conditional Target P2 probability',
        objective_scope='First-Feature CE+2TV at actual Target/Draft sampling probabilities or pure negative expected P/F/P accepted length; no KL term in the window objective')
    (out/'config.json').write_text(json.dumps(config, indent=2)+chr(10))
    (out/'inference.json').write_text(json.dumps(dict(draft_top_p=1., feature_block_depth=1,
        feature_candidate_width=64, target_sampling=dict(temperature=.9, top_p=.9)), indent=2)+chr(10))
    core = [p for name, p in model.named_parameters() if not name.startswith('fusion.')]
    optimizer = torch.optim.AdamW([dict(params=core, lr=args.lr),
        dict(params=model.fusion.parameters(), lr=args.base_lr)], weight_decay=.01)
    counts = Counter(r['source_index'] for r in train['rows'])
    sampling = torch.tensor([1/counts[r['source_index']] for r in train['rows']])
    rng = torch.Generator().manual_seed(args.seed+1)
    best, history, recent_loss = -float('inf'), [], []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    def save(path, step, validation):
        torch.save(dict(model_state=model.state_dict(), kind='fusion', draft_architecture='diffusion_context',
            initial=str(initial), config=config, feature_candidate_width=64, draft_top_p=1.,
            feature_block_depth=1, step=step, validation=validation), path)

    for step in range(args.steps+1):
        if step % args.evaluate_every == 0 or step == args.steps:
            validation = measure(model, development)
            entry = dict(step=step, seconds=time.perf_counter()-started,
                recent_training_loss=sum(recent_loss)/len(recent_loss) if recent_loss else None, **validation)
            history.append(entry)
            (out/'curve.json').write_text(json.dumps(history, indent=2, allow_nan=False)+chr(10))
            if validation['selection_score'] > best:
                best = validation['selection_score']
                save(out/'selected.pt', step, validation)
            print(json.dumps(entry, allow_nan=False), flush=True)
            recent_loss = []
        if step == args.steps:
            break
        model.train()
        chosen = torch.multinomial(sampling, args.batch_size, replacement=True, generator=rng)
        inputs, teacher, rows = batch(train, chosen, 64)
        logits = model(**inputs)[:, 0]
        p = policy_batch(teacher[:, 0].float(), .9, .9).detach()
        q = (logits.float()/.9).softmax(-1)
        if args.loss == 'window':
            objective = window_loss(p, q, rows)
        else:
            ce = -(p*q.log()).sum(-1)
            tv = .5*(p-q).abs().sum(-1)
            objective = (ce+2*tv).mean()
        if not bool(torch.isfinite(objective)):
            raise ArithmeticError('Training objective is not finite')
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise ArithmeticError('Window gradient is not finite')
        schedule = min(1., (step+1)/(args.steps*.04))*.5*(1+math.cos(math.pi*step/args.steps))
        optimizer.param_groups[0]['lr'] = args.lr*schedule
        optimizer.param_groups[1]['lr'] = args.base_lr*schedule
        optimizer.step()
        recent_loss.append(float(objective.detach()))
    torch.cuda.synchronize()
    save(out/'last.pt', args.steps, validation)
    (out/'result.json').write_text(json.dumps(dict(status='complete', steps=args.steps,
        seconds=time.perf_counter()-started, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        selected_score=best, parameters=config['parameters'], initial_step=saved['step']), indent=2)+chr(10))


if __name__ == '__main__':
    main()
