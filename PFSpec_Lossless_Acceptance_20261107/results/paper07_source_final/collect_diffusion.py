"""Collect committed Target context for a masked two-Feature Draft."""
import argparse
from collections import Counter
from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
import random
import sys
import time


def configure(args):
    sys.path[:0] = [str(args.previous_source), str(args.source),
                    str(args.source / 'reference'), str(args.source / 'reference/runtime'),
                    str(args.project)]
    from research import setup
    setup(args)


def prepare(args):
    import torch
    selection = json.loads((args.root / 'data/selection.json').read_text())
    names = list(selection['files']['train'])
    random.Random(args.seed).shuffle(names)
    cases = []
    for i in range(args.training_sources):
        seed = 74001 + i
        cases.append(dict(pair_id=f'opd-train-bos-{seed}', source=f'opd-bos-{seed}',
                          cohort='bos', prefix=0, seed=seed, prompt=[12096], split='train'))
    for i, name in enumerate(names[:args.training_sources]):
        path = args.project / 'draft_a/data/teacher_sequences' / name
        if not path.exists():
            matches = list((args.project / 'draft_a/runs/rollout500_20261013').rglob(name))
            if len(matches) != 1:
                raise FileNotFoundError(name)
            path = matches[0]
        value = torch.load(path, map_location='cpu', weights_only=False)
        body = value['tokens'].long().tolist()
        if body and body[0] == 12096:
            body = body[1:]
        if body and body[-1] == 12096:
            body = body[:-1]
        length = min(256, len(body) // 4 * 2)
        if length < 2:
            raise ValueError(f'Training prefix is empty: {name}')
        cases.append(dict(pair_id=f'opd-train-completion-{name[:-3]}', source=name,
                          cohort='completion', prefix=length, seed=74101 + i,
                          prompt=[12096] + body[:length], split='train'))
    development = json.loads((args.root / 'acceptance_study/development_inputs.json').read_text())
    for row in development['contexts']:
        cases.append(dict(row, split='val'))
    train_sources = {c['source'] for c in cases if c['split'] == 'train'}
    val_sources = {c['source'] for c in cases if c['split'] == 'val'}
    if train_sources & val_sources:
        raise ValueError('Training and development sources overlap')
    formal = json.loads((args.root / 'acceptance_study/formal_natural_inputs.json').read_text())
    formal_sources = {c['source'] for c in formal['contexts']}
    if train_sources & formal_sources or val_sources & formal_sources:
        raise ValueError('Feedback inputs include previous formal sources')
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / 'feedback_inputs.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(dict(contexts=cases, seed=args.seed,
        scope='Training uses new BOS seeds and existing training-trajectory prefixes; '
              'validation uses prior development inputs. Formal sources are excluded.'), indent=2) + '\n')
    print('Prepared', len(cases), 'inputs', flush=True)


class FeedbackRecorder:
    """Observe the existing propose/verify/decide calls; never draw decoder RNG."""
    def __init__(self, model, tree, shared, *, max_rows, seed):
        self.model, self.tree, self.shared = model, tree, shared
        self.max_rows = max_rows
        self.rng = random.Random(seed)
        self.seen = 0
        self.current = None
        self.rows = []
        self.events = Counter()
        self.previous_event = 'start'
        self.anchors = []

    def finish(self, event):
        item = self.current
        self.events[event] += 1
        self.anchors.append(dict(anchor=item['anchor'], event=event))
        if item['slot'] is not None:
            row = item['saved']
            row.update(event=event, position_accepted=event != 'position_rejected',
                       feature_accepted=event == 'feature_accepted',
                       previous_event=self.previous_event)
            if item['slot'] == len(self.rows):
                self.rows.append(row)
            else:
                self.rows[item['slot']] = row
        self.previous_event = event
        self.current = None

    @contextmanager
    def context(self):
        import bench_gaussiangpt_pf_block2_cow as decisions
        original_propose = self.model.propose
        original_verify = self.tree.verify
        original_decide = decisions.decide

        def propose(prefix, position, state, generator):
            if self.current is not None:
                raise RuntimeError('Previous verification feedback is unfinished')
            q = original_propose(prefix, position, state, generator)
            self.seen += 1
            slot = len(self.rows) if len(self.rows) < self.max_rows else self.rng.randrange(self.seen)
            if slot >= self.max_rows:
                slot = None
            self.current = dict(stage=0, slot=slot, anchor=len(prefix)-1, q=q)
            if slot is not None:
                self.current['saved'] = dict(state=state.detach().float().cpu().clone(),
                    previous_feature=prefix[-1]-8000, position=int(position),
                    anchor=len(prefix)-1, previous_position=int(prefix[-2]))
            return q

        def verify(cache, pending, position, features, next_positions):
            result = original_verify(cache, pending, position, features, next_positions)
            if self.current is None or len(features) != 1:
                raise RuntimeError('Expected one observed Feature proposal')
            if self.current['slot'] is not None:
                import torch
                logits = result[0]
                if logits.ndim == 3:
                    logits = logits[0]
                p_position = self.shared.policy(logits[0])
                p = self.shared.policy(logits[1])[8000:12096]
                q = self.current['q']
                local = int(features[0])-8000
                self.current['saved'].update(logits=logits[1,8000:12096].detach().float().cpu().clone(),
                    weight=float(p_position[position].item()), candidate=local,
                    candidate_p=float(p[local].item()), candidate_q=float(q[local].item()),
                    overlap=float(torch.minimum(p,q).sum().item()))
            return result

        def decide(alpha, generator, device):
            outcome = original_decide(alpha, generator, device)
            if self.current is not None:
                stage = self.current['stage']
                if stage == 0 and not bool(outcome):
                    self.finish('position_rejected')
                elif stage == 1 and not bool(outcome):
                    self.finish('feature_rejected')
                elif stage == 2:
                    self.finish('feature_accepted')
                else:
                    self.current['stage'] += 1
            return outcome

        self.model.propose = propose
        self.tree.verify = verify
        decisions.decide = decide
        try:
            yield self
            if self.current is not None:
                raise RuntimeError('Final verification feedback is unfinished')
        finally:
            self.model.propose = original_propose
            self.tree.verify = original_verify
            decisions.decide = original_decide


def collect(args):
    from diffusion_memory import MemoryRecorder
    import torch
    from research import context, write
    from state_draft import LayerTap, StateTargetGraph, state_capture, make_draft
    from method_adapters import load_method
    from tree_runtime import TargetTreeRuntime
    from tree_decoder import decode
    from proposal_adapters import build_proposal
    from draft_a.pf_feature_unified import MLPAdapter, _shared
    from draft_a.pf_ranker_eval import cache_for, validate_sequence
    from acceptance_metrics import normalize
    cases = json.loads((args.out / 'feedback_inputs.json').read_text())['contexts']
    ctx = context(args)
    target = ctx['target']
    checkpoint = args.root / 'training/fusion_long/selected.pt'
    model = make_draft(target.lm_head.weight[8000:12096].detach(), 'fusion', checkpoint)
    model = model.cuda().float().eval().requires_grad_(False)
    adapter = load_method('pf_mlp', ctx)
    adapter.model = MLPAdapter(model, bf16=False)
    tap = LayerTap(target)
    runtime = StateTargetGraph(target, tap, cache_capacity_buckets=(16384,))
    tree = TargetTreeRuntime(target, runtime, graph=True, max_width=1)
    proposal = build_proposal(adapter, verify=False)
    factory = lambda core, length: runtime.cache_for(core, length, factory=cache_for)
    started = time.perf_counter()
    rows = []

    def run(case, recorder=None, budget=None):
        with ExitStack() as stack:
            stack.enter_context(tap.context(target))
            stack.enter_context(state_capture(tap))
            stack.enter_context(proposal.context())
            stack.enter_context(runtime.attention_context())
            if recorder:
                stack.enter_context(recorder.context())
            return decode(target, adapter.model, ctx['ranker'], ctx['ranks'],
                          case['prompt'], budget or args.budget, case['seed'],
                          tree_runtime=tree, width=1, cache_factory=factory)

    try:
        with torch.inference_mode():
            if args.validate:
                reports = []
                for case in (cases[0], cases[-1]):
                    plain_counts, plain = run(case, budget=128)
                    recorder = MemoryRecorder(adapter.model, tree, _shared(), max_rows=128, seed=args.seed)
                    observed_counts, observed = run(case, recorder, budget=128)
                    fields = [k for k,v in plain_counts.items() if isinstance(v,(int,list,dict,bool))
                              and 'runtime_stats' not in k and not k.endswith('_rate')]
                    equal_counts = all(plain_counts[k] == observed_counts[k] for k in fields)
                    report = dict(pair_id=case['pair_id'], tokens_equal=plain == observed,
                                  actions_equal=equal_counts, observed_rounds=recorder.seen,
                                  events=dict(recorder.events))
                    reports.append(report)
                    if plain != observed or not equal_counts:
                        raise ArithmeticError('Observation changed sampling or decode actions')
                write(args.out / 'observation_validation.json', dict(reports=reports, status='complete'))
                print('Observation equivalence passed', flush=True)
                return
            for index, case in enumerate(cases):
                shard = args.out / 'feedback_shards' / (case['pair_id'] + '.pt')
                if shard.exists():
                    saved = torch.load(shard, map_location='cpu', weights_only=False)
                    rows.append(saved['record'])
                    continue
                recorder = MemoryRecorder(adapter.model, tree, _shared(), max_rows=args.rows_per_source,
                                              seed=args.seed + index)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                begin = time.perf_counter()
                counts, suffix = run(case, recorder)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-begin
                validate_sequence(case['prompt'], suffix, ctx['ranks'])
                if recorder.seen != counts['tree_verify_calls']:
                    raise ArithmeticError('Observed and decoded full-round counts differ')
                if recorder.events['feature_accepted'] != counts['feature_accepted']:
                    raise ArithmeticError('Observed Feature acceptance differs from decoder')
                record = dict(cohort=case['cohort'], source=case['source'], pair_id=case['pair_id'],
                    method='fusion_long', split=case['split'], seed=case['seed'], prefix=case['prefix'],
                    seconds=elapsed, generated_tokens=len(suffix), eos=bool(suffix and suffix[-1]==12096),
                    counts=counts, peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
                    observed_rows=len(recorder.rows), events=dict(recorder.events))
                record['acceptance'] = normalize(record)
                shard.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(memory=torch.stack(recorder.memory), rows=recorder.rows, anchors=recorder.anchors, record=record,
                                prompt=case['prompt'], suffix=suffix), shard)
                rows.append(record)
                write(args.out / 'feedback_progress.json', dict(status='running', completed=len(rows),
                      expected=len(cases), last=record, seconds=time.perf_counter()-started))
                print(case['pair_id'], len(suffix), dict(recorder.events), flush=True)
            write(args.out / 'feedback_summary.json', dict(records=rows, gpu=torch.cuda.get_device_name(),
                policy=dict(fp32=True, tf32=False, temperature=.9, top_p=.9),
                initial_checkpoint=str(checkpoint), extra_target_calls=0,
                scope='Actual single-Feature starts and outcomes; committed histories remain Target-distributed. '
                      'No recursive draft-only hidden-state exposure is claimed.',
                seconds=time.perf_counter()-started))
            write(args.out / 'feedback_progress.json', dict(status='complete', completed=len(rows),
                  expected=len(cases), seconds=time.perf_counter()-started))
    finally:
        proposal.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare','collect'])
    parser.add_argument('--project', type=Path, default=Path('/root/workspace/GaussianGPT'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    parser.add_argument('--previous-source', type=Path, default=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source'))
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/pfspec_feature_states_20261105'))
    parser.add_argument('--out', type=Path, default=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash'))
    parser.add_argument('--training-sources', type=int, default=16)
    parser.add_argument('--rows-per-source', type=int, default=384)
    parser.add_argument('--budget', type=int, default=16000)
    parser.add_argument('--seed', type=int, default=20261106)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    configure(args)
    if args.phase == 'prepare':
        prepare(args)
    else:
        collect(args)


if __name__ == '__main__':
    main()
