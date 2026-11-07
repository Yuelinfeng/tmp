"""Compare the actual per-model draft policies on common Target states."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path[:0]=['/root/autodl-tmp/pfspec_feature_states_20261105/source',
              '/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04',
              '/root/workspace/GaussianGPT']

from acceptance_metrics import paired_difference, source_average
from research import setup, write


def cohort_summary(rows):
    fields = ('feature_overlap', 'pair_joint_probability', 'position_probability',
              'raw_teacher_cross_entropy', 'target_top1_agreement')
    groups = {}
    for cohort in sorted({r['cohort'] for r in rows}):
        own = [r for r in rows if r['cohort'] == cohort]
        methods = {}
        for method in sorted({r['method'] for r in own}):
            selected = [r for r in own if r['method'] == method]
            result = dict(states=len(selected), sources=len({r['source'] for r in selected}),
                          state_means={k: sum(r[k] for r in selected)/len(selected) for k in fields},
                          source_means={k: source_average(selected, k) for k in fields})
            if method != 'w1':
                result['paired_differences_vs_w1'] = {
                    k: paired_difference(own, method, 'w1', k) for k in fields}
            methods[method] = result
        groups[cohort] = methods
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/root/workspace/GaussianGPT'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp/pfspec_feature_states_20261105'))
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--split', default='val')
    parser.add_argument('--methods', default='w1,fusion,fusion_long,top_capacity')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    setup(args)
    import torch
    from collect_feature_teacher_from_s3 import policy_batch
    from state_draft import make_draft
    data = torch.load(args.data, map_location='cpu', weights_only=True)
    source_plan = json.loads(args.selection.read_text())
    selection = source_plan['files'][args.split]
    weight = torch.load(args.root/'data/feature_weight.pt', map_location='cpu', weights_only=True)['weight']
    tokens_path = args.root/'data/token_embeddings.pt'
    token_weight = torch.load(tokens_path, map_location='cpu', weights_only=True) if tokens_path.exists() else None
    args.out.mkdir(parents=True)
    rows = []
    models = {}
    with torch.inference_mode(), (args.out/'state_metrics.jsonl').open('w') as handle:
        for method in args.methods.split(','):
            checkpoint = (args.project/'draft_a/runs/feature_target_continue_20261014/selected.pt'
                          if method == 'w1' else args.root/'training'/method/'selected.pt')
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            kind = 'top' if method == 'w1' else saved['kind']
            model = make_draft(weight, kind, checkpoint, token_weight).cuda().float().eval().requires_grad_(False)
            draft_top_p=float(saved.get('draft_top_p',.9))
            torch.cuda.reset_peak_memory_stats()
            begun=time.perf_counter()
            models[method] = dict(checkpoint=str(checkpoint), kind=kind,
                draft_sampling=dict(temperature=.9,top_p=draft_top_p),
                parameters=sum(p.numel() for p in model.parameters()))
            for start in range(0, len(data['state']), args.batch_size):
                sl = slice(start, start+args.batch_size)
                _, logits = model(data['state'][sl].cuda(), data['previous_feature'][sl].cuda(), data['position'][sl].cuda())
                teacher = data['logits'][sl].cuda().float()
                p=policy_batch(teacher,.9,.9)
                q=(logits.float()/.9).softmax(-1) if draft_top_p==1 else policy_batch(logits.float(),.9,draft_top_p)
                overlap = torch.minimum(p, q).sum(-1).cpu().tolist()
                ce = (-(teacher.softmax(-1)*logits.float().log_softmax(-1)).sum(-1)).cpu().tolist()
                top1 = (p.argmax(-1) == q.argmax(-1)).cpu().tolist()
                for offset, alpha in enumerate(overlap):
                    index = start+offset
                    number = int(data['source'][index])
                    name = selection[number]
                    identity = Path(name).parent.parent.name if Path(name).is_absolute() else Path(name).stem
                    metadata = source_plan.get('source_metadata',{}).get(name,{})
                    cohort = metadata.get('cohort', 'bos' if 'bos-' in identity else 'completion' if 'completion-' in identity else 'unspecified')
                    identity = metadata.get('source',identity)
                    w = float(data['weight'][index])
                    row = dict(method=method, source=identity, cohort=cohort,
                               pair_id=f'{metadata.get("pair_id",identity)}:{int(data["pair_index"][index])}',
                               history_pair_index=int(data['pair_index'][index]),
                               position_candidate=int(data['position'][index]),
                               position_probability=w, feature_overlap=alpha,
                               pair_joint_probability=w*alpha,
                               raw_teacher_cross_entropy=ce[offset], target_top1_agreement=int(top1[offset]))
                    handle.write(json.dumps(row, allow_nan=False)+'\n')
                    rows.append(row)
            handle.flush()
            torch.cuda.synchronize()
            models[method].update(seconds=time.perf_counter()-begun,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            del model
            torch.cuda.empty_cache()
    write(args.out/'summary.json', dict(status='complete', cohorts=cohort_summary(rows), models=models,
          data=str(args.data), selection=str(args.selection), split=args.split,
          gpu=torch.cuda.get_device_name(), fp32=True, tf32=False, batch_size=args.batch_size,
          temperature=.9, top_p=.9,top_p_scope='Target; Draft policy listed per model',
          target_information='Saved states and labels shared by all methods; labels used only for evaluation',
          cross_entropy='Raw full-vocabulary teacher soft-label CE; top1 means Target argmax agreement',
          joint_scope='Mean of per-state Position probability times Feature overlap'))
    print(json.dumps(dict(status='complete', rows=len(rows), output=str(args.out))))


if __name__ == '__main__':
    main()
