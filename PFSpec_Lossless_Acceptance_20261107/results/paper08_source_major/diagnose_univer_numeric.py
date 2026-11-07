"""Evaluate the new independent-branch tree on the two earlier numeric cases."""
import json
from pathlib import Path
import sys
import time


def main():
    previous = Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared = Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    root = Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash')
    sys.path[:0] = [str(previous/'source'), str(shared), str(shared/'reference'),
                    str(shared/'reference/runtime'), '/root/workspace/GaussianGPT']
    from types import SimpleNamespace
    from research import setup, context
    args = SimpleNamespace(project=Path('/root/workspace/GaussianGPT'), source=shared)
    setup(args)
    import torch
    from train_diffusion import load_data, batch
    from diffusion_draft import make_model
    from state_draft import LayerTap, StateTargetGraph, state_capture
    from draft_a.pf_feature_unified import _shared
    from draft_a.pf_ranker_eval import cache_for, typed_kinds
    from tree_runtime import TargetTreeRuntime
    from multi_runtime import MultiTargetTreeRuntime
    from univer_tree_runtime import UniVerTargetTreeRuntime as LongTargetTreeRuntime
    from greedy_coupling import candidates
    from collect_feature_teacher_from_s3 import policy_batch
    import bench_gaussiangpt_pf_block2_cow as capture_module
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ctx = context(args)
    target, policy = ctx['target'], _shared().policy
    data = load_data(root, 'val')
    saved = torch.load(root/'training/diffusion_context_w64/selected.pt', map_location='cpu', weights_only=False)
    model = make_model(saved['initial'], target.transformer.wte.weight.detach(),
                       target.lm_head.weight[8000:12096].detach(),
                       layers=5, memory_tokens=64)
    model.load_state_dict(saved['model_state'])
    model = model.cuda().float().eval().requires_grad_(False)
    MultiTargetTreeRuntime.maximum_width = 64
    tap = LayerTap(target)
    results = []
    started = time.perf_counter()
    work = dict(prefix_calls=0, prefix_tokens=0, verification_nodes=0)
    with torch.inference_mode(), tap.context(target), state_capture(tap):
        for source, anchor in [('dev-bos-69401', 1834), ('sequence_20265456.pt', 2532)]:
            index = next(i for i, r in enumerate(data['rows'])
                         if data['origins'][r['source_index']]['source'] == source and r['anchor'] == anchor)
            example = data['rows'][index]
            origin = data['origins'][example['source_index']]
            tokens = json.loads((root/'collection_tokens'/(Path(origin['shard']).stem+'.json')).read_text())
            known = tokens['prompt'] + tokens['suffix']
            inputs, teacher, _ = batch(data, [index], 64)
            q = (model(**inputs).float()/.9).softmax(-1)[0]
            rng = torch.Generator(device='cuda').manual_seed(20261106+index)
            first, _ = candidates(q[0], 64, rng)
            second, _ = candidates(q[1], 4, rng)
            p1, p2 = int(example['position']), int(example['second_position'])
            ranks = torch.as_tensor(ctx['ranks'], device='cpu')
            valid = torch.nonzero(ranks > ranks[p2]).flatten()
            p3 = int(valid[0]) if len(valid) else p2
            policies = {}
            hidden_errors = {}
            for style in ['linear2', 'tree162']:
                linear = StateTargetGraph(target, tap, cache_capacity_buckets=(32768,))
                cache = cache_for(target, 32768)
                capture = capture_module.CaptureLastBlock(target)
                with linear.attention_context(enabled=False):
                    last_states = None
                    for start in range(0, anchor, 64):
                        values = known[start:min(anchor, start+64)]
                        _, last_states = capture.run(target, cache, values, typed_kinds(start, len(values)), 'cuda')
                        work['prefix_calls'] += 1
                        work['prefix_tokens'] += len(values)
                    hidden_errors[style] = float((last_states[-1]-inputs['state'][0]).abs().max())
                    if style == 'linear2':
                        logits, _ = capture.run(target, cache, [known[anchor], p1], typed_kinds(anchor, 2), 'cuda')
                        nodes = 2
                    else:
                        if style == 'tree4':
                            tree = TargetTreeRuntime(target, linear, graph=False, max_width=1)
                            ids = [first[-1]]
                            kwargs = {}
                        elif style == 'tree130':
                            tree = MultiTargetTreeRuntime(target, linear, graph=False, max_width=64)
                            ids, kwargs = first, {}
                        else:
                            tree = LongTargetTreeRuntime(target, linear, graph=False, max_width=16)
                            ids = first[:16]
                            kwargs = dict(second_features=[[f+8000 for f in candidates(q[1],4,rng)[0]] for _ in ids], third_position=p3)
                        logits, _ = tree.verify(cache, known[anchor], p1, [f+8000 for f in ids], [p2]*len(ids), **kwargs)
                        nodes = 162 if style == 'tree162' else 2+2*len(ids)
                        tree.commit(cache, [])
                    work['verification_nodes'] += nodes
                    policies[style] = [policy(logits[slot].clone()) for slot in [0, 1]]
                capture.close()
                del cache, linear
                torch.cuda.empty_cache()
            comparisons = {}
            for style in ['tree162']:
                comparisons[style] = [dict(tv=float((policies[style][slot]-policies['linear2'][slot]).abs().sum()/2),
                    support_equal=bool(torch.equal(policies[style][slot]>0, policies['linear2'][slot]>0)))
                    for slot in [0, 1]]
            saved_feature = policy_batch(teacher[:, 0].float(), .9, .9)[0]
            current_feature = policies['linear2'][1][8000:12096]
            results.append(dict(source=source, anchor=anchor, stored_position_probability=example['weight'],
                linear_position_probability=float(policies['linear2'][0][p1]),
                saved_feature_vs_fresh_linear_tv=float((saved_feature-current_feature).abs().sum()/2),
                saved_feature_vs_fresh_linear_support_equal=bool(torch.equal(saved_feature>0, current_feature>0)),
                saved_anchor_hidden_maximum_errors=hidden_errors, fresh_tree_vs_linear=comparisons))
    result = dict(status='complete', rows=results, work=work, seconds=time.perf_counter()-started,
                  scope='Identical independent fresh caches and 64-token prefill chunks; two retained numeric outliers, ordinary execution only')
    (root.parent/'paper08_univer/numeric_history_diagnostic.json').write_text(json.dumps(result, indent=2)+chr(10))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
