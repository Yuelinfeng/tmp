"""Measure all methods on the same full-support observer development histories."""
import json
from pathlib import Path
import sys
import time
import torch


def main():
    previous = Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared = Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    root = Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
    out = root/'paper07_aat_window/development_curve_01'
    sys.path[:0] = [str(previous/'source'), str(shared), str(shared/'reference'),
                    str(shared/'reference/runtime'), '/root/workspace/GaussianGPT']
    from train_diffusion import load_data, batch
    from diffusion_draft import make_model
    from state_draft import make_draft
    from collect_feature_teacher_from_s3 import policy_batch
    from window_objective import window_values
    from acceptance_metrics import source_average, paired_difference
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = load_data(root/'paper07_aat_window', 'val')
    tokens = torch.load(previous/'data/token_embeddings.pt', map_location='cpu', weights_only=True)
    if isinstance(tokens, dict):
        tokens = tokens['weight']
    weight = torch.load(previous/'data/feature_weight.pt', map_location='cpu', weights_only=True)['weight']
    initial = previous/'training/fusion_long/selected.pt'
    models = {}
    paths = {'diffusion_full_w64': root/'paper06_dflash/training/diffusion_full_w64/selected.pt',
             'diffusion_window_ce_tv_w64': root/'paper07_aat_window/training/diffusion_window_ce_tv_w64/selected.pt',
             'diffusion_window_w64': root/'paper07_aat_window/training/diffusion_window_w64/selected.pt'}
    for method, path in paths.items():
        saved = torch.load(path, map_location='cpu', weights_only=False)
        model = make_model(saved['initial'], tokens, weight, layers=5, memory_tokens=64)
        model.load_state_dict(saved['model_state'])
        models[method] = model.cuda().float().eval().requires_grad_(False)
    fusion = make_draft(weight, 'fusion', initial).cuda().float().eval().requires_grad_(False)
    w1 = make_draft(weight, 'top', Path('/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt')).cuda().float().eval().requires_grad_(False)
    out.mkdir()
    rows = []
    started = time.perf_counter()
    with torch.inference_mode(), (out/'state_metrics.jsonl').open('w') as stream:
        for start in range(0, len(data['rows']), 32):
            inputs, teacher, examples = batch(data, range(start, min(start+32, len(data['rows']))), 64)
            p = policy_batch(teacher[:, 0].float(), .9, .9)
            _, w1_logits = w1(inputs['state'], inputs['previous_feature'], inputs['p1'])
            _, fusion_logits = fusion(inputs['state'], inputs['previous_feature'], inputs['p1'])
            policies = {'w1': policy_batch(w1_logits.float(), .9, .9),
                        'fusion_long': policy_batch(fusion_logits.float(), .9, .9)}
            policies['multi_greedy_w64'] = policies['fusion_long']
            for method, model in models.items():
                policies[method] = (model(**inputs)[:, 0].float()/.9).softmax(-1)
            for method, q in policies.items():
                width = 1 if method in ['w1', 'fusion_long'] else 64
                values = window_values(p, q, examples, width)
                active = torch.minimum((q > 0).sum(-1), torch.full_like((q > 0).sum(-1), width))
                for i, example in enumerate(examples):
                    origin = data['origins'][example['source_index']]
                    row = dict(method=method, source=origin['source'], cohort=origin['cohort'],
                        pair_id=f'{origin["source"]}:{example["anchor"]}', state_index=example['anchor'],
                        feature_overlap=float(values['feature_overlap'][i]),
                        feature_set_acceptance=float(values['feature_set_acceptance'][i]),
                        pair_joint_probability=float(values['first_joint'][i]),
                        second_position_joint_sample=float(values['second_position_joint_sample'][i]),
                        accepted_draft_length_sample=float(values['accepted_draft_length_sample'][i]),
                        position_probability=example['weight'],
                        ce=None if bool(((p[i]>0) & (q[i]==0)).any()) else float(-(p[i]*q[i].clamp_min(1e-30).log()).sum()),
                        ce_infinite=bool(((p[i]>0) & (q[i]==0)).any()),
                        tv=float((p[i]-q[i]).abs().sum()/2),
                        top1_match=bool(p[i].argmax() == q[i].argmax()),
                        actual_feature_candidates=int(active[i]), target_verification_nodes=2+2*int(active[i]))
                    rows.append(row)
                    stream.write(json.dumps(row, allow_nan=False)+chr(10))
    fields = ['feature_overlap', 'feature_set_acceptance', 'pair_joint_probability',
              'second_position_joint_sample', 'accepted_draft_length_sample', 'ce', 'tv', 'top1_match']
    results = {}
    for cohort in ['bos', 'completion']:
        own = [r for r in rows if r['cohort'] == cohort]
        results[cohort] = {}
        for method in policies:
            selected = [r for r in own if r['method'] == method]
            results[cohort][method] = dict(states=len(selected), sources=len({r['source'] for r in selected}),
                source_means={k: source_average(selected, k) for k in fields},
                differences_vs_full={k: paired_difference(own, method, 'diffusion_full_w64', k) for k in fields},
                differences_vs_control={k: paired_difference(own, method, 'diffusion_window_ce_tv_w64', k) for k in fields})
    (out/'summary.json').write_text(json.dumps(dict(status='complete', rows=len(rows), cohorts=results,
        seconds=time.perf_counter()-started, extra_target_calls=0,
        target_sampling=dict(temperature=.9, top_p=.9),
        scope='All six methods use the same saved development states and full-support observation sample; not formal data',
        window_scope='Exact first Feature set mass; importance estimate for its full conditional second Position; actual advancement is measured online',
        ce_scope='CE is null with ce_infinite=true when positive Target mass lies outside Draft support; no sampling mass is added'), indent=2, allow_nan=False)+chr(10))
    print('Full-support common development histories complete', len(rows), flush=True)


if __name__ == '__main__':
    main()
