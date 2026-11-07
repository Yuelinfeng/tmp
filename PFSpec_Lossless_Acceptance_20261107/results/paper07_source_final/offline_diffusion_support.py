"""Measure full Draft support and the second Feature under recorded branch conditions."""
import json
from pathlib import Path
import sys
import time
import torch


def main():
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash')
    sys.path[:0]=[str(previous/'source'),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    from train_diffusion import load_data,batch,set_acceptance
    from diffusion_draft import make_model
    from collect_feature_teacher_from_s3 import policy_batch
    from acceptance_metrics import source_average,paired_difference
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=load_data(root,'val')
    tokens=torch.load(previous/'data/token_embeddings.pt',map_location='cpu',weights_only=True)
    if isinstance(tokens,dict):tokens=tokens['weight']
    weight=torch.load(previous/'data/feature_weight.pt',map_location='cpu',weights_only=True)['weight']
    saved=torch.load(root/'training/diffusion_context_w64/selected.pt',map_location='cpu',weights_only=False)
    config=saved['config']
    model=make_model(saved['initial'],tokens,weight,layers=config['layers'],memory_tokens=config['memory_tokens'])
    model.load_state_dict(saved['model_state'])
    model=model.cuda().float().eval().requires_grad_(False)
    output=root/'development_support_01'
    output.mkdir()
    rows=[]
    started=time.perf_counter()
    with torch.inference_mode(),(output/'state_metrics.jsonl').open('w') as stream:
        for start in range(0,len(data['rows']),32):
            inputs,teacher,examples=batch(data,range(start,min(start+32,len(data['rows']))),config['memory_tokens'])
            logits=model(**inputs)
            target=policy_batch(teacher.flatten(0,1).float(),.9,.9).reshape_as(teacher)
            policies=[policy_batch(logits.flatten(0,1).float(),.9,.9).reshape_as(logits),(logits.float()/.9).softmax(-1)]
            for top_p,q in zip([.9,1.],policies):
                first_set=set_acceptance(target[:,0],q[:,0],64)
                second_set=set_acceptance(target[:,1],q[:,1],4)
                overlap=torch.minimum(target,q).sum(-1)
                for index,example in enumerate(examples):
                    origin=data['origins'][example['source_index']]
                    row=dict(method='diffusion_context_w64' if top_p==.9 else 'diffusion_full_w64',
                        source=origin['source'],cohort=origin['cohort'],pair_id=f'{origin["source"]}:{example["anchor"]}',
                        state_index=example['anchor'],feature_overlap=float(overlap[index,0]),
                        feature_set_acceptance=float(first_set[index]),pair_joint_probability=float(first_set[index])*example['weight'],
                        second_feature_overlap=float(overlap[index,1]),second_feature_set_acceptance=float(second_set[index]),
                        second_position_probability=example['second_position_probability'],
                        requested_feature_candidates=64,actual_feature_candidates=min(64,int((q[index,0]>0).sum())),draft_top_p=top_p)
                    rows.append(row)
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
    fields=['feature_overlap','feature_set_acceptance','pair_joint_probability','second_feature_overlap','second_feature_set_acceptance']
    results={}
    for cohort in sorted({row['cohort'] for row in rows}):
        own=[row for row in rows if row['cohort']==cohort]
        results[cohort]=dict(methods={method:{key:source_average([row for row in own if row['method']==method],key) for key in fields}
            for method in ['diffusion_context_w64','diffusion_full_w64']},
            full_minus_truncated={key:paired_difference(own,'diffusion_full_w64','diffusion_context_w64',key) for key in fields})
    (output/'summary.json').write_text(json.dumps(dict(status='complete',cohorts=results,rows=len(rows),seconds=time.perf_counter()-started,
        target_sampling=dict(temperature=.9,top_p=.9),draft_top_p=[.9,1.],
        second_feature_scope='Target conditions on the recorded fusion_long first Feature proposal; these diagnostics are not a joint acceptance estimate for the new first Feature output'),indent=2)+'\n')
    print('Masked Draft support measurements complete',len(rows),flush=True)


if __name__=='__main__':
    main()
