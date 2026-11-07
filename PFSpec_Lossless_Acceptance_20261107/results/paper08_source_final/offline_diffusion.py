"""Compare all Feature policies on identical observed Target histories."""
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
    from state_draft import make_draft
    from collect_feature_teacher_from_s3 import policy_batch
    from acceptance_metrics import source_average,paired_difference
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    data=load_data(root,'val')
    tokens=torch.load(previous/'data/token_embeddings.pt',map_location='cpu',weights_only=True)
    if isinstance(tokens,dict):tokens=tokens['weight']
    weight=torch.load(previous/'data/feature_weight.pt',map_location='cpu',weights_only=True)['weight']
    initial=previous/'training/fusion_long/selected.pt'
    saved=torch.load(root/'training/diffusion_context_w64/selected.pt',map_location='cpu',weights_only=False)
    config=saved['config']
    student=make_model(initial,tokens,weight,layers=config['layers'],memory_tokens=config['memory_tokens'])
    student.load_state_dict(saved['model_state'])
    student=student.cuda().float().eval().requires_grad_(False)
    fusion=make_draft(weight,'fusion',initial).cuda().float().eval().requires_grad_(False)
    w1=make_draft(weight,'top',Path('/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt')).cuda().float().eval().requires_grad_(False)
    output=root/'development_curve_01'
    output.mkdir()
    rows=[]
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),(output/'state_metrics.jsonl').open('w') as stream:
        for start in range(0,len(data['rows']),32):
            inputs,teacher,examples=batch(data,range(start,min(start+32,len(data['rows']))),config['memory_tokens'])
            prediction=student(**inputs)
            _,fusion_logits=fusion(inputs['state'],inputs['previous_feature'],inputs['p1'])
            _,w1_logits=w1(inputs['state'],inputs['previous_feature'],inputs['p1'])
            target=policy_batch(teacher[:,0].float(),.9,.9)
            target2=policy_batch(teacher[:,1].float(),.9,.9)
            policies={'w1':policy_batch(w1_logits.float(),.9,.9),
                'fusion_long':policy_batch(fusion_logits.float(),.9,.9),
                'diffusion_context_w64':policy_batch(prediction[:,0].float(),.9,.9)}
            policies['multi_greedy_w64']=policies['fusion_long']
            second_overlap=torch.minimum(target2,policy_batch(prediction[:,1].float(),.9,.9)).sum(-1)
            for method,q in policies.items():
                width=64 if method in ('multi_greedy_w64','diffusion_context_w64') else 1
                overlap=torch.minimum(target,q).sum(-1)
                acceptance=set_acceptance(target,q,width)
                candidates=torch.minimum((q>0).sum(-1),torch.full_like((q>0).sum(-1),width))
                for index,example in enumerate(examples):
                    origin=data['origins'][example['source_index']]
                    row=dict(method=method,source=origin['source'],cohort=origin['cohort'],
                        pair_id=f'{origin["source"]}:{example["anchor"]}',state_index=example['anchor'],
                        feature_overlap=float(overlap[index]),feature_set_acceptance=float(acceptance[index]),
                        position_probability=example['weight'],pair_joint_probability=float(acceptance[index])*example['weight'],
                        requested_feature_candidates=width,actual_feature_candidates=int(candidates[index]),
                        target_verification_nodes=2+2*int(candidates[index]),memory_end=example['memory_end'])
                    if method=='diffusion_context_w64':row['second_feature_overlap']=float(second_overlap[index])
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
                    rows.append(row)
    torch.cuda.synchronize()
    fields=['feature_overlap','feature_set_acceptance','pair_joint_probability','actual_feature_candidates']
    results={}
    for cohort in sorted({row['cohort'] for row in rows}):
        own=[row for row in rows if row['cohort']==cohort]
        results[cohort]={}
        for method in policies:
            selected=[row for row in own if row['method']==method]
            results[cohort][method]=dict(states=len(selected),sources=len({row['source'] for row in selected}),
                source_means={key:source_average(selected,key) for key in fields},
                differences_vs_fusion_long={key:paired_difference(own,method,'fusion_long',key) for key in fields},
                differences_vs_greedy64={key:paired_difference(own,method,'multi_greedy_w64',key) for key in fields})
    (output/'summary.json').write_text(json.dumps(dict(status='complete',cohorts=results,rows=len(rows),
        seconds=time.perf_counter()-started,peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30,
        parameters=sum(p.numel() for p in student.parameters()),selected_step=saved['step'],
        target_sampling=dict(temperature=.9,top_p=.9),draft_sampling=dict(temperature=.9,top_p=.9),
        scope='All sampled development anchors including Position rejection and zero Position probabilities; source-equal comparison of the same full Target histories'),indent=2)+'\n')
    print('Masked context common histories complete',len(rows),flush=True)


if __name__=='__main__':
    main()
