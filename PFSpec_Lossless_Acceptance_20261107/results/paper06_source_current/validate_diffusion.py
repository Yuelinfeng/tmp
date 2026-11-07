"""Test masked-block visibility, finite policies, and actual gradients."""
import json
from pathlib import Path
import sys
import torch


def main():
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper06_dflash')
    sys.path[:0]=[str(previous/'source'),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    from diffusion_draft import make_model
    from train_diffusion import load_data,batch
    from collect_feature_teacher_from_s3 import policy_batch
    torch.set_num_threads(1)
    torch.manual_seed(20261106)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    tokens=torch.load(previous/'data/token_embeddings.pt',map_location='cpu',weights_only=True)
    if isinstance(tokens,dict):tokens=tokens['weight']
    weight=torch.load(previous/'data/feature_weight.pt',map_location='cpu',weights_only=True)['weight']
    model=make_model(previous/'training/fusion_long/selected.pt',tokens,weight).cuda().float()
    data=load_data(root,'train')
    indexes=sorted(range(len(data['rows'])),key=lambda index:data['rows'][index]['memory_end'])[:4]
    inputs,teacher,rows=batch(data,indexes,64)
    with torch.no_grad():
        first,_=model.fusion(inputs['state'],inputs['previous_feature'],inputs['p1'])
        initial=15.*torch.tanh(torch.nn.functional.linear(first,model.fusion.frozen_head)/15.)
        computed=model(**inputs)
        initial_difference=float((initial-computed[:,0]).abs().max())
        model.result.weight.copy_(torch.eye(1024,device='cuda')*.02)
        before=model(**inputs)
        altered=dict(inputs,memory=inputs['memory'].clone())
        altered['memory'][~inputs['valid']]=torch.randn_like(altered['memory'][~inputs['valid']])*100
        padding_difference=float((before-model(**altered)).abs().max())
        future_input_difference=0.
        for offset,row in enumerate(rows):
            source,end=row['source_index'],row['memory_end']
            memory=data['memories'][source]
            saved=memory[end:].clone()
            memory[end:]+=100
            rebuilt,_,_=batch(data,[indexes[offset]],64)
            future_input_difference=max(future_input_difference,float((rebuilt['memory']-inputs['memory'][offset:offset+1]).abs().max()))
            memory[end:]=saved
        q=policy_batch(before.flatten(0,1),.9,.9)
        mass_error=float((q.sum(-1)-1).abs().max())
        changed=dict(inputs,memory=inputs['memory']+torch.randn_like(inputs['memory']))
        context_response=float((before-model(**changed)).abs().max())
    model.zero_grad(set_to_none=True)
    logits=model(**inputs)
    loss=-(teacher.softmax(-1)*logits.log_softmax(-1)).sum(-1).mean()
    loss.backward()
    finite=bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    context_gradient=float(model.context_mix.weight.grad.abs().max())
    result=dict(status='complete',examples=len(rows),parameters=sum(p.numel() for p in model.parameters()),
        initial_fusion_logit_max_difference=initial_difference,padded_context_logit_max_difference=padding_difference,
        future_target_context_input_max_difference=future_input_difference,policy_mass_max_error=mass_error,
        context_logit_response=context_response,context_parameter_gradient_max=context_gradient,
        loss=float(loss.detach()),gradients_finite=finite,target_embedding_requires_grad=model.target_tokens.requires_grad,
        feature_head_requires_grad=model.fusion.frozen_head.requires_grad)
    if initial_difference>1e-5 or padding_difference>1e-5 or future_input_difference!=0 or mass_error>1e-5:
        raise ArithmeticError(str(result))
    if not finite or context_response<=0 or context_gradient<=0 or model.target_tokens.requires_grad or model.fusion.frozen_head.requires_grad:
        raise ArithmeticError(str(result))
    (root/'numeric_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
