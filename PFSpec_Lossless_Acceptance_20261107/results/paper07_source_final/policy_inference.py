"""Use each saved draft sampling policy with the existing exact W1 decoder."""


def configure():
    import torch
    import state_draft
    import draft_a.pf_feature_unified as module
    original_factory=state_draft.make_draft
    original_adapter=module.MLPAdapter
    policies={}

    def factory(weight,kind,checkpoint,token_weight=None):
        model=original_factory(weight,kind,checkpoint,token_weight)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        top_p=float(saved.get('draft_top_p',.9))
        model._pfspec_draft_top_p=top_p
        policies[str(checkpoint)]=dict(temperature=.9,top_p=top_p)
        return model

    class PolicyAdapter(original_adapter):
        def propose(self,prefix,p1,h_prev,rng):
            from collect_feature_teacher_from_s3 import policy_batch
            shared=module._shared()
            shared._require(self._prepared and h_prev is not None,'MLP needs a prepared Target state')
            self.forward_calls+=1
            self.input_tokens+=1

            def forward():
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=self.bf16):
                    _,logits=self.feature(h_prev.unsqueeze(0),
                        torch.tensor([prefix[-1]-module.P_VOCAB],device=h_prev.device),
                        torch.tensor([p1],device=h_prev.device))
                return logits

            logits=self._run('feature_forward',forward,gpu=True,input_tokens=1,operation='state_mlp')
            top_p=float(getattr(self.feature,'_pfspec_draft_top_p',.9))
            q=self._run('sampling',lambda:(logits.float()/.9).softmax(-1)[0] if top_p==1 else
                        policy_batch(logits.float(),.9,top_p)[0],gpu=True,operation='feature_policy').float()
            shared._require(q.ndim==1 and q.numel()==module.F_VOCAB,'Expected 4096 Feature probabilities')
            return q

    state_draft.make_draft=factory
    module.MLPAdapter=PolicyAdapter
    return policies
