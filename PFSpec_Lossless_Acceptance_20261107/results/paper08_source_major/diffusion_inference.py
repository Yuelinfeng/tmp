"""Read committed Target context for masked-block Feature proposals."""
from contextlib import contextmanager
import torch


class LiveMemory:
    def __init__(self, adapter, tree):
        self.adapter,self.tree = adapter,tree
        self.values = {}
        self.last_tree = None

    def store(self, start, states):
        states = states.detach()
        if states.ndim==3:
            states=states[0]
        for key in list(self.values):
            if key>=start:
                del self.values[key]
        for offset,value in enumerate(states):
            self.values[start+offset]=value.clone()
        retain_from=start+len(states)-self.adapter.feature.memory_tokens-8
        for key in list(self.values):
            if key<retain_from:
                del self.values[key]

    def inputs(self, prefix, p1, p2, state):
        end=len(prefix)-1
        length=self.adapter.feature.memory_tokens
        start=max(0,end-length)
        if any(index not in self.values for index in range(start,end)):
            raise ArithmeticError('Missing committed Target context for masked proposal')
        if not torch.equal(self.values[end-1],state):
            raise ArithmeticError('Masked proposal anchor differs from the Target state')
        valid_length=end-start
        memory=torch.zeros(length,3072,device=state.device)
        memory[-valid_length:]=torch.stack([self.values[index] for index in range(start,end)])
        self.adapter.context_tokens+=valid_length
        return dict(memory=memory[None],memory_positions=torch.arange(end-length,end,device=state.device).clamp_min(0)[None],
            valid=(torch.arange(length,device=state.device)>=length-valid_length)[None],
            ends=torch.tensor([end],device=state.device),state=state[None],
            previous_feature=torch.tensor([prefix[-1]-8000],device=state.device),
            p1=torch.tensor([p1],device=state.device),p2=torch.tensor([p2],device=state.device))

    @contextmanager
    def context(self):
        import bench_gaussiangpt_pf_block2_cow as module
        original_capture=module.CaptureLastBlock
        original_verify=self.tree.verify
        original_commit=self.tree.commit
        owner=self

        class Capture(original_capture):
            def run(self,target,cache,tokens,*args,**kwargs):
                before=int(cache.get_pos())
                result=super().run(target,cache,tokens,*args,**kwargs)
                owner.store(before,result[1])
                return result

        def verify(*args,**kwargs):
            result=original_verify(*args,**kwargs)
            owner.last_tree=result[1].detach().clone()
            return result

        def commit(cache,path):
            before=int(cache.get_pos())
            result=original_commit(cache,path)
            owner.store(before,owner.last_tree[path])
            return result

        self.adapter.memory=self
        self.adapter.context_tokens=0
        module.CaptureLastBlock=Capture
        self.tree.verify=verify
        self.tree.commit=commit
        try:
            yield self
        finally:
            module.CaptureLastBlock=original_capture
            self.tree.verify=original_verify
            self.tree.commit=original_commit
            self.adapter.memory=None


def configure():
    import state_draft,policy_inference
    import draft_a.pf_feature_unified as module
    original_factory=state_draft.make_draft
    original_configure=policy_inference.configure

    def factory(weight,kind,checkpoint,token_weight=None):
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved.get('draft_architecture')!='diffusion_context':
            return original_factory(weight,kind,checkpoint,token_weight)
        from diffusion_draft import DiffusionDraft
        initial=saved.get('initial') or saved['config']['initial']
        fusion=original_factory(weight,'fusion',initial,token_weight)
        config=saved['config']
        model=DiffusionDraft(fusion,token_weight,layers=config['layers'],memory_tokens=config['memory_tokens'])
        model.load_state_dict(saved['model_state'])
        model._pfspec_diffusion_context=True
        return model

    def policies():
        result=original_configure()
        original_adapter=module.MLPAdapter
        original_policy_factory=state_draft.make_draft

        def configured_factory(weight,kind,checkpoint,token_weight=None):
            model=original_policy_factory(weight,kind,checkpoint,token_weight)
            from pathlib import Path
            import json
            path=Path(checkpoint).parent/'inference.json'
            if path.exists():
                settings=json.loads(path.read_text())
                top_p=float(settings.get('draft_top_p',.9))
                if not 0<top_p<=1:
                    raise ValueError('Invalid recorded Draft sampling policy')
                model._pfspec_draft_top_p=top_p
                model._pfspec_feature_block_depth=int(settings.get('feature_block_depth',1))
                result[str(checkpoint)]=dict(temperature=.9,top_p=top_p)
            return model

        state_draft.make_draft=configured_factory

        class Adapter(original_adapter):
            def propose(self,prefix,p1,state,rng):
                if not getattr(self.feature,'_pfspec_diffusion_context',False):
                    return super().propose(prefix,p1,state,rng)
                from collect_feature_teacher_from_s3 import policy_batch
                self.forward_calls+=1
                self.input_tokens+=5
                inputs=self.memory.inputs(prefix,p1,self.next_position,state)
                logits=self._run('feature_forward',lambda:self.feature(**inputs),gpu=True,
                                 input_tokens=5,operation='masked_feature_block')
                top_p=float(getattr(self.feature,'_pfspec_draft_top_p',.9))
                def sampling():
                    if top_p==1:
                        return (logits.float()/.9).softmax(-1)[0]
                    return policy_batch(logits.flatten(0,1).float(),.9,top_p).reshape(1,2,4096)[0]
                probabilities=self._run('sampling',sampling,gpu=True,operation='feature_policy').float()
                self.second_distribution=probabilities[1]
                return probabilities[0]

        module.MLPAdapter=Adapter
        return result

    state_draft.make_draft=factory
    policy_inference.configure=policies
