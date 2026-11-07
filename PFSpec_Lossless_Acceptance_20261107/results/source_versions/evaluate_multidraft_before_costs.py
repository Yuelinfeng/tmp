"""Keep the existing evaluation, changing only the Feature set coupling."""
from pathlib import Path
import sys


def main():
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0]=[str(previous),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),
                  '/root/workspace/GaussianGPT']
    from policy_inference import configure
    policies=configure()
    import torch,state_draft,tree_decoder,tree_runtime,acceptance_metrics
    from multi_runtime import MultiTargetTreeRuntime
    from multi_decoder import decode as multi_decode
    from multi_metrics import configure as metrics_configure
    metrics_configure(acceptance_metrics)
    original_factory=state_draft.make_draft
    original_decode=tree_decoder.decode
    scopes={}
    MultiTargetTreeRuntime.maximum_width=1

    def factory(weight,kind,checkpoint,token_weight=None):
        model=original_factory(weight,kind,checkpoint,token_weight)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        width=int(saved.get('feature_candidate_width',1))
        model._pfspec_multi_width=width
        MultiTargetTreeRuntime.maximum_width=max(MultiTargetTreeRuntime.maximum_width,width)
        scopes[str(checkpoint)]=dict(requested_feature_candidates=width,
            coupling='Greedy fixed set plus one random remaining candidate' if width>1 else 'Original single candidate',
            draft_policy=policies[str(checkpoint)])
        return model

    def decode(target,adapter,*args,**kwargs):
        width=int(getattr(adapter.feature,'_pfspec_multi_width',1))
        if width>1:
            kwargs['width']=width
            return multi_decode(target,adapter,*args,**kwargs)
        return original_decode(target,adapter,*args,**kwargs)

    state_draft.make_draft=factory
    tree_decoder.decode=decode
    tree_runtime.TargetTreeRuntime=MultiTargetTreeRuntime
    import evaluate
    original_write=evaluate.write

    def write(path,value):
        if Path(path).name=='plan.json':
            value['target_sampling']=dict(temperature=.9,top_p=.9)
            value['feature_candidates']={method:scopes.get(value['checkpoints'].get(method,{}).get('path'),
                dict(requested_feature_candidates=1,coupling='Original single candidate',draft_policy=dict(temperature=.9,top_p=.9)))
                for method in value['methods'] if method not in ('opt_target','ranker_only')}
            value['feature_metric_scope']='Single overlap and multi-candidate set acceptance are different quantities'
        original_write(path,value)

    evaluate.write=write
    evaluate.main()


if __name__=='__main__':
    main()
