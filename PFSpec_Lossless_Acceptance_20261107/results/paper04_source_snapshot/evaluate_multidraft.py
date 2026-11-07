"""Keep the existing evaluation, changing only the Feature set coupling."""
from pathlib import Path
import sys
import time


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
    warmups=[]
    startup={}
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
            result=multi_decode(target,adapter,*args,**kwargs)
        else:
            result=original_decode(target,adapter,*args,**kwargs)
        if int(args[3])==16:
            tree=kwargs['tree_runtime']
            warmups.append(dict(requested_tokens=16,seed=int(args[4]),stats=result[0],
                runtime=tree.linear_runtime.stats(),tree=tree.stats()))
        return result

    state_draft.make_draft=factory
    tree_decoder.decode=decode
    tree_runtime.TargetTreeRuntime=MultiTargetTreeRuntime
    import evaluate
    original_context=evaluate.context
    original_normalize=acceptance_metrics.normalize
    original_write=evaluate.write

    def context(args):
        start=time.perf_counter()
        result=original_context(args)
        torch.cuda.synchronize()
        startup.update(context_initialization_seconds=time.perf_counter()-start,
            context_time_scope='Loading the common Target, Ranker and evaluation context before warmup or generation',
            gpu_allocated_after_context_gib=torch.cuda.memory_allocated()/2**30)
        return result

    def normalize(row):
        row['peak_reserved_gpu_gib']=torch.cuda.max_memory_reserved()/2**30
        return original_normalize(row)

    def write(path,value):
        if Path(path).name=='plan.json':
            value['target_sampling']=dict(temperature=.9,top_p=.9)
            value['feature_candidates']={method:scopes.get(value['checkpoints'].get(method,{}).get('path'),
                dict(requested_feature_candidates=1,coupling='Original single candidate',draft_policy=dict(temperature=.9,top_p=.9)))
                for method in value['methods'] if method not in ('opt_target','ranker_only')}
            value['feature_metric_scope']='Single overlap and multi-candidate set acceptance are different quantities'
            value['startup']=startup
        if Path(path).name=='runtime.json':
            from runtime_accounting import initialization
            for item in value.values():item['graph_initialization_accounting']=initialization(item['runtime'],item.get('tree'))
        if Path(path).name=='progress.json' and value.get('status')=='complete':
            from runtime_accounting import initialization
            for item in warmups:item['graph_initialization_accounting']=initialization(item['runtime'],item['tree'])
            original_write(Path(path).parent/'startup_and_warmup.json',dict(startup=startup,warmups=warmups,
                scope='Discarded 16-token warmup requests retain all logical counters and graph initialization work'))
        original_write(path,value)

    evaluate.context=context
    acceptance_metrics.normalize=normalize
    evaluate.write=write
    evaluate.main()


if __name__=='__main__':
    main()
