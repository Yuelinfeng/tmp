"""Retain complete evaluator time and actual Draft parameter counts for final runs."""
from pathlib import Path
import sys,time


def main():
    started=time.perf_counter()
    old=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0]=[str(old),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    import evaluate
    original_load=evaluate.load_method
    original_write=evaluate.write
    initial_drafts=[]

    def load(method,context):
        begun=time.perf_counter()
        adapter=original_load(method,context)
        if method=='pf_mlp':
            model=adapter.model.feature
            initial_drafts.append(dict(seconds=time.perf_counter()-begun,
                parameters=sum(p.numel() for p in model.parameters()),
                inference_grad_enabled_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                model_type=type(model).__name__))
        return adapter

    def write(path,value):
        name=Path(path).name
        if name=='plan.json':
            value['initial_draft_loads']=initial_drafts
            if 'w1' in value['methods']:
                value['checkpoints']['w1']=dict(
                    path='/root/workspace/GaussianGPT/draft_a/runs/feature_target_continue_20261014/selected.pt',
                    kind='Original W1',parameters=initial_drafts[0]['parameters'])
            value['selection_scope']='Formal inputs are excluded from training and prespecified method selection'
        if name in ('progress.json','validation.json') and value.get('status')=='complete':
            value['full_evaluator_seconds']=time.perf_counter()-started
            value['full_time_scope']='Imports, context, Draft loading, warmup, generation and saved results'
        original_write(path,value)

    evaluate.load_method=load
    evaluate.write=write
    method_text=sys.argv[sys.argv.index('--methods')+1] if '--methods' in sys.argv else ''
    if any(method.startswith('diffusion_') for method in method_text.split(',')):
        from evaluate_diffusion import main as run
    else:
        from evaluate_block import main as run
    run()


if __name__=='__main__':main()
