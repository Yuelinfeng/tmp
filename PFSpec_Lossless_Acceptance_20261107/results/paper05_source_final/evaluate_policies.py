"""Run the unchanged natural evaluation with explicitly recorded draft policies."""
from pathlib import Path
import sys


def main():
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0]=[str(previous),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),
                  '/root/workspace/GaussianGPT']
    from policy_inference import configure
    policies=configure()
    import evaluate
    original_write=evaluate.write

    def write(path,value):
        if Path(path).name=='plan.json':
            value['target_sampling']=dict(temperature=.9,top_p=.9)
            value['draft_sampling']={method:policies.get(value['checkpoints'].get(method,{}).get('path'),dict(temperature=.9,top_p=.9))
                                     for method in value['methods'] if method not in ('opt_target','ranker_only')}
            value['top_p_scope']='Target; per-method draft policies are listed separately'
        original_write(path,value)

    evaluate.write=write
    evaluate.main()


if __name__=='__main__':
    main()
