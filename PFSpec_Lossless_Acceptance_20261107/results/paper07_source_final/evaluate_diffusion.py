"""Evaluate the masked context Draft with the existing Greedy candidate set."""
from pathlib import Path
import sys


def main():
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0]=[str(previous),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    from diffusion_inference import configure,LiveMemory
    configure()
    import multi_decoder,multi_runtime,multi_metrics
    from diffusion_decoder import decode as diffusion_decode
    from long_decoder import decode as long_decode
    from long_hybrid_runtime import HybridTargetTreeRuntime
    from long_metrics import configure as long_metrics_configure
    multi_runtime.MultiTargetTreeRuntime=HybridTargetTreeRuntime
    original_metrics_configure=multi_metrics.configure

    def metrics_configure(module):
        original_metrics_configure(module)
        long_metrics_configure(module)

    multi_metrics.configure=metrics_configure
    original_decode=multi_decoder.decode

    def decode(target,adapter,*args,**kwargs):
        if not getattr(adapter.feature,'_pfspec_diffusion_context',False):
            return original_decode(target,adapter,*args,**kwargs)
        memory=LiveMemory(adapter,kwargs['tree_runtime'])
        with memory.context():
            block_depth=int(getattr(adapter.feature,'_pfspec_feature_block_depth',1))
            run=long_decode if block_depth==2 else diffusion_decode
            stats,tokens=run(target,adapter,*args,**kwargs)
            stats.update(draft_architecture='diffusion_context',draft_masked_feature_outputs=2*adapter.forward_calls,
                draft_target_context_tokens=adapter.context_tokens,draft_block_input_tokens=5*adapter.forward_calls,
                draft_context_limit=adapter.feature.memory_tokens)
            return stats,tokens

    multi_decoder.decode=decode
    import evaluate
    original_write=evaluate.write

    def write(path,value):
        if Path(path).name=='plan.json':
            import json
            scopes={}
            for method,checkpoint in value['checkpoints'].items():
                folder=Path(checkpoint.get('path','')).parent
                settings=folder/'inference.json'
                settings=json.loads(settings.read_text()) if settings.is_file() else {}
                depth=int(settings.get('feature_block_depth',1))
                if method.startswith('diffusion_'):
                    scopes[method]=dict(masked_feature_predictions=2,consumed_feature_predictions=depth,
                        first_feature_candidates=64,second_feature_candidates=4 if depth==2 else 0,
                        maximum_target_nodes_per_full_round=642 if depth==2 else 130,
                        maximum_emitted_tokens_per_round=6 if depth==2 else 4,
                        second_feature_proposal_scope='Shared proposal set sampled independently of first selection; full Target conditional distributions for every first Feature and second Position')
            value['masked_block_scopes']=scopes
        original_write(path,value)

    evaluate.write=write
    from evaluate_multidraft import main as evaluate
    evaluate()


if __name__=='__main__':
    main()
