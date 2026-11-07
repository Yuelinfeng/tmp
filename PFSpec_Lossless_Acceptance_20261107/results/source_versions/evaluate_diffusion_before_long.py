"""Evaluate the masked context Draft with the existing Greedy candidate set."""
from pathlib import Path
import sys


def main():
    previous=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0]=[str(previous),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    from diffusion_inference import configure,LiveMemory
    configure()
    import multi_decoder
    from diffusion_decoder import decode as diffusion_decode
    original_decode=multi_decoder.decode

    def decode(target,adapter,*args,**kwargs):
        if not getattr(adapter.feature,'_pfspec_diffusion_context',False):
            return original_decode(target,adapter,*args,**kwargs)
        memory=LiveMemory(adapter,kwargs['tree_runtime'])
        with memory.context():
            stats,tokens=diffusion_decode(target,adapter,*args,**kwargs)
            stats.update(draft_architecture='diffusion_context',draft_masked_feature_outputs=2*adapter.forward_calls,
                draft_target_context_tokens=adapter.context_tokens,draft_block_input_tokens=5*adapter.forward_calls,
                draft_context_limit=adapter.feature.memory_tokens)
            return stats,tokens

    multi_decoder.decode=decode
    from evaluate_multidraft import main as evaluate
    evaluate()


if __name__=='__main__':
    main()
