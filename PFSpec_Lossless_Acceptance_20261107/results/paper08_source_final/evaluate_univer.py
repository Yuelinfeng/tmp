"""Evaluate UniVer with the same committed-context Draft and Target backend."""
import json,sys
from pathlib import Path


def main():
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    old=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    sys.path[:0]=[str(old),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    import long_decoder,long_hybrid_runtime
    from univer_decoder import decode
    from univer_tree_runtime import UniVerTargetTreeRuntime
    long_decoder.decode=decode
    long_hybrid_runtime.LongTargetTreeRuntime=UniVerTargetTreeRuntime
    original_stats=long_hybrid_runtime.HybridTargetTreeRuntime.stats

    def stats(self):
        value=original_stats(self)
        if self.long_runtime is not None:
            value['long_tree_execution_reason']='Independent per-parent UniVer candidates; ordinary execution validated on full branches and retained numeric cases'
        return value

    long_hybrid_runtime.HybridTargetTreeRuntime.stats=stats
    import evaluate
    original_write=evaluate.write

    def write(path,value):
        if Path(path).name=='plan.json':
            for method in value['methods']:
                if 'univer' not in method:continue
                width=value['feature_candidates'][method]['requested_feature_candidates']
                value['masked_block_scopes'][method].update(first_feature_candidates=width,
                    maximum_target_nodes_per_full_round=2+10*width,
                    second_feature_proposal_scope='Three deterministic second candidates plus an independent remaining draw for every first Feature parent',
                    verification='UniVer conditional allocation and post-order decision')
        original_write(path,value)

    evaluate.write=write
    from evaluate_diffusion import main as run
    run()


if __name__=='__main__':main()
