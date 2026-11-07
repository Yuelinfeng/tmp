"""Select block verification only for checkpoints that explicitly request it."""


def main():
    import sys
    from pathlib import Path
    old=Path('/root/autodl-tmp/pfspec_feature_states_20261105/source')
    shared=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    sys.path[:0]=[str(old),str(shared),str(shared/'reference'),str(shared/'reference/runtime'),'/root/workspace/GaussianGPT']
    import torch,policy_inference,state_draft,tree_decoder,multi_decoder
    from block_decoder import decode as block_decode
    original_configure=policy_inference.configure
    original_single=tree_decoder.decode
    original_multi=multi_decoder.decode
    blocks={}

    def configure():
        policies=original_configure()
        original_factory=state_draft.make_draft
        def factory(weight,kind,checkpoint,token_weight=None):
            model=original_factory(weight,kind,checkpoint,token_weight)
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            model._pfspec_block_pair=bool(saved.get('block_pair_verification',False))
            blocks[str(checkpoint)]=model._pfspec_block_pair
            return model
        state_draft.make_draft=factory
        return policies

    def single(target,adapter,*args,**kwargs):
        if getattr(getattr(adapter,'feature',None),'_pfspec_block_pair',False):
            return block_decode(target,adapter,*args,**kwargs)
        return original_single(target,adapter,*args,**kwargs)

    def multiple(target,adapter,*args,**kwargs):
        if getattr(getattr(adapter,'feature',None),'_pfspec_block_pair',False):
            return block_decode(target,adapter,*args,**kwargs)
        return original_multi(target,adapter,*args,**kwargs)

    policy_inference.configure=configure
    tree_decoder.decode=single
    multi_decoder.decode=multiple
    import evaluate
    original_write=evaluate.write
    def write(path,value):
        if Path(path).name=='plan.json':
            value['block_pair_verification']={m:blocks.get(v.get('path'),False)
                for m,v in value['checkpoints'].items()}
            value['block_scope']='Algorithm 2 with two-token P/F blocks and reuse of fixed Feature branches'
        original_write(path,value)
    evaluate.write=write
    from evaluate_multidraft import main as evaluate
    evaluate()


if __name__=='__main__':main()
