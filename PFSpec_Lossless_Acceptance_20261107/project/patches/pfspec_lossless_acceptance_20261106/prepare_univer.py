"""Save the selected model with explicit new proposal and verification settings."""
import argparse,json
from pathlib import Path
import torch


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--width',type=int,default=16)
    options=parser.parse_args()
    if options.width not in [16,32,64]:raise ValueError('Unsupported candidate width')
    root=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
    source=root/'paper06_dflash/training/diffusion_context_w64/selected.pt'
    folder=root/f'paper08_univer/training/diffusion_univer_w{options.width}'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'selected.pt'
    if path.exists():raise FileExistsError(path)
    saved=torch.load(source,map_location='cpu',weights_only=False)
    saved.update(feature_candidate_width=options.width,feature_block_depth=2,draft_top_p=1.,
                 verification='UniVer',starting_checkpoint=str(source),weight_update_steps=0)
    torch.save(saved,path)
    (folder/'inference.json').write_text(json.dumps(dict(draft_top_p=1.,feature_block_depth=2,
        first_feature_candidates=options.width,second_feature_candidates_per_parent=4,
        second_candidate_randomness='Independent remaining draw for every first Feature parent',
        target_temperature=.9,target_top_p=.9,verification='UniVer',
        selected_training_step=2800,weight_update_steps=0,source_checkpoint=str(source)),indent=2)+'\n')
    print('Prepared',path.stat().st_size,flush=True)


if __name__=='__main__':main()
