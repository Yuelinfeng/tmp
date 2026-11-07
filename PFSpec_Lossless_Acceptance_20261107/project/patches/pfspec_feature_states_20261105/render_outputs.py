"""Decode saved research outputs to real Gaussians and six camera views."""
import argparse
import json
from pathlib import Path

from research import setup, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('/root/workspace/GaussianGPT'))
    parser.add_argument('--source', type=Path, default=Path('/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04'))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--tokens', type=Path, nargs='+')
    inputs.add_argument('--evaluation', type=Path)
    parser.add_argument('--methods', default='fusion,fusion_long')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.mkdir(parents=True)
    if args.evaluation:
        records = [json.loads(line) for line in (args.evaluation/'records.jsonl').read_text().splitlines()]
        selected = []
        for method in args.methods.split(','):
            for cohort in ('bos', 'completion'):
                candidates = [row for row in records if row['method']==method and row['cohort']==cohort
                              and row['eos'] and row['generated_tokens']>1]
                if not candidates:
                    raise ValueError(f'No nonempty EOS output for {method}/{cohort}')
                selected.append(candidates[0])
        args.tokens = [args.evaluation/'tokens'/f'{row["pair_id"]}_{row["method"]}.pt' for row in selected]
        write(args.out/'selection.json', dict(
            criterion='First nonempty natural-EOS output per method/cohort in recorded order; no appearance ranking',
            evaluation=str(args.evaluation), rows=[{k:row[k] for k in
                ('pair_id','source','cohort','method','seed','prefix','generated_tokens','eos')} for row in selected]))
    setup(args)
    import sys
    dependencies = args.project.parent/'deps'
    if dependencies.is_dir():
        sys.path.insert(0, str(dependencies))
    import numpy as np
    import torch
    from model.gaussian_vqvae import GaussianVQVAE
    from draft_a.pf_multiscene.full_3d_eval import camera_views, gaussian_tensors, render_views, save_views
    with torch.inference_mode():
        vq = GaussianVQVAE.load_from_checkpoint(str(args.project/'checkpoints/vqvae_vfront.ckpt'),
                                               map_location='cpu').cuda().float().eval().requires_grad_(False)
        cameras = camera_views(20*vq.autoencoder.stride*vq.autoencoder.default_grid_size)
        write(args.out/'cameras.json', cameras)
        reports = []
        for path in args.tokens:
            saved = torch.load(path, map_location='cpu', weights_only=False)
            suffix = list(saved['suffix'])
            eos = bool(suffix and int(suffix[-1]) == 12096)
            body = list(saved['prompt'][1:])+suffix[:-1] if eos else list(saved['prompt'][1:])+suffix
            body = [int(x) for x in body]
            if not body or len(body)%2:
                raise ValueError('Expected complete Position/Feature pairs')
            tensors, quality = gaussian_tensors(vq, body)
            if not all(torch.isfinite(x).all().item() for x in tensors.values()):
                raise ArithmeticError('Nonfinite Gaussian parameter')
            directory = args.out/path.stem
            directory.mkdir()
            np.savez_compressed(directory/'gaussians.npz', **{k:v.cpu().numpy() for k,v in tensors.items()})
            rgb, alpha = render_views(tensors, cameras)
            save_views(rgb, directory)
            if tuple(rgb.shape) != (6,384,384,3) or not torch.isfinite(rgb).all().item():
                raise ArithmeticError('Invalid rendered views')
            reports.append(dict(source=str(path), tokens=len(body), eos=eos, quality=quality,
                                images=str(directory), dimensions=list(rgb.shape)))
            write(args.out/'results.json', dict(status='running', scope='Functional VQ decoding and rendering; no request timing', rows=reports))
            print(path.stem, len(body), flush=True)
        write(args.out/'results.json', dict(status='complete', scope='Functional VQ decoding and rendering; no request timing', rows=reports))


if __name__ == '__main__':
    main()
