"""Collect completed selected-method outputs and the three-method reports."""
import json
import tarfile
from pathlib import Path


def main():
    root = Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106')
    if json.loads((root / 'formal_best_process.json').read_text())['status'] != 'complete':
        raise RuntimeError('Selected-method evaluation is unfinished')
    formal = root / 'formal_acceptance_01'
    for name, count in [('natural_comparison.json', 270), ('capped_comparison.json', 360)]:
        report = json.loads((formal / name).read_text())
        if report['status'] != 'complete' or report['verified_output_token_files'] != count:
            raise ValueError('Complete token evidence is required')
    files = []
    for name in ['diffusion_full_w64_natural', 'diffusion_full_w64_capped', 'common_histories']:
        files.extend(p for p in (formal / name).rglob('*') if p.is_file() and not p.is_symlink())
    files.extend([formal / 'natural_comparison.json', formal / 'capped_comparison.json',
                  root / 'formal_best_process.json', root / 'formal_best.log',
                  root / 'selection_before_final.json'])
    output = root / 'formal_best_deliverables.tgz'
    with tarfile.open(output, 'w:gz') as archive:
        for path in sorted(set(files)):
            archive.add(path, arcname=path.relative_to(root), recursive=False)
    print(json.dumps(dict(archive=str(output), bytes=output.stat().st_size,
                          files=len(set(files)), token_files=sum(
                              p.suffix == '.pt' and 'tokens' in p.parts for p in files))))


if __name__ == '__main__':
    main()
