"""Package the selected weights, evidence, formal outputs and research sources."""
import argparse
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    artifacts = args.project / 'docs/project/artifacts/lossless_acceptance_20261106'
    formal = artifacts / 'formal_acceptance_01'
    for filename, count in [('natural_comparison.json', 270), ('capped_comparison.json', 360)]:
        result = json.loads((formal / filename).read_text())
        if result['status'] != 'complete' or result['verified_output_token_files'] != count:
            raise ValueError('Formal outputs are unfinished')
    selected_weights = {
        artifacts / 'paper06_dflash/training/diffusion_context_w64/selected.pt',
        artifacts / 'dependencies/fusion_long/selected.pt',
    }
    files = {}
    omitted_weights = []
    for path in artifacts.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        if path.suffix in {'.tgz', '.zip', '.gz'} or '__pycache__' in path.parts or path.name == 'handoff_archive_read_result.json':
            continue
        if path.suffix == '.pt' and path.stat().st_size > 1000000 and path not in selected_weights:
            omitted_weights.append(str(path.relative_to(artifacts)))
            continue
        files['results/' + str(path.relative_to(artifacts))] = path
    for directory in ['patches/pfspec_lossless_acceptance_20261106',
                      'patches/pfspec_feature_states_20261105',
                      'docs/project/artifacts/feature_acceptance_study/shared_backend_source_run04']:
        root = args.project / directory
        for path in root.rglob('*'):
            if path.is_file() and not path.is_symlink() and path.suffix in {'.py', '.sh', '.json', '.md'}:
                if '__pycache__' not in path.parts:
                    files['project/' + str(path.relative_to(args.project))] = path
    for name in ['FEATURE_DRAFT_LITERATURE_STUDY.md', 'ARCHITECTURE.md',
                 'OPERATIONS.md', 'EXPERIMENTS.md', 'LOGIC_VALIDATION.md', 'PROJECT_STATUS.md']:
        path = args.project / 'docs/project' / name
        if path.exists():
            files['project/' + str(path.relative_to(args.project))] = path
    start = args.project / '.codex/AI_START_HERE.md'
    files['project/.codex/AI_START_HERE.md'] = start
    for path in args.research.iterdir():
        if path.is_file() and (path.name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_'))
                or path.name.endswith(('_evidence.md', '_route.json', '_plan.json'))
                or path.name.startswith('univer_public_')):
            files['literature/' + path.name] = path
    official = args.research / 'dflash_official'
    for path in official.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.json', '.md'}:
            files['literature/dflash_official/' + str(path.relative_to(official))] = path
    if len([name for name in files if '/tokens/' in name and name.endswith('.pt')
            and name.startswith('results/formal_acceptance_01/')]) != 630:
        raise ValueError('Expected 630 formal token files')
    for path in selected_weights:
        if path not in files.values():
            raise ValueError('Selected weight or initialization dependency is missing')
    note = dict(status='complete', files=len(files), formal_token_files=630,
                selected_weight='results/paper06_dflash/training/diffusion_context_w64/selected.pt',
                initialization_dependency='results/dependencies/fusion_long/selected.pt',
                server_root='/root/autodl-tmp/pfspec_lossless_acceptance_20261106',
                retained_unselected_weights=sorted(omitted_weights),
                external_dependencies='Original Target, VQ, Ranker, runtime and large teacher data remain at documented server locations')
    note_path = artifacts / 'handoff_contents.json'
    note_path.write_text(json.dumps(note, indent=2) + '\n')
    files['results/handoff_contents.json'] = note_path
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError('Use a new handoff output filename')
    with tarfile.open(args.out, 'w:gz') as archive:
        for name, path in sorted(files.items()):
            archive.add(path, arcname='PFSpec_Lossless_Acceptance_20261107/' + name, recursive=False)
    print(json.dumps(dict(archive=str(args.out), bytes=args.out.stat().st_size, files=len(files))))


if __name__ == '__main__':
    main()
