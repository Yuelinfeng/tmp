"""Describe shared evaluation histories from saved formal W1 outputs."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    cases = json.loads(args.inputs.read_text())['contexts']
    records = [json.loads(line) for line in (args.evaluation/'records.jsonl').read_text().splitlines()]
    w1 = {r['pair_id']: r for r in records if r['method'] == 'w1'}
    metadata = {}
    for case in cases:
        row = w1[case['pair_id']]
        path = args.evaluation/'tokens'/f'{case["pair_id"]}_w1.pt'
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata[str(path)] = {k: case[k] for k in ('pair_id', 'source', 'cohort', 'seed', 'prefix')}
        metadata[str(path)].update(eos=row['eos'], generated_tokens=row['generated_tokens'])
    result = dict(train_files=[], val_files=list(metadata), source_metadata=metadata,
                  scope='Evaluation only; saved W1 histories; Target queried for shared labels',
                  inputs=str(args.inputs), evaluation=str(args.evaluation))
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(histories=len(metadata), output=str(args.out))))


if __name__ == '__main__':
    main()
