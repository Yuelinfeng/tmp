"""Prepare independent acceptance inputs before selecting any new final weights."""
import argparse
import json
from pathlib import Path
import re


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--catalog',type=Path,action='append',required=True)
    parser.add_argument('--previous-formal',type=Path,required=True)
    parser.add_argument('--natural-supervision-plan',type=Path,required=True)
    parser.add_argument('--feedback-plan',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    import torch
    if args.out.exists():
        raise FileExistsError(args.out)
    prior=json.loads(args.previous_formal.read_text())['contexts']
    feedback=json.loads(args.feedback_plan.read_text())['contexts']
    excluded={str(c['source']) for c in prior+feedback}
    excluded.update(re.findall(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
                              args.natural_supervision_plan.read_text()))
    selected={}
    catalog_counts=[]
    for path in args.catalog:
        rows=torch.load(path,map_location='cpu',weights_only=True)
        added=0
        for index,row in enumerate(rows):
            source=str(row.get('scene') or row.get('source') or Path(str(row['path'])).parents[1].name)
            if source in excluded or source in selected:
                continue
            tokens=row['tokens'].tolist() if hasattr(row['tokens'],'tolist') else list(row['tokens'])
            tokens=[int(t) for t in tokens]
            if len(tokens)<260 or len(tokens)%2:
                continue
            prefix=tokens[:256]
            if not all(0<=p<8000 for p in prefix[::2]) or not all(8000<=f<12096 for f in prefix[1::2]):
                raise ValueError(f'Invalid P/F ranges in {source}')
            selected[source]=dict(source=source,tokens=tokens,catalog=str(path),row=index,
                                  original_path=str(row.get('path','')))
            added+=1
            if len(selected)==30:
                break
        catalog_counts.append(dict(path=str(path),selected_sources=added))
        if len(selected)==30:
            break
    if len(selected)!=30:
        raise ValueError(f'Only {len(selected)} unused sources have a valid 256-token prefix')
    natural,capped=[],[]
    for index,(source,row) in enumerate(selected.items()):
        for seed in (73101,73102):
            natural.append(dict(pair_id=f'new-completion-{index:02d}-{seed}',source=source,
                cohort='completion',prefix=256,seed=seed,prompt=[12096]+row['tokens'][:256],budget=16000))
        for length in (64,256):
            for seed in (73101,73102):
                capped.append(dict(pair_id=f'new-capped-{index:02d}-{length}-{seed}',source=source,
                    cohort='capped',prefix=length,seed=seed,prompt=[12096]+row['tokens'][:length],budget=512))
    for seed in range(73201,73231):
        source=f'new-bos-{seed}'
        if source in excluded or any(c.get('seed')==seed for c in prior+feedback):
            raise ValueError('A final BOS seed was used for training or development')
        natural.append(dict(pair_id=source,source=source,cohort='bos',prefix=0,seed=seed,
                            prompt=[12096],budget=16000))
    args.out.mkdir(parents=True)
    for name,contexts in [('natural_inputs.json',natural),('capped_inputs.json',capped)]:
        (args.out/name).write_text(json.dumps(dict(contexts=contexts,scope='Reserved for final acceptance evaluation; '
            'excluded from current drafter training and weight selection'),indent=2)+'\n')
    (args.out/'sources.json').write_text(json.dumps(dict(
        sources=[{k:v for k,v in row.items() if k!='tokens'} for row in selected.values()],
        catalog_counts=catalog_counts,excluded_sources=sorted(excluded),
        previous_formal=str(args.previous_formal),natural_supervision=str(args.natural_supervision_plan),
        feedback=str(args.feedback_plan),
        source_scope='Catalog labels refer to older room studies. This reserve is independent of the '
                     'current W1/fusion_long training and new candidate weight selection, not a claim '
                     'that no room was ever used for another historical experiment.'),indent=2)+'\n')
    print(json.dumps(dict(status='complete',sources=30,natural=len(natural),capped=len(capped),
                          catalog_counts=catalog_counts)),flush=True)


if __name__=='__main__':
    main()
