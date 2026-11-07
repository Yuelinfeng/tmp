"""Enumerate complete candidate trees and output laws at up to four levels."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from univer_coupling import allocation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(20261108)
    records=[]
    for case in range(24):
        widths=[1,2,1,2] if case%2==0 else [2,2,2]
        depth=len(widths)
        n=3
        p={}
        q={}
        fixed={}
        rest={}
        for level in range(depth+1):
            for h in itertools.product(range(n),repeat=level):
                p[h]=rng.dirichlet(np.ones(n))
                if case%4==0:
                    p[h][0]=0;p[h]/=p[h].sum()
                if level==depth:continue
                q[h]=rng.dirichlet(np.ones(n))
                if widths[level]==1:
                    # PFSpec deterministic Position proposal.
                    token=int(np.argmax(q[h]));q[h][:]=0;q[h][token]=1
                fixed[h]=np.argsort(-q[h],kind='stable')[:widths[level]-1].tolist()
                rest[h]=q[h].copy();rest[h][fixed[h]]=0;rest[h]/=rest[h].sum()

        def trees(h,c):
            if len(h)==depth:
                yield 1.,dict(h=h,c=c,children=[])
                return
            for token in np.flatnonzero(rest[h]>0):
                a=allocation(c,p[h],rest[h],fixed[h],int(token))
                choices=[list(trees(h+(child,),conditional)) for child,conditional in zip(a['candidates'],a['conditional'])]
                for items in itertools.product(*choices):
                    probability=float(rest[h][token])
                    for prob,_ in items:probability*=prob
                    yield probability,dict(h=h,c=c,children=[item[1] for item in items],allocation=a)

        def events(node):
            for child in node['children']:yield from events(child)
            if not node['children']:
                yield node['h'],node['c'],p[node['h']]
            else:
                a=node['allocation'];mass=float(a['outside'].sum())
                yield node['h'],a['fallback'],a['outside']/mass if mass>0 else np.zeros(n)

        sequences=list(itertools.product(range(n),repeat=depth+1))
        index={s:i for i,s in enumerate(sequences)}
        observed=np.zeros(len(sequences))
        exact=np.zeros_like(observed)
        for s,i in index.items():
            exact[i]=np.prod([p[s[:level]][s[level]] for level in range(depth+1)])
        probability_sum=0.
        maximum_survival=0.
        tree_count=0
        for probability,tree in trees((),1.):
            tree_count+=1;probability_sum+=probability
            survival=1.
            for h,acceptance,bonus in events(tree):
                selected=survival*acceptance
                if selected>0:
                    for suffix in itertools.product(range(n),repeat=depth+1-len(h)):
                        s=h+suffix
                        value=bonus[suffix[0]]
                        for level in range(len(h)+1,depth+1):value*=p[s[:level]][s[level]]
                        observed[index[s]]+=probability*selected*value
                survival*=1-acceptance
            maximum_survival=max(maximum_survival,abs(survival))
        error=float(np.max(np.abs(observed-exact)))
        if error>1e-12 or abs(probability_sum-1)>1e-12 or maximum_survival>1e-12:
            raise ArithmeticError((case,error,probability_sum,maximum_survival))
        records.append(dict(case=case,widths=widths,tree_count=tree_count,complete_output_sequences=len(sequences),
            maximum_output_error=error,total_output_mass=float(observed.sum()),
            total_tree_probability=probability_sum,maximum_unresolved_probability=maximum_survival))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(status='complete',cases=records,
        maximum_output_error=max(x['maximum_output_error'] for x in records),
        scope='Independent exact enumeration of candidate randomness and every post-order decision, including complete conditional P/F/P/F analogues; no GPU or KV claim'),indent=2)+'\n')
    print(json.dumps(dict(cases=len(records),trees=sum(x['tree_count'] for x in records),maximum_output_error=max(x['maximum_output_error'] for x in records))))


if __name__=='__main__':main()
