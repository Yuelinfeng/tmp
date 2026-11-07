"""Enumerate remaining draws and verify the shallow P/F/P equivalence."""
import argparse,json
from pathlib import Path
import numpy as np
from univer_coupling import allocation


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(20261107)
    cases=[]
    for i in range(96):
        n=3+i%5
        p=rng.dirichlet(np.ones(n))
        q=rng.dirichlet(np.ones(n))
        if i%4==0:
            p[0]=0;p/=p.sum()
        width=1+i%min(n,4)
        fixed=np.argsort(-q,kind='stable')[:width-1].tolist()
        rest=q.copy();rest[fixed]=0;rest/=rest.sum()
        c=[0.,1.,float(rng.random())][i%3]
        c2=rng.random(n)
        if i%5==0:c2[0]=0
        if i%7==0:c2[-1]=1
        expected=np.zeros(n)
        expected_rejection=0.
        pair=0.
        triple=0.
        normalization_error=0.
        traversal_error=0.
        for sampled in np.flatnonzero(rest>0):
            a=allocation(c,p,rest,fixed,int(sampled))
            expected+=rest[sampled]*a['mass']
            expected_rejection+=rest[sampled]*a['rejected']
            normalization_error=max(normalization_error,abs(a['mass'].sum()+a['rejected']-1))
            survival=1.
            for token,conditional in zip(a['candidates'],a['conditional']):
                # Each Feature branch has one deterministic second Position.
                # Its accepted subtree mass is split between P2 and F1 fallback.
                accepted_p2=conditional*c2[token]
                fallback=(conditional-accepted_p2)/(1-accepted_p2) if accepted_p2<1 else 0.
                accepted_feature=accepted_p2+(1-accepted_p2)*fallback
                traversal_error=max(traversal_error,abs(survival*accepted_feature-a['mass'][token]))
                pair+=rest[sampled]*survival*accepted_feature
                triple+=rest[sampled]*survival*accepted_p2
                survival*=1-accepted_feature
        block_mass=np.minimum(c*p,rest)
        block_mass[fixed]=c*p[fixed]
        expected_pair=float(block_mass.sum())
        expected_triple=float(block_mass@c2)
        cases.append(dict(case=i,classes=n,width=width,prefix_mass=c,
            normalization_error=normalization_error,
            local_target_error=float(np.max(np.abs(expected-c*p))),
            rejection_error=abs(expected_rejection-(1-c)),
            traversal_error=traversal_error,
            shallow_pair_difference=abs(pair-expected_pair),
            shallow_triple_difference=abs(triple-expected_triple),
            accepted_length_difference=abs((c+pair+triple)-(c+expected_pair+expected_triple))))
    fields=[k for k in cases[0] if 'error' in k or 'difference' in k]
    maximum={k:max(a[k] for a in cases) for k in fields}
    if max(maximum.values())>1e-12:raise ArithmeticError(maximum)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(status='complete',cases=cases,maximum=maximum,
        scope='Local equations and deterministic P/F/P specialization; deeper tree law and GPU behavior are not established by this experiment'),indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),maximum=maximum)))


if __name__=='__main__':main()
