"""Enumerate the complete conditional P/F law across both Feature levels."""
import argparse
import json
from pathlib import Path
import numpy as np


def normalize(x):
    return x/x.sum(axis=-1,keepdims=True)


def greedy_outcomes(p,q,width):
    support=np.flatnonzero(q>0)
    fixed=np.argsort(-q)[:min(width-1,len(support)-1)]
    remainder=q.copy()
    remainder[fixed]=0
    remainder/=remainder.sum()
    residual=np.maximum(p-remainder,0)
    residual=residual/residual.sum() if residual.sum()>0 else p
    outcomes=[]
    for random in np.flatnonzero(remainder>0):
        accepted=min(1.,p[random]/remainder[random])
        law=(1-accepted)*residual.copy()
        law[random]+=accepted
        membership=np.zeros(len(p),dtype=bool)
        membership[fixed]=True
        membership[random]=True
        outcomes.append((remainder[random],law,membership))
    return outcomes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(20261106)
    errors=[]
    greedy_errors=[]
    for case in range(48):
        p1=normalize(rng.random(3))
        f1=normalize(rng.random((3,7)))
        p2=normalize(rng.random((3,7,3)))
        f2=normalize(rng.random((3,7,3,7)))
        q1=normalize(rng.random(7))
        q2=normalize(rng.random(7))
        if case%8==0:
            p1[0]=0;p1=normalize(p1)
        if case%8==1:
            p1[:]=0;p1[0]=1
        if case%8==2:
            p2[:,:,1]=0;p2=normalize(p2)
        if case%8==3:
            p2[:]=0;p2[:,:,1]=1
        if case%8==4:
            q1[:]=0;q1[2]=1
        if case%8==5:
            q2[:]=0;q2[4]=1
        expected=p1[:,None,None,None]*f1[:,:,None,None]*p2[:,:,:,None]*f2
        actual=np.zeros_like(expected)
        actual[1:]=expected[1:]
        for weight1,law1,hit1 in greedy_outcomes(f1[0],q1,4):
            for first_feature in range(7):
                first_mass=p1[0]*weight1*law1[first_feature]
                if not hit1[first_feature]:
                    actual[0,first_feature]+=first_mass*p2[0,first_feature,:,None]*f2[0,first_feature]
                    continue
                for second_position in range(3):
                    mass=first_mass*p2[0,first_feature,second_position]
                    if second_position!=1:
                        actual[0,first_feature,second_position]+=mass*f2[0,first_feature,second_position]
                        continue
                    second_law=np.zeros(7)
                    for weight2,law2,_ in greedy_outcomes(f2[0,first_feature,1],q2,4):
                        second_law+=weight2*law2
                    greedy_errors.append(float(np.max(np.abs(second_law-f2[0,first_feature,1]))))
                    actual[0,first_feature,1]+=mass*second_law
        errors.append(float(np.max(np.abs(actual-expected))))
        if abs(actual.sum()-1)>1e-12:
            raise ArithmeticError('Complete P/F mass differs')
    if max(errors)>1e-12 or max(greedy_errors)>1e-12:
        raise ArithmeticError('Two Feature conditional law differs')
    result=dict(status='complete',cases=len(errors),complete_pf_joint_max_error=max(errors),
        second_feature_conditional_max_error=max(greedy_errors),
        scope='Different full Target Feature distributions for every first Feature and second Position; shared second Draft proposals independent of first selection; zero and certain Position proposals included')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
