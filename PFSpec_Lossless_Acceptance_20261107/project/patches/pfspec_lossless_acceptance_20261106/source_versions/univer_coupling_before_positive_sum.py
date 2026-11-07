"""Conditional mass allocation from UniVer equations 3–7.

Inputs are the complete Target probabilities, remaining Draft probabilities,
ordered fixed candidates, one independent remaining draw, and prefix mass.
The returned allocation includes outcomes outside the candidates and rejection.
"""


def allocation(prefix_mass, target, remaining, fixed, sampled):
    import numpy as np
    p=np.asarray(target,dtype=np.float64)
    q=np.asarray(remaining,dtype=np.float64)
    c=float(prefix_mass)
    fixed=[int(x) for x in fixed]
    sampled=int(sampled)
    if not 0<=c<=1 or p.ndim!=1 or q.shape!=p.shape:
        raise ValueError('Invalid conditional allocation input')
    if np.any(p<0) or np.any(q<0) or abs(p.sum()-1)>1e-12 or abs(q.sum()-1)>1e-12:
        raise ValueError('Normalized nonnegative probabilities required')
    if len(set(fixed))!=len(fixed) or sampled in fixed or q[sampled]<=0 or np.any(q[fixed]!=0):
        raise ValueError('Fixed candidates and remaining draw are inconsistent')
    accepted=min(1.,c*p[sampled]/q[sampled])
    residual=np.maximum(c*p-q,0.)
    denominator=1-c+float(residual.sum())
    if denominator==0:
        if accepted!=1:raise ArithmeticError('Zero residual mass requires certain acceptance')
        mass=np.zeros_like(p)
        rejected=0.
    else:
        mass=residual*((1-accepted)/denominator)
        rejected=(1-accepted)*(1-c)/denominator
    mass[sampled]=accepted
    ids=fixed+[sampled]
    conditional=[]
    used=0.
    for token in ids:
        available=1-used
        conditional.append(float(mass[token]/available) if available>0 else 0.)
        used+=float(mass[token])
    outside=mass.copy()
    outside[ids]=0.
    available=1-used
    fallback=float(outside.sum()/available) if available>0 else 0.
    return dict(mass=mass,rejected=float(rejected),candidates=ids,
                conditional=conditional,outside=outside,fallback=fallback)
