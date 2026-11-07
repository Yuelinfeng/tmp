"""Algorithm 2 of arXiv:2403.10444v3 for a point Position and one Feature."""


def parameters(position_probability, feature_p, feature_q, random_token):
    import torch
    c=float(position_probability)
    residual=(c*feature_p-feature_q).clamp_min(0)
    mass=float(residual.sum())
    denominator=mass+1-c
    first=mass/denominator if denominator>0 else 0.
    second=min(1.,c*float(feature_p[random_token])/float(feature_q[random_token]))
    return first,second,residual


def select(position_probability, feature_p, ids, feature_q, rng, decide, draw):
    first,second,residual=parameters(position_probability,feature_p,feature_q,ids[-1])
    accepted_position=bool(decide(first,rng,feature_p.device))
    accepted_pair=bool(decide(second,rng,feature_p.device))
    if accepted_pair:
        return 2,ids[-1],len(ids)-1
    if not accepted_position:
        return 0,None,None
    mass=residual.sum()
    if float(mass)<=0:
        raise ArithmeticError('A selected one-token prefix needs positive Feature residual mass')
    token=int(draw(residual/mass,rng))
    return 1,token,ids.index(token) if token in ids else None


def expected_joint(position_probability, feature_p, feature_q, width):
    import torch
    c=float(position_probability)
    width=min(int(width),int((feature_q>0).sum()))
    if width<1:raise ValueError('Positive Draft support required')
    fixed=torch.argsort(feature_q,descending=True,stable=True)[:width-1]
    rest=feature_q.clone()
    if fixed.numel():
        rest[fixed]=0
        rest/=rest.sum()
    return c*feature_p[fixed].sum()+torch.minimum(c*feature_p,rest).sum()
