"""Section 5 of arXiv:2502.18779v2, using the actual proposal law."""


def candidates(q, width, rng):
    import torch
    if q.ndim!=1 or not bool(torch.isfinite(q).all()) or bool((q<0).any()):
        raise ValueError('Invalid draft probabilities')
    active=int((q>0).sum())
    if active==0 or width<1:
        raise ValueError('Candidate construction needs positive draft support')
    width=min(int(width),active)
    ordered=torch.argsort(q,descending=True,stable=True)
    fixed=ordered[:width-1]
    rest=q.clone()
    if fixed.numel():
        rest[fixed]=0
        rest=rest/rest.sum()
    random_token=int(torch.multinomial(rest,1,generator=rng))
    return [int(x) for x in fixed]+[random_token],rest


def select(p, ids, rest, rng, decide, draw):
    from tree_sampling import torch_acceptance_probability,torch_positive_residual
    random_token=ids[-1]
    alpha=torch_acceptance_probability(p,rest,random_token)
    random_accepted=bool(decide(alpha,rng,p.device))
    token=random_token if random_accepted else int(draw(torch_positive_residual(p,rest),rng))
    branch=ids.index(token) if token in ids else None
    return token,branch,random_accepted


def expected_acceptance(p,q,width):
    import torch
    width=min(int(width),int((q>0).sum()))
    if width<1:
        raise ValueError('Candidate construction needs positive draft support')
    fixed=torch.argsort(q,descending=True,stable=True)[:width-1]
    rest=q.clone()
    if fixed.numel():
        rest[fixed]=0
        rest=rest/rest.sum()
    return p[fixed].sum()+torch.minimum(p,rest).sum()
