"""Exhaustive conditional-law reconstruction for the actual greedy proposal."""
import json,sys
from pathlib import Path


def main():
    import torch
    from scipy.optimize import linprog
    from greedy_coupling import candidates,expected_acceptance,select
    sys.path.insert(0,'/root/autodl-tmp/gaussiangpt_feature_tree_20261021/source_run04')
    rng=torch.Generator().manual_seed(20261106)
    examples=[([.1,.2,.7],[.6,.3,.1]),([0,.4,.6],[.7,.3,0]),([.2,.3,.5],[0,1,0]),
              ([.2,.3,.5],[.2,.3,.5])]
    examples.extend((torch.rand(7,generator=rng).tolist(),torch.rand(7,generator=rng).tolist()) for _ in range(12))
    rows=[]
    for example,(px,qx) in enumerate(examples):
        p=torch.tensor(px,dtype=torch.float64);p/=p.sum()
        q=torch.tensor(qx,dtype=torch.float64);q/=q.sum()
        for width in (1,2,4,16):
            ids,rest=candidates(q,width,rng);fixed=ids[:-1]
            diagonal=torch.minimum(p,rest);raw=(p-rest).clamp_min(0)
            joint=torch.diag(diagonal)
            if float(raw.sum())>0:
                joint+=raw[:,None]/raw.sum()*(rest-diagonal)[None,:]
            reconstructed=joint.sum(-1)
            error=float((reconstructed-p).abs().max())
            proposal_error=float((joint.sum(0)-rest).abs().max())
            alpha=float(expected_acceptance(p,q,width))
            active=(rest>0).nonzero().flatten().tolist();v=len(p)
            costs=[]
            for i in range(v):
                costs.extend(-float(i in fixed or i==j) for j in active)
            ae=torch.zeros((v+len(active),v*len(active)),dtype=torch.float64)
            for i in range(v):ae[i,i*len(active):(i+1)*len(active)]=1
            for k in range(len(active)):ae[v+k,k::len(active)]=1
            solution=linprog(costs,A_eq=ae.numpy(),b_eq=torch.cat((p,rest[active])).numpy(),bounds=(0,None),method='highs')
            if not solution.success:raise ArithmeticError(solution.message)
            optimum=-float(solution.fun);difference=abs(alpha-optimum)
            if error>1e-12 or proposal_error>1e-12 or difference>1e-10:raise ArithmeticError('Greedy distribution or optimum differs')
            rows.append(dict(example=example,width=width,actual_width=len(ids),target_error=error,
                             proposal_error=proposal_error,set_acceptance=alpha,transport_optimum=optimum,optimum_error=difference))
    # Exercise the actual random selection, including residual hits in fixed candidates.
    p=torch.tensor([.1,.2,.7]);q=torch.tensor([.6,.3,.1]);counts=torch.zeros(3);hits=0
    decide=lambda alpha,rng,device:bool(torch.rand((),generator=rng,device=device)<alpha)
    draw=lambda probabilities,rng:int(torch.multinomial(probabilities,1,generator=rng))
    for _ in range(20000):
        ids,rest=candidates(q,2,rng);token,branch,_=select(p,ids,rest,rng,decide,draw)
        counts[token]+=1;hits+=branch is not None
    frequencies=counts/20000
    if float((frequencies-p).abs().max())>.02:raise ArithmeticError('Actual selection frequencies differ')
    expected_hits=float(expected_acceptance(p,q,2))
    if abs(hits/20000-expected_hits)>.02:raise ArithmeticError('Actual set membership differs')
    out=Path(sys.argv[1]);out.write_text(json.dumps(dict(status='complete',rows=rows,
        maximum_target_error=max(r['target_error'] for r in rows),
        maximum_optimum_error=max(r['optimum_error'] for r in rows),
        actual_sampler=dict(draws=20000,target=p.tolist(),frequencies=frequencies.tolist(),set_hits=hits,
                            expected_set_acceptance=expected_hits)),indent=2)+'\n')
    print('Greedy coupling numeric validation complete',len(rows),flush=True)


if __name__=='__main__':main()
