"""Reconstruct the complete conditional P/F output law of the two-token block."""
import json,sys
from pathlib import Path


def main():
    import torch
    from scipy.optimize import linprog
    from greedy_coupling import candidates
    from block_coupling import parameters,expected_joint
    rng=torch.Generator().manual_seed(20261106)
    rows=[]
    for case in range(24):
        pp=torch.rand(3,generator=rng,dtype=torch.float64);pp/=pp.sum()
        pf=torch.rand((3,4),generator=rng,dtype=torch.float64);pf/=pf.sum(-1,keepdim=True)
        q=torch.rand(4,generator=rng,dtype=torch.float64);q/=q.sum()
        if case==0:pp=torch.tensor([0.,.4,.6],dtype=torch.float64)
        if case==1:pp=torch.tensor([1.,0.,0.],dtype=torch.float64)
        if case==2:q=torch.tensor([0.,1.,0.,0.],dtype=torch.float64)
        if case==3:q=pf[0].clone();pp=torch.tensor([1.,0.,0.],dtype=torch.float64)
        for width in (1,2,4,16):
            ids,rest=candidates(q,width,rng);fixed=ids[:-1];c=float(pp[0])
            joint=torch.zeros_like(pf);hits=0.;prefix_acceptance=0.
            for r in (rest>0).nonzero().flatten().tolist():
                first,second,residual=parameters(c,pf[0],rest,r)
                probability=float(rest[r]);full=probability*second
                corrected=probability*(1-second)*first
                rejected=probability*(1-second)*(1-first)
                joint[0,r]+=full;hits+=full;prefix_acceptance+=full+corrected
                if corrected:
                    law=residual/residual.sum();joint[0]+=corrected*law
                    hits+=corrected*float(law[fixed].sum())
                if rejected:
                    residual_p=pp.clone();residual_p[0]=0;residual_p/=residual_p.sum()
                    joint+=rejected*residual_p[:,None]*pf
            truth=pp[:,None]*pf;error=float((joint-truth).abs().max())
            expected=float(expected_joint(c,pf[0],q,width))
            active=(rest>0).nonzero().flatten().tolist();v=truth.numel();m=len(active)
            costs=[-float(i//4==0 and (i%4 in fixed or i%4==r)) for i in range(v) for r in active]
            ae=torch.zeros((v+m,v*m),dtype=torch.float64)
            for i in range(v):ae[i,i*m:(i+1)*m]=1
            for k in range(m):ae[v+k,k::m]=1
            solution=linprog(costs,A_eq=ae.numpy(),b_eq=torch.cat((truth.flatten(),rest[active])).numpy(),bounds=(0,None),method='highs')
            if not solution.success:raise ArithmeticError(solution.message)
            optimum=-float(solution.fun)
            if error>1e-12 or abs(hits-expected)>1e-12 or abs(prefix_acceptance-c)>1e-12 or abs(expected-optimum)>1e-10:
                raise ArithmeticError('Complete Target law, prefix probability or candidate-set optimum differs')
            rows.append(dict(case=case,width=width,actual_width=len(ids),target_joint_error=error,
                position_error=abs(prefix_acceptance-c),set_probability=hits,formula_error=abs(hits-expected),
                transport_error=abs(expected-optimum)))
    out=Path(sys.argv[1]);out.write_text(json.dumps(dict(status='complete',rows=rows,
        maximum_target_joint_error=max(r['target_joint_error'] for r in rows),
        maximum_set_formula_error=max(r['formula_error'] for r in rows),
        maximum_transport_error=max(r['transport_error'] for r in rows),
        scope='All Position and Feature outputs, nonconstant Feature conditionals, zero support, deterministic Position and set reuse'),indent=2)+'\n')
    print('Block complete P/F law validated',len(rows),flush=True)


if __name__=='__main__':main()
