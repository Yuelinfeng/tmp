"""Compose complete conditional allocations and select one submitted tree path."""
import numpy as np
from univer_coupling import allocation


def select(logits,first_position,second_position,third_position,first_ids,first_remaining,
           second_ids,second_remaining,rng,policy,decide,draw):
    import torch
    device=logits.device
    n=len(first_ids)
    long=third_position is not None
    stride=10 if long else 2
    probabilities={}
    allocations={}
    effective={0:1.}
    paths={0:[0]}
    children={}
    child_tokens={}

    def distribute(parent,ids,remaining,nodes,offset=0):
        p=probabilities[parent]
        a=allocation(effective[parent],p,remaining,ids[:-1],ids[-1])
        allocations[parent]=a
        children[parent]=nodes
        for token,node,c in zip(ids,nodes,a['conditional']):
            child_tokens[node]=int(token)+offset
            effective[node]=c
            paths[node]=paths[parent]+[node]

    def position(parent,token,node):
        p=policy(logits[parent]).detach().double().cpu().numpy()
        p/=p.sum()
        probabilities[parent]=p
        q=np.zeros_like(p);q[token]=1
        distribute(parent,[token],q,[node])

    def feature(parent,ids,q,nodes):
        probabilities[parent]=policy(logits[parent])[8000:12096].detach().double().cpu().numpy()
        probabilities[parent]/=probabilities[parent].sum()
        remaining=q.detach().double().cpu().numpy()
        remaining/=remaining.sum()
        distribute(parent,ids,remaining,nodes,8000)

    position(0,int(first_position),1)
    feature(1,first_ids,first_remaining,[2+stride*i for i in range(n)])
    for i in range(n):
        node=2+stride*i
        position(node,int(second_position),node+1)
        if long:
            feature(node+1,second_ids[i],second_remaining[i],[node+2+2*j for j in range(4)])
            for j in range(4):position(node+2+2*j,int(third_position),node+3+2*j)

    def postorder(node):
        for child in children.get(node,[]):yield from postorder(child)
        yield node

    for node in postorder(0):
        leaf=node not in allocations
        chance=effective[node] if leaf else allocations[node]['fallback']
        if not bool(decide(float(chance),rng,device)):continue
        if leaf:
            bonus=int(draw(policy(logits[node]),rng))
        else:
            p=allocations[node]['outside']
            mass=float(p.sum())
            if mass<=0:raise ArithmeticError('Selected fallback has no positive output mass')
            # The double allocation retains all positive probabilities.
            dist=torch.as_tensor(p/mass,device=device,dtype=torch.float64)
            offset=8000 if len(paths[node])%2==0 else 0
            bonus=int(draw(dist,rng))+offset
        path=paths[node]
        tokens=[child_tokens[x] for x in path[1:]]
        return dict(path=path,tokens=tokens,bonus=bonus,accepted_length=len(tokens),
                    allocation_nodes=len(allocations),selected_node=node)
    raise ArithmeticError('The root must resolve all remaining probability')
