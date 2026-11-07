"""A masked Feature block with Target context in every attention layer."""
import math
import torch
from torch import nn
import torch.nn.functional as F


def rotary(x, positions):
    half = x.shape[-1]//2
    frequencies = torch.exp(torch.arange(half, device=x.device, dtype=torch.float32)
                            * (-math.log(10000.)/half))
    angles = positions[:,None,:,None].float()*frequencies
    a,b = x[...,:half],x[...,half:]
    return torch.cat((a*angles.cos()-b*angles.sin(),b*angles.cos()+a*angles.sin()),-1)


class ContextLayer(nn.Module):
    def __init__(self, dim=1024, heads=16, intermediate=2816):
        super().__init__()
        self.dim,self.heads = dim,heads
        self.query = nn.Linear(dim,dim,bias=False)
        self.key = nn.Linear(dim,dim,bias=False)
        self.value = nn.Linear(dim,dim,bias=False)
        self.output = nn.Linear(dim,dim,bias=False)
        self.up = nn.Linear(dim,intermediate,bias=False)
        self.gain = nn.Linear(dim,intermediate,bias=False)
        self.down = nn.Linear(intermediate,dim,bias=False)

    def forward(self, block, memory, memory_positions, block_positions, valid):
        batch,length,_ = block.shape
        normalized = F.rms_norm(block,(self.dim,))
        context = torch.cat((memory,normalized),1)
        def heads(x):
            return x.reshape(batch,-1,self.heads,self.dim//self.heads).transpose(1,2)
        q = F.rms_norm(heads(self.query(normalized)),(self.dim//self.heads,))
        k = F.rms_norm(heads(self.key(context)),(self.dim//self.heads,))
        v = heads(self.value(context))
        q = rotary(q,block_positions)
        k = rotary(k,torch.cat((memory_positions,block_positions),1))
        visible = torch.cat((valid,torch.ones(batch,length,device=valid.device,dtype=torch.bool)),1)
        attended = F.scaled_dot_product_attention(q,k,v,attn_mask=visible[:,None,None,:])
        block = block+self.output(attended.transpose(1,2).reshape(batch,length,self.dim))
        normalized = F.rms_norm(block,(self.dim,))
        return block+self.down(F.silu(self.gain(normalized))*self.up(normalized))


class DiffusionDraft(nn.Module):
    def __init__(self, fusion, token_weight, *, layers=5, memory_tokens=64):
        super().__init__()
        self.fusion = fusion
        self.layers = nn.ModuleList([ContextLayer() for _ in range(layers)])
        self.context_mix = nn.Linear(3072,1024,bias=False)
        self.result = nn.Linear(1024,1024,bias=False)
        nn.init.zeros_(self.result.weight)
        self.register_buffer('target_tokens',token_weight.detach(),persistent=False)
        self.mask_feature = nn.Parameter(token_weight[8000:12096].detach().mean(0).clone())
        self.memory_tokens = memory_tokens

    def forward(self, memory, memory_positions, valid, ends, previous_feature, p1, p2, state):
        if memory.shape[-1]!=3072 or memory.shape[1]!=self.memory_tokens:
            raise ValueError('Unexpected Target context shape')
        memory = F.rms_norm(memory.reshape(*memory.shape[:-1],3,1024),(1024,)).flatten(-2)
        memory = F.rms_norm(self.context_mix(memory),(1024,))
        f0 = F.embedding(previous_feature.long()+8000,self.target_tokens)
        position1 = F.embedding(p1.long(),self.target_tokens)
        position2 = F.embedding(p2.long(),self.target_tokens)
        mask = self.mask_feature.expand_as(f0)
        block = torch.stack((f0,position1,mask,position2,mask),1)
        positions = ends[:,None]+torch.arange(5,device=ends.device)[None,:]
        for layer in self.layers:
            block = layer(block,memory,memory_positions,positions,valid)
        predicted = self.result(F.rms_norm(block[:,[2,4]],(1024,)))
        first,_ = self.fusion(state,previous_feature,p1)
        second,_ = self.fusion(state,previous_feature,p2)
        hidden = F.rms_norm(torch.stack((first,second),1)+predicted,(1024,))
        logits = F.linear(hidden,self.fusion.frozen_head)
        return 15.*torch.tanh(logits.float()/15.)


def make_model(initial, token_weight, feature_weight, *, layers=5, memory_tokens=64):
    from state_draft import make_draft
    fusion = make_draft(feature_weight,'fusion',initial)
    return DiffusionDraft(fusion,token_weight,layers=layers,memory_tokens=memory_tokens)
