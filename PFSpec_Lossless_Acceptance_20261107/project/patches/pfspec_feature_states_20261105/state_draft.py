"""Position-conditioned drafting from states already computed by the Target."""
from contextlib import contextmanager

import torch
from torch import nn
import torch.nn.functional as F

from train_feature_state_distill import FeatureStateDraft
from dynamic_target_graph import DynamicTargetGraph
from bench_gaussiangpt_pf_block2_cow import CaptureLastBlock


class LayerTap:
    def __init__(self, target):
        n = len(target.transformer.h)
        self.layers = (max(0, n // 4 - 1), max(0, n // 2 - 1), n - 1)
        self.values = {}
        self.handles = []

    @contextmanager
    def context(self, target):
        if self.handles:
            raise RuntimeError('Layer readers already active')
        for index in self.layers:
            def save(module, inputs, output, index=index):
                self.values[index] = output.detach()
            self.handles.append(target.transformer.h[index].register_forward_hook(save))
        try:
            yield self
        finally:
            for handle in self.handles:
                handle.remove()
            self.handles.clear()
            self.values.clear()

    def packed(self):
        return torch.cat([self.values[index] for index in self.layers], dim=-1)


class StateTargetGraph(DynamicTargetGraph):
    def __init__(self, target, tap, **kwargs):
        super().__init__(target, **kwargs)
        self.tap = tap

    def _trunk(self, *args, **kwargs):
        logits, high, normalized = super()._trunk(*args, **kwargs)
        # The extra output is inside the captured computation, so every replay
        # publishes states belonging to this call, including tree branches.
        return logits, self.tap.packed(), normalized


@contextmanager
def state_capture(tap):
    import bench_gaussiangpt_pf_block2_cow as module
    original = module.CaptureLastBlock

    class PackedCapture(original):
        def run(self, *args, **kwargs):
            logits, states = super().run(*args, **kwargs)
            if states.shape[-1] == 1024:
                states = tap.packed()[0]
            if states.shape[-1] != 3072:
                raise RuntimeError('Unexpected packed state dimension')
            return logits, states

    module.CaptureLastBlock = PackedCapture
    try:
        yield
    finally:
        module.CaptureLastBlock = original


class ConditionalStateDraft(FeatureStateDraft):
    def __init__(self, weight, kind, token_weight=None):
        super().__init__(weight)
        self.kind = kind
        if kind in ('fusion', 'semantic', 'capacity'):
            self.layer_mix = nn.Linear(2048, 256, bias=False)
            nn.init.zeros_(self.layer_mix.weight)
        if kind == 'semantic':
            if token_weight is None:
                raise ValueError('Target token embeddings are required')
            self.register_buffer('target_tokens', token_weight.detach(), persistent=False)
            self.token_mix = nn.Linear(2048, 256, bias=False)
            nn.init.zeros_(self.token_mix.weight)

    def forward(self, states, previous_feature, position):
        high = states[..., -1024:].float()
        state_input = self.project(self.in_norm(high))
        if self.kind in ('fusion', 'semantic'):
            lower = states[..., :2048].reshape(*states.shape[:-1], 2, 1024)
            lower = F.rms_norm(lower, (1024,)).flatten(-2)
            state_input = state_input + self.layer_mix(lower)
        if self.kind == 'capacity':
            # Same learned parameter count as fusion; only the top state is read.
            high_norm = F.rms_norm(high, (1024,))
            high_pair = torch.cat((high_norm, F.gelu(high_norm)), dim=-1)
            state_input = state_input + self.layer_mix(high_pair)
        if self.kind == 'semantic':
            ids = torch.stack((previous_feature.long()+8000, position.long()), -1)
            tokens = F.rms_norm(F.embedding(ids,self.target_tokens), (1024,)).flatten(-2)
            state_input = state_input + self.token_mix(tokens)
        inputs = torch.cat((state_input, self.feature_embed(previous_feature.long()),
                            self.position_embed(position.long())), -1)
        delta_input = self.delta[:-1](inputs)
        normalized = F.rms_norm(high + self.delta[-1](delta_input).float(), (1024,))
        raw = F.linear(normalized, self.frozen_head)
        return normalized, 15.0 * torch.tanh(raw.float() / 15.0)


def make_draft(weight, kind, checkpoint, token_weight=None):
    model = ConditionalStateDraft(weight, kind, token_weight)
    value = torch.load(checkpoint, map_location='cpu', weights_only=False)
    missing, unexpected = model.load_state_dict(value['model_state'], strict=False)
    allowed = {'layer_mix.weight', 'token_mix.weight'}
    if unexpected or set(missing) - allowed:
        raise RuntimeError(f'Incompatible model: {missing}, {unexpected}')
    return model
