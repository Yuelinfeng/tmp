"""Retain the actual positive observation law without extra decoder draws."""
from contextlib import contextmanager
from diffusion_memory import MemoryRecorder


class WindowMemoryRecorder(MemoryRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.minimum_probability = 1.

    @contextmanager
    def context(self):
        with super().context():
            original = self.model.propose

            def propose(*args, **kwargs):
                import torch
                q = original(*args, **kwargs)
                if q.numel() != 4096 or not bool(torch.isfinite(q).all()) or not bool((q > 0).all()):
                    raise ArithmeticError('Observation proposal must cover all Feature classes')
                self.minimum_probability = min(self.minimum_probability, float(q.min()))
                return q

            self.model.propose = propose
            try:
                yield self
            finally:
                self.model.propose = original
