"""Observe committed Target states and masked-block teacher rows."""
from contextlib import contextmanager
from verification_feedback import FeedbackRecorder


class MemoryRecorder(FeedbackRecorder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory = []
        self.last_tree = None

    def store(self, start, states):
        states = states.detach().float().cpu().clone()
        if states.ndim == 3:
            states = states[0]
        if start > len(self.memory):
            raise ArithmeticError('Committed Target history has missing tokens')
        self.memory[start:] = list(states.unbind(0))

    @contextmanager
    def context(self):
        import bench_gaussiangpt_pf_block2_cow as module
        original_capture = module.CaptureLastBlock
        original_commit = self.tree.commit
        owner = self

        class Capture(original_capture):
            def run(self, target, cache, tokens, *args, **kwargs):
                start = int(cache.get_pos())
                result = super().run(target, cache, tokens, *args, **kwargs)
                owner.store(start, result[1])
                return result

        def commit(cache, path):
            before = int(cache.get_pos())
            result = original_commit(cache, path)
            owner.store(before, owner.last_tree[path])
            return result

        module.CaptureLastBlock = Capture
        self.tree.commit = commit
        try:
            with super().context():
                original_propose = self.model.propose
                original_verify = self.tree.verify

                def propose(prefix, position, state, generator):
                    result = original_propose(prefix, position, state, generator)
                    end = len(prefix)-1
                    if end > len(self.memory):
                        raise ArithmeticError('Draft anchor exceeds committed Target history')
                    import torch
                    if not torch.equal(self.memory[end-1], state.detach().float().cpu()):
                        raise ArithmeticError('Draft state differs from its committed Target token')
                    if self.current['slot'] is not None:
                        self.current['saved']['memory_end'] = end
                    return result

                def verify(cache, pending, position, features, next_positions):
                    result = original_verify(cache, pending, position, features, next_positions)
                    self.last_tree = result[1].detach().clone()
                    if self.current['slot'] is not None:
                        logits = result[0]
                        if logits.ndim == 3:
                            logits = logits[0]
                        self.current['saved'].update(
                            second_logits=logits[3,8000:12096].detach().float().cpu().clone(),
                            second_position=int(next_positions[0]),
                            second_position_probability=float(self.shared.policy(logits[2])[next_positions[0]]))
                    return result

                self.model.propose = propose
                self.tree.verify = verify
                try:
                    yield self
                finally:
                    self.model.propose = original_propose
                    self.tree.verify = original_verify
        finally:
            module.CaptureLastBlock = original_capture
            self.tree.commit = original_commit
