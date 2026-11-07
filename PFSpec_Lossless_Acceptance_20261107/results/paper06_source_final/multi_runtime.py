"""Reuse the existing tree computation for a larger number of Feature branches."""
from tree_runtime import TargetTreeRuntime


class MultiTargetTreeRuntime(TargetTreeRuntime):
    maximum_width=16

    def __init__(self,target,linear_runtime,*,graph=True,max_width=1):
        width=int(self.maximum_width)
        if not 1<=width<=4096:
            raise ValueError('Candidate width must fit the Feature vocabulary')
        super().__init__(target,linear_runtime,graph=graph,max_width=min(4,width))
        self.max_width=width
