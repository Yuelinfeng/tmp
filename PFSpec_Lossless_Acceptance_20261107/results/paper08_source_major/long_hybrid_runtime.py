"""Use a two-level Target tree only for explicitly requested block verification."""
from multi_runtime import MultiTargetTreeRuntime
from long_tree_runtime import LongTargetTreeRuntime


class HybridTargetTreeRuntime(MultiTargetTreeRuntime):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.long_runtime=None
        self.long_pending=False

    def verify(self,cache,pending_f,p1,features,p2s,*,second_features=None,third_position=None):
        if second_features is None:
            if self.long_pending:
                raise RuntimeError('Commit the two-level tree before starting another tree')
            return super().verify(cache,pending_f,p1,features,p2s)
        if self._pending is not None or self.long_pending:
            raise RuntimeError('Commit the previous tree before starting another tree')
        if self.long_runtime is None:
            self.long_runtime=LongTargetTreeRuntime(self.target,self.linear_runtime,
                                                   graph=False,max_width=self.max_width)
        result=self.long_runtime.verify(cache,pending_f,p1,features,p2s,
                                       second_features=second_features,third_position=third_position)
        self.long_pending=True
        return result

    def commit(self,cache,path):
        if not self.long_pending:
            return super().commit(cache,path)
        result=self.long_runtime.commit(cache,path)
        self.long_pending=False
        return result

    def stats(self):
        value=super().stats()
        if self.long_runtime is None:
            return value
        long=self.long_runtime.stats()
        for key in ['calls','fallbacks','copy_bytes']:
            combined=value[key]
            for name,number in long[key].items():
                combined[name]=combined.get(name,0)+number
        value['graph_count']+=long['graph_count']
        value['graph_events']+=long['graph_events']
        value['runtime_init_ms']+=long['runtime_init_ms']
        value['pending_verify']=value['pending_verify'] or long['pending_verify']
        value['feature_levels']=2
        value['long_tree_graph_enabled']=False
        value['long_tree_execution_reason']='Captured 642-node execution changed a Top-p boundary in native comparison; use ordinary execution for this tree'
        return value
