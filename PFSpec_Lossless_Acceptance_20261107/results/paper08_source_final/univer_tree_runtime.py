"""Use independent remaining Feature draws for every expanded parent."""
from long_tree_runtime import LongTargetTreeRuntime


class UniVerTargetTreeRuntime(LongTargetTreeRuntime):
    def verify(self,cache,pending_f,p1,features,p2s,*,second_features,third_position):
        rows=[tuple(int(value) for value in row) for row in second_features]
        if len(rows)!=len(features) or any(len(row)!=4 for row in rows):
            raise ValueError('Four second Feature candidates required for each first branch')
        self._branch_second_features=rows
        return super().verify(cache,pending_f,p1,features,p2s,
            second_features=rows[0],third_position=third_position)

    def _node_tokens(self,pending_f,p1,features,p2s):
        tokens,parents,depths,kinds,paths=super()._node_tokens(pending_f,p1,features,p2s)
        target=self.target
        first=0 if bool(target.sparse_shared_vocab) else int(target.sparse_feature_token_offset)
        end=first+int(target.sparse_feature_vocab_size)
        for branch,row in enumerate(self._branch_second_features):
            if len(set(row))!=4 or any(not first<=value<end for value in row):
                raise ValueError('Second Feature candidates must be distinct legal token ids')
            for second,token in enumerate(row):tokens[4+10*branch+2*second]=token
        return tokens,parents,depths,kinds,paths
