"""Exercise every accepted-prefix length against the actual causal Position state."""
import copy,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/root/workspace/GaussianGPT')
from gpt_position_spec_decode_smoke import CausalRanker


def main():
    records=[]
    for long in [False,True]:
        for accepted in range(6 if long else 4):
            original=CausalRanker(np.arange(8000),8000)
            original.propose(10)
            shadow=copy.deepcopy(original);shadow.finalize(11);shadow.propose(11)
            shadow2=copy.deepcopy(shadow);shadow2.finalize(13);shadow2.propose(13)
            online=original
            if accepted>=4:
                online=shadow2
                if accepted>=5:online.finalize(16)
            elif accepted>=2:
                online=shadow
                if accepted>=3:online.finalize(13)
            elif accepted>=1:online=shadow
            positions=[11,13,16][:(accepted+1)//2]
            if accepted%2==0:
                online.finalize(21);positions.append(21)
            online.drop_eos()
            expected=CausalRanker(np.arange(8000),8000)
            previous=10
            for position in positions:
                expected.propose(previous);expected.finalize(position);previous=position
            if online.n_finalized_positions!=expected.n_finalized_positions:
                raise ArithmeticError('Finalized Position count differs')
            if list(online.predictor.gaps)!=list(expected.predictor.gaps):
                raise ArithmeticError('Causal gap history differs')
            if online.last_seen!=expected.last_seen:
                raise ArithmeticError('Position observation history differs')
            online.propose(previous)
            online.drop_eos()
            records.append(dict(two_feature_levels=long,accepted_prefix_length=accepted,
                                submitted_positions=positions,next_proposal_ready=True))
    output=Path('/root/autodl-tmp/pfspec_lossless_acceptance_20261106/paper08_univer/position_state_validation.json')
    output.write_text(json.dumps(dict(status='complete',cases=records),indent=2)+'\n')
    print('Actual causal Position cases passed',len(records),flush=True)


if __name__=='__main__':main()
