"""Separate a verified Feature set from the single sampled random proposal."""


def configure(module):
    original=module.normalize
    original_summary=module.summarize

    def normalize(row):
        stats=row['counts']
        if not stats.get('greedy_candidate_set'):
            result=original(row)
            if result.get('feature_metrics_available'):
                result['feature_acceptance_scope']='Single original sampled Feature candidate'
            return result
        eligible=stats['tree_verify_calls']
        reached=stats['feature_verified']
        hits=stats['feature_branch_hits']
        if not 0<=hits<=reached<=eligible:
            raise ValueError('Feature set counts exceed reached full rounds')
        if stats['feature_attempts']!=reached or stats['feature_accepted']!=hits:
            raise ValueError('Feature set selection counts differ')
        if stats['random_proposal_accepts']+stats['residual_set_hits']!=hits:
            raise ValueError('Feature set outcomes differ')
        result={k:row[k] for k in ('cohort','pair_id','source','method','seed','prefix')}
        divide=module.ratio
        lengths=stats['round_generated_lengths']
        result.update(emitted_tokens=row['generated_tokens'],total_generation_seconds=row['seconds'],
            generated_tokens_per_second=divide(row['generated_tokens'],row['seconds']),
            eos=row['eos'],draft_proposal_seconds=None,feature_metrics_available=True,
            eligible_full_rounds=eligible,first_position_accepted=reached,
            first_feature_accepted=hits,first_pair_accepted=hits,
            feature_rejected=reached-hits,feature_not_reached=eligible-reached,
            total_draft_tokens=stats['proposed'],total_accepted_draft_tokens=stats['accepted'],
            target_verification_calls=eligible,target_verification_tokens=stats['target_tree_nodes'],
            total_target_calls=stats['target_calls'],total_target_input_tokens=stats['target_input_tokens'],
            rounds=stats['rounds'],round_emitted_tokens=sum(lengths),
            position_acceptance=divide(reached,eligible),feature_conditional_acceptance=divide(hits,reached),
            set_feature_conditional_acceptance=divide(hits,reached),pair_joint_acceptance=divide(hits,eligible),
            mean_emitted_per_round=divide(sum(lengths),stats['rounds']),
            feature_acceptance_scope='Output belongs to the preverified candidate set, including residual hits',
            random_proposal_accepts=stats['random_proposal_accepts'],residual_set_hits=stats['residual_set_hits'],
            random_proposal_acceptance=divide(stats['random_proposal_accepts'],reached),
            candidate_count=stats['candidate_tree_nodes'],candidate_width_counts=stats['candidate_width_counts'],
            requested_candidate_width=stats['tree_width'])
        return result

    def summarize(rows):
        result=original_summary(rows)
        result['feature_scope']='Per-method scope is recorded: single candidate or output membership in a verified set'
        for cohort,methods in result['cohorts'].items():
            for method,item in methods.items():
                own=[r for r in rows if r['cohort']==cohort and r['method']==method]
                item['feature_acceptance_scope']=own[0].get('feature_acceptance_scope')
                if 'candidate_count' in own[0]:
                    item['candidate_count']=sum(r['candidate_count'] for r in own)
                    item['random_proposal_accepts']=sum(r['random_proposal_accepts'] for r in own)
                    item['residual_set_hits']=sum(r['residual_set_hits'] for r in own)
        return result

    module.normalize=normalize
    module.summarize=summarize
