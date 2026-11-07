"""Keep first P/F metrics comparable and report the second Feature separately."""


def configure(module):
    original=module.normalize
    original_summary=module.summarize

    def normalize(row):
        stats=row['counts']
        if not stats.get('masked_feature_levels'):
            return original(row)
        first=dict(stats)
        first['feature_verified']-=stats['feature2_attempts']
        first['feature_accepted']-=stats['feature2_hits']
        result=original(dict(row,counts=first))
        result['total_draft_tokens']=stats['proposed']
        result['total_accepted_draft_tokens']=stats['accepted']
        if stats['feature2_random_accepts']+stats['feature2_residual_hits']!=stats['feature2_hits']:
            raise ArithmeticError('Second Feature outcomes differ')
        result.update(second_feature_eligible_rounds=stats['feature2_eligible_rounds'],
            second_feature_reached=stats['feature2_attempts'],second_feature_hits=stats['feature2_hits'],
            second_feature_conditional_acceptance=module.ratio(stats['feature2_hits'],stats['feature2_attempts']),
            second_feature_joint_acceptance=module.ratio(stats['feature2_hits'],stats['feature2_eligible_rounds']),
            second_feature_distinct_candidates=stats['feature2_distinct_candidates'],
            second_feature_branch_nodes=stats['feature2_branch_nodes'],
            total_feature_candidate_nodes=stats['candidate_tree_nodes']+stats['feature2_branch_nodes'],
            total_target_tokens_per_call=module.ratio(row['generated_tokens'],stats['target_calls']),
            first_feature_scope='Same first P/F counters as the single-Feature tree; second Feature outcomes are separate',
            maximum_round_output_tokens=6)
        return result

    module.normalize=normalize

    def summarize(rows):
        result=original_summary(rows)
        for cohort,methods in result['cohorts'].items():
            for method,item in methods.items():
                own=[row for row in rows if row['cohort']==cohort and row['method']==method
                     and 'second_feature_reached' in row]
                if not own:
                    continue
                item['second_feature_counts']={key:sum(row[key] for row in own) for key in
                    ['second_feature_eligible_rounds','second_feature_reached','second_feature_hits',
                     'second_feature_distinct_candidates','second_feature_branch_nodes','total_feature_candidate_nodes']}
                for key in ['second_feature_conditional_acceptance','second_feature_joint_acceptance','total_target_tokens_per_call']:
                    item['source_mean_metrics'][key]=module.source_average(own,key)
        return result

    module.summarize=summarize
