"""Small count examples that distinguish accepted drafts from emitted tokens."""
import unittest

from acceptance_metrics import normalize, online_counts, paired_difference, source_average
from offline_acceptance import aggregate


class AcceptanceCountsTest(unittest.TestCase):
    def test_rejection_and_bonus_samples_do_not_inflate_acceptance(self):
        # Four full rounds: reject P; reject F; accept F/reject P2; accept P/F/P2.
        # Their generated lengths are 2, 2, 4 and 4. Accepted drafts: 0, 1, 2, 3.
        stats = dict(tree_width=1, tree_verify_calls=4, feature_verified=3,
                     feature_attempts=3, feature_accepted=2, feature_branch_hits=2,
                     feature_proposed=4, proposed=12, accepted=6,
                     target_tree_nodes=16, rounds=4, round_generated_lengths=[2, 2, 4, 4])
        result = online_counts(stats)
        self.assertEqual(result['position_acceptance'], 3/4)
        self.assertEqual(result['feature_conditional_acceptance'], 2/3)
        self.assertEqual(result['pair_joint_acceptance'], 1/2)
        self.assertEqual(result['round_emitted_tokens'], 12)
        self.assertEqual(result['total_accepted_draft_tokens'], 6)

    def test_no_eligible_round_has_undefined_rate(self):
        stats = dict(tree_width=1, tree_verify_calls=0, feature_verified=0,
                     feature_attempts=0, feature_accepted=0, feature_branch_hits=0,
                     feature_proposed=0, proposed=1, accepted=0,
                     target_tree_nodes=0, rounds=1, round_generated_lengths=[1])
        result = online_counts(stats)
        self.assertIsNone(result['feature_conditional_acceptance'])
        self.assertIsNone(result['pair_joint_acceptance'])

    def test_older_records_use_output_length_without_bos_bootstrap(self):
        stats = dict(tree_width=1, tree_verify_calls=4, feature_verified=3,
                     feature_attempts=3, feature_accepted=2, feature_branch_hits=2,
                     feature_proposed=4, proposed=12, accepted=6,
                     target_tree_nodes=16, rounds=4)
        row = dict(cohort='bos', pair_id='a', source='a', method='w1', seed=1,
                   prefix=0, generated_tokens=14, seconds=1., eos=True, counts=stats)
        self.assertEqual(normalize(row)['mean_emitted_per_round'], 3.)
        row.update(cohort='completion', prefix=256, generated_tokens=12)
        self.assertEqual(normalize(row)['mean_emitted_per_round'], 3.)

    def test_repeated_inputs_do_not_multiply_source_weight(self):
        rows = []
        for case, source, change in [('a1', 'A', .1), ('a2', 'A', .1), ('b1', 'B', -.1)]:
            for method, value in [('w1', .3), ('fusion', .3+change)]:
                rows.append(dict(pair_id=case, source=source, method=method, value=value))
        result = paired_difference(rows, 'fusion', 'w1', 'value', repeats=100)
        self.assertAlmostEqual(result['difference'], 0)
        self.assertEqual(result['sources'], 2)
        self.assertAlmostEqual(source_average([r for r in rows if r['method']=='fusion'], 'value')['mean'], .3)

    def test_joint_uses_statewise_products(self):
        rows = []
        for index, (position, feature) in enumerate([(1., .1), (.1, 1.)]):
            rows.append(dict(method='w1', source=str(index), cohort='bos', pair_id=str(index),
                             position_probability=position, feature_overlap=feature,
                             pair_joint_probability=position*feature,
                             raw_teacher_cross_entropy=0., target_top1_agreement=0))
        result = aggregate(rows)['bos']['w1']['state_means']
        self.assertAlmostEqual(result['pair_joint_probability'], .1)
        self.assertNotAlmostEqual(result['pair_joint_probability'],
                                  result['position_probability']*result['feature_overlap'])


if __name__ == '__main__':
    unittest.main()
