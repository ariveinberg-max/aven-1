import copy
import unittest
from compare_capabilities import compare, MATCH_FIELDS

class ComparisonTests(unittest.TestCase):
    def fixture(self):
        r = {key: 'same' for key in MATCH_FIELDS}
        r.update(checkpoint_sha256='a', results=[dict(id='a', status='evaluated', response='3',
                 correct=False, answers=['4'], prompt='Two plus two?', category='math', prompt_tokens=4)])
        return r

    def test_improvement_and_unknown_overlap(self):
        a=self.fixture(); b=copy.deepcopy(a)
        b['results'][0].update(response='4',correct=True)
        r=compare(a,b)
        self.assertEqual(r['categories']['math']['accuracy_change_percentage_points'],100)
        self.assertEqual(r['overlap_evidence']['before'],'NOT CHECKED')

    def test_mismatch_rejected(self):
        for field in MATCH_FIELDS:
            a=self.fixture(); b=copy.deepcopy(a); b[field]='different'
            with self.subTest(field=field), self.assertRaises(ValueError): compare(a,b)

    def test_bad_scores_and_skipped_cases_rejected(self):
        for update in ({'correct':True},{'status':'prompt_too_long'}):
            a=self.fixture(); b=copy.deepcopy(a); b['results'][0].update(update)
            with self.assertRaises(ValueError): compare(a,b)

    def test_case_changes_rejected(self):
        a=self.fixture(); b=copy.deepcopy(a); b['results'][0]['prompt']='Different?'
        with self.assertRaises(ValueError): compare(a,b)

if __name__=='__main__': unittest.main()
