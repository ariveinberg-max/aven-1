import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from compare_capabilities import compare, MATCH_FIELDS

class ComparisonTests(unittest.TestCase):
    def fixture(self):
        r = {key: 'same' for key in MATCH_FIELDS}
        r.update(checkpoint_sha256='a'*64, suite_sha256='b'*64, tokenizer_sha256='c'*64,
                 evaluator_sha256='d'*64, code_sha256={'brain.py':'e'*64}, max_new_tokens=32, results=[dict(id='a', status='evaluated', response='3',
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

    def test_malformed_answers_cannot_use_substring_scoring(self):
        for answers in ('34', [], [None], ['']):
            a=self.fixture(); a['results'][0]['answers']=answers
            with self.subTest(answers=answers), self.assertRaises(ValueError): compare(a,a)

    def test_missing_or_invalid_metadata(self):
        for update in ({'suite_sha256':None}, {'code_sha256':{}}, {'max_new_tokens':True},
                       {'checkpoint_sha256':'not-a-hash'}, {'results':None}):
            a=self.fixture();a.update(update)
            with self.subTest(update=update), self.assertRaises(ValueError): compare(a,a)

    def test_regression_not_hidden_by_equal_aggregate_score(self):
        a=self.fixture()
        second=copy.deepcopy(a['results'][0]);second.update(id='b',response='4',correct=True)
        a['results'].append(second)
        b=copy.deepcopy(a)
        b['results'][0].update(response='4',correct=True)
        b['results'][1].update(response='3',correct=False)
        result=compare(a,b)
        self.assertEqual(result['regression_count'],1)
        self.assertEqual(result['improvement_count'],1)
        self.assertEqual(result['categories']['math']['accuracy_change_percentage_points'],0)

    def test_cli_regression_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); a=self.fixture();b=copy.deepcopy(a)
            a['results'][0].update(response='4',correct=True)
            for name, report in [('before.json',a),('after.json',b)]:
                (root/name).write_text(json.dumps(report))
            command=[sys.executable,str(Path(__file__).with_name('compare_capabilities.py')),
                     str(root/'before.json'),str(root/'after.json')]
            regular=subprocess.run(command,capture_output=True,text=True)
            guarded=subprocess.run(command+['--fail-on-regression'],capture_output=True,text=True)
            self.assertEqual(regular.returncode,0,regular.stderr)
            self.assertEqual(guarded.returncode,1,guarded.stderr)
            self.assertEqual(json.loads(guarded.stdout)['regression_count'],1)

if __name__=='__main__': unittest.main()
