import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(shutil.which('bash'), 'Bash not installed')
class ResilientTests(unittest.TestCase):
    def run_wrapper(self, outcomes, continuous='0', retries='2', extra_args=(), save=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copyfile('run_resilient.sh', root / 'run.sh')
            fake = root / 'fake-python'
            fake.write_text('#!' + sys.executable + '\n' + '''import json,os,sys
from pathlib import Path
p=Path('calls.json')
calls=json.loads(p.read_text()) if p.exists() else []
calls.append(sys.argv[1:]);p.write_text(json.dumps(calls))
if os.environ['FAKE_SAVE']=='1':
    Path('checkpoints').mkdir(exist_ok=True);Path('checkpoints/latest.pt').write_text('saved')
outcomes=json.loads(os.environ['FAKE_OUTCOMES'])
raise SystemExit(outcomes[min(len(calls)-1,len(outcomes)-1)])
''')
            fake.chmod(0o755)
            env = dict(os.environ, PYTHON=str(fake), MAX_RETRIES=retries, CONTINUOUS=continuous,
                       BACKOFF_SECONDS='0', LOG_FILE=str(root/'run.log'), FAKE_OUTCOMES=json.dumps(outcomes), FAKE_SAVE='1' if save else '0')
            result = subprocess.run(['bash', str(root/'run.sh'), '--steps', '1', *extra_args], env=env, capture_output=True, text=True, timeout=10)
            calls = json.loads((root/'calls.json').read_text()) if (root/'calls.json').exists() else []
            return result, calls

    def test_clean_chunks_do_not_consume_failure_budget(self):
        result, calls = self.run_wrapper([0, 1, 0, 1, 2], continuous='1')
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertEqual(len(calls), 5)
        self.assertTrue(all(call.count('--resume') <= 1 for call in calls))

    def test_consecutive_failures_are_bounded(self):
        result, calls = self.run_wrapper([1, 1, 1])
        self.assertEqual(result.returncode, 1)
        self.assertEqual(len(calls), 2)

    def test_interrupt_exit_never_retried(self):
        for code in [130, 143]:
            result, calls = self.run_wrapper([code], continuous='1')
            self.assertEqual(result.returncode, 130)
            self.assertEqual(len(calls), 1)

    def test_bad_configuration_never_retried(self):
        result, calls = self.run_wrapper([2])
        self.assertEqual(result.returncode, 2)
        self.assertEqual(len(calls), 1)
        result, calls = self.run_wrapper([0], retries='0')
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls, [])


    def test_init_from_removed_after_first_saved_checkpoint(self):
        result, calls = self.run_wrapper([1, 0], extra_args=('--init-from', 'parent', '--output-dir', 'checkpoints'))
        self.assertEqual(result.returncode, 0)
        self.assertIn('--init-from', calls[0])
        self.assertNotIn('--init-from', calls[1])
        self.assertNotIn('parent', calls[1])
        self.assertIn('--resume', calls[1])

    def test_preparation_failure_without_checkpoint_retries_original(self):
        result, calls = self.run_wrapper([1, 2], save=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(calls[0], calls[1])
        self.assertNotIn('--resume', calls[1])
