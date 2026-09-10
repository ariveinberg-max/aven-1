import copy
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from capability_errors import worksheet
import test_compare_capabilities as fixtures

class ErrorWorksheetTests(unittest.TestCase):
    def fixture(self): return fixtures.ComparisonTests().fixture()

    def test_case_and_empty_labels(self):
        a=self.fixture();a['results'][0].update(response='QUIET',answers=['quiet'])
        self.assertIn('Case-only mismatch',worksheet(a))
        a['results'][0]['response']=''
        self.assertIn('Empty response',worksheet(a))

    def test_skipped_is_not_presented_as_evaluated(self):
        a=self.fixture();a['results'][0].update(status='prompt_too_long',response=None,correct=False)
        out=worksheet(a)
        self.assertIn('Not evaluated: prompt exceeds context',out)
        self.assertIn('| 0 | 1 | 0 |',out)

    def test_escape_untrusted_response(self):
        a=self.fixture();a['results'][0]['response']='<script>alert(1)</script>|[click](https://example.com)\n# header'
        out=worksheet(a)
        self.assertNotIn('<script>',out)
        self.assertIn('&lt;script&gt;',out)
        self.assertIn('&#124;',out)

    def test_rejects_inconsistent_scores(self):
        a=self.fixture();a['results'][0]['correct']=True
        with self.assertRaises(ValueError):worksheet(a)

    def test_rejects_duplicate_ids(self):
        a=self.fixture();a['results'].append(copy.deepcopy(a['results'][0]))
        with self.assertRaises(ValueError):worksheet(a)

    def test_cli_preserves_input_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'report.json'
            output = root / 'worksheet.md'
            source.write_text(json.dumps(self.fixture()))
            original = source.read_bytes()
            command = [sys.executable, str(Path(__file__).with_name('capability_errors.py')),
                       str(source), '--output', str(output)]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            saved = output.read_bytes()
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(second.returncode, 2)
            self.assertEqual(output.read_bytes(), saved)
            self.assertEqual(source.read_bytes(), original)

    def test_invalid_report_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'report.json'
            output = root / 'worksheet.md'
            source.write_text('{}')
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('capability_errors.py')),
                 str(source), '--output', str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertFalse(output.exists())

if __name__=='__main__':unittest.main()
