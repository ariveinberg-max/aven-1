import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import torch
from brain import Brain, Config
from tokenizer import Tokenizer
from evaluate_capabilities import score, overlap, sha


class EvaluationTests(unittest.TestCase):
    def test_strict_scoring(self):
        self.assertTrue(score(' 43 ', ['43']))
        self.assertFalse(score('The answer is 43', ['43']))
        self.assertFalse(score('quiet', ['QUIET']))
        self.assertFalse(score('43 or 44', ['43']))

    def test_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'corpus.txt';p.write_text('ONE two three four five six seven eight nine')
            self.assertEqual(len(overlap([{'id':'a','prompt':'one two three four five six seven eight'}],[p])),1)

    def test_streaming_overlap_matches_whole_file_protocol(self):
        cases = [dict(id='a', prompt='one two three four five six seven eight nine'),
                 dict(id='b', prompt='STRASSE two three four five six seven eight'),
                 dict(id='c', prompt='a question too short')]
        text = 'two three four five six seven eight nine ' + ('unrelated ' * 30) + 'ONE\tTWO\nthree  four five six seven eight STRAßE two three four five six seven eight'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'corpus.txt'
            path.write_text(text)
            normalized = ' '.join(text.casefold().split())
            expected = []
            for case in cases:
                words = case['prompt'].casefold().split()
                for index in range(max(0, len(words) - 7)):
                    span = ' '.join(words[index:index+8])
                    if span in normalized:
                        expected.append(dict(case_id=case['id'], corpus=str(path), matching_span=span))
                        break
            for chunk_size in [1, 2, 7, 31, 1024]:
                with self.subTest(chunk_size=chunk_size):
                    self.assertEqual(overlap(cases, [path], chunk_size=chunk_size), expected)

    def test_runner_and_identity_guard(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); cp=root/'latest.pt'; tk=root/'tokenizer.json'; out=root/'report.json'
            model=Brain(Config(width=16,layers=1,heads=2,context=256,vocab=256))
            torch.save({'config':vars(model.config),'model':model.state_dict(),'step':0},cp)
            Tokenizer().save(tk)
            before=(sha(cp),sha(tk))
            cmd=[sys.executable,'evaluate_capabilities.py','--checkpoint',str(cp),'--tokenizer',str(tk),'--expect-sha256',before[0],'--expect-tokenizer-sha256',before[1],'--output',str(out),'--max-new-tokens','2']
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads(out.read_text())
            self.assertEqual(len(report['results']),30)
            self.assertEqual(len(report['categories']),5)
            self.assertTrue(all(r['status']=='evaluated' for r in report['results']))
            self.assertEqual(before,(sha(cp),sha(tk)))
            out.unlink()
            cmd[cmd.index('--expect-sha256')+1]='0'*64
            result=subprocess.run(cmd,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            self.assertIn('Checkpoint SHA-256 mismatch',result.stderr)
            self.assertFalse(out.exists())
            print('PASS: 30 cases / 5 categories; temporary model only; checkpoint/tokenizer unchanged; bad hash rejected without report')

if __name__=='__main__': unittest.main()
