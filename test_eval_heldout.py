import unittest
import tempfile
from pathlib import Path
import json
import subprocess
import sys
from tokenizer import Tokenizer
from brain import Brain, Config
from artifact_io import file_sha256
from types import SimpleNamespace
import torch
from eval_heldout import score, score_file


class RecordingModel:
    config = SimpleNamespace(context=4)
    def __init__(self):
        self.targets = []
    def eval(self):
        return self
    def __call__(self, x, y):
        assert torch.equal(x + 1, y)
        self.targets.extend(y.flatten().tolist())
        return None, y.float().mean(), None


class HeldoutTests(unittest.TestCase):
    def test_every_target_once_and_weighted(self):
        for length in [2, 4, 5, 6, 9, 10, 18]:
            for batch in [1, 3]:
                with self.subTest(length=length, batch=batch):
                    model = RecordingModel()
                    result = score(model, torch.arange(length), 4, batch, 'cpu')
                    self.assertEqual(model.targets, list(range(1, length)))
                    self.assertAlmostEqual(result, length / 2)

    def test_empty_and_invalid(self):
        self.assertIsNone(score(RecordingModel(), torch.arange(1), 4, 1, 'cpu'))
        for context, batch in [(0, 1), (5, 1), (4, 0)]:
            with self.assertRaises(ValueError):
                score(RecordingModel(), torch.arange(10), context, batch, 'cpu')


    def test_streamed_file_scores_every_target_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            path.write_bytes(bytes(range(33, 100)))
            for chunk_bytes in [1, 7, 65, 1024]:
                model = RecordingModel()
                result = score_file(model, path, Tokenizer(), 4, 3, 'cpu', chunk_bytes)
                self.assertEqual(model.targets, list(range(34, 100)))
                self.assertEqual(result['targets'], 66)
                self.assertEqual(result['tokens'], 67)
                self.assertEqual(result['sha256'], file_sha256(path))

    def test_streamed_bpe_matches_same_chunk_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'text.txt'
            raw = b'some text with repeated words and different pieces. ' * 5
            path.write_bytes(raw)
            tokenizer = Tokenizer()
            tokenizer.train(raw, 275)
            model = Brain(Config(width=16, layers=1, heads=2, context=16, vocab=tokenizer.vocab_size))
            ids = []
            for offset in range(0, len(raw), 17):
                ids.extend(tokenizer.encode_ids(raw[offset:offset+17]))
            reference = score(model, torch.tensor(ids), 16, 3, 'cpu')
            result = score_file(model, path, tokenizer, 16, 3, 'cpu', 17)
            self.assertAlmostEqual(result['loss'], reference, places=6)
            self.assertEqual(result['targets'], len(ids) - 1)

    def test_cli_report_records_protocol_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root/'data'
            data.mkdir()
            (data/'probe.txt').write_text('A small independent text sample. ' * 3)
            model = Brain(Config(width=16, layers=1, heads=2, context=16))
            checkpoint = root/'latest.pt'
            torch.save(dict(config=vars(model.config), model=model.state_dict(), step=1), checkpoint)
            Tokenizer().save(root/'tokenizer.json')
            before = file_sha256(checkpoint)
            output = root/'report.json'
            command = [sys.executable, 'eval_heldout.py', str(checkpoint), '--data-dir', str(data),
                       '--output', str(output), '--chunk-bytes', '17', '--batch-size', '2', '--device', 'cpu']
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertEqual(report['protocol'], 'external-text-v2')
            self.assertIn('NOT ESTABLISHED', report['training_exposure'])
            self.assertEqual(report['checkpoint_sha256'], before)
            self.assertEqual(file_sha256(checkpoint), before)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
