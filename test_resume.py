"""Resume regressions using real tiny-model steps, isolated from project artifacts."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock

import torch
import train


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / 'training.txt'
        self.data.write_text('A small real training corpus for resume testing.\n' * 120)
        self.root_patch = patch.object(train, 'ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        for sig in (signal.SIGINT, signal.SIGTERM):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
        self.invoke('--width', '16', '--layers', '1', '--heads', '2', '--context', '8', '--vocab', '256')
        self.checkpoint = self.root / 'checkpoints/latest.pt'
        self.saved = torch.load(self.checkpoint, weights_only=True)
        # Independent parameter count from a real model (including tied embeddings).
        self.params = sum(p.numel() for p in train.Brain(train.Config(**self.saved['config'])).parameters())

    def invoke(self, *args):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['train.py', '--data', str(self.data), '--steps', '1', '--batch-size', '2', '--device', 'cpu', *args]), contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            train.main()
        return output.getvalue()

    def snapshot(self):
        return {str(p.relative_to(self.root)): (hashlib.sha256(p.read_bytes()).hexdigest(), p.stat().st_mtime_ns)
                for p in self.root.rglob('*') if p.is_file()}

    def test_wrong_expect_params_has_no_side_effects(self):
        for mode in ('--resume', '--finetune'):
            before = self.snapshot()
            fake_wandb = types.SimpleNamespace(init=Mock())
            with patch.dict(sys.modules, {'wandb': fake_wandb}), patch.object(train, 'Brain') as model, patch.object(torch.optim, 'AdamW') as optimizer:
                with self.assertRaises(SystemExit) as error:
                    self.invoke(mode, '--expect-params', str(self.params + 1), '--wandb')
                self.assertEqual(error.exception.code, 2)
                model.assert_not_called()
                optimizer.assert_not_called()
                fake_wandb.init.assert_not_called()
            self.assertEqual(before, self.snapshot())
        print('PASS wrong --expect-params: exit 2; model/optimizer/W&B calls=0; all file hashes and mtimes unchanged')

    def test_explicit_architecture_mismatches(self):
        for mode in ('--resume', '--finetune'):
            for field in ('width', 'layers', 'heads', 'context', 'vocab'):
                before = self.snapshot()
                with self.assertRaises(SystemExit) as error:
                    self.invoke(mode, '--' + field, str(self.saved['config'][field] + 1))
                self.assertEqual(error.exception.code, 2)
                self.assertEqual(before, self.snapshot())
        print('PASS all five explicit architecture mismatches rejected for resume and finetune')

    def test_normal_resume_without_expect_params(self):
        output = self.invoke('--resume')
        after = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(after['step'], self.saved['step'] + 1)
        self.assertEqual(after['config'], self.saved['config'])
        self.assertTrue(any(not torch.equal(after['model'][k], v) for k, v in self.saved['model'].items()))
        self.assertTrue(after['optimizer']['state'])
        print('PASS normal resume without --expect-params: ' + next(x for x in output.splitlines() if x.startswith('Step ')))

    def test_matching_explicit_expectations(self):
        args = ['--resume', '--expect-params', str(self.params)]
        for field in ('width', 'layers', 'heads', 'context', 'vocab'):
            args += ['--' + field, str(self.saved['config'][field])]
        self.invoke(*args)
        self.assertEqual(torch.load(self.checkpoint, weights_only=True)['step'], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
