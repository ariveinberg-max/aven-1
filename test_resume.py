"""Resume regressions using real tiny-model steps, isolated from project artifacts."""
import contextlib
import hashlib
import io
import json
import os
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

    def test_distinct_execution_ids_and_legacy_lineage(self):
        # Simulate two independent continuations of the SAME old checkpoint.
        legacy = dict(self.saved)
        legacy.pop('lineage_id', None)
        torch.save(legacy, self.checkpoint)
        parent_bytes = self.checkpoint.read_bytes()
        parent_hash = hashlib.sha256(parent_bytes).hexdigest()
        ids = []
        for _ in range(2):
            self.checkpoint.write_bytes(parent_bytes)
            run = Mock(url='offline test', summary={})
            fake = types.SimpleNamespace(init=Mock(return_value=run), watch=Mock())
            with patch.dict(sys.modules, {'wandb': fake}):
                output = self.invoke('--resume', '--wandb')
            kwargs = fake.init.call_args.kwargs
            config = kwargs['config']
            ids.append(kwargs['id'])
            self.assertNotEqual(kwargs['id'], legacy['run_id'])
            self.assertEqual(kwargs['resume'], 'never')
            self.assertEqual(kwargs['group'], legacy['run_id'])
            self.assertIn(legacy['run_id'], kwargs['tags'])
            self.assertEqual(config['lineage_id'], legacy['run_id'])
            self.assertEqual(config['parent_checkpoint_sha256'], parent_hash)
            self.assertEqual(config['starting_step'], 1)
            self.assertEqual(config['platform'], 'cpu')
            after = torch.load(self.checkpoint, weights_only=True)
            self.assertEqual(after['lineage_id'], legacy['run_id'])
            self.assertEqual(after['run_id'], kwargs['id'])
            self.assertEqual(after['step'], 2)
        self.assertNotEqual(*ids)
        # A subsequent resume carries the lineage, rather than its parent's execution ID.
        self.invoke('--resume')
        after = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(after['lineage_id'], legacy['run_id'])
        self.assertNotIn(after['run_id'], ids)
        print('PASS sibling resumes: distinct execution IDs; shared legacy lineage/group/tag; exact parent SHA; starting_step=1; platform=cpu')

    def test_legacy_without_any_id_gets_stable_lineage(self):
        legacy = dict(self.saved)
        legacy.pop('lineage_id', None)
        legacy.pop('run_id', None)
        torch.save(legacy, self.checkpoint)
        source = self.checkpoint.read_bytes()
        lineages = []
        for _ in range(2):
            self.checkpoint.write_bytes(source)
            self.invoke('--resume')
            lineages.append(torch.load(self.checkpoint, weights_only=True)['lineage_id'])
        self.assertTrue(lineages[0].startswith('lineage-'))
        self.assertEqual(*lineages)

    def test_finetune_keeps_parent_lineage_and_step(self):
        self.data.write_text('A different real fine tuning corpus for testing.\n' * 120)
        self.invoke('--finetune')
        after = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(after['stage'], 'finetune')
        self.assertEqual(after['starting_step'], self.saved['step'])
        self.assertEqual(after['lineage_id'], self.saved['lineage_id'])
        self.assertNotEqual(after['run_id'], self.saved['run_id'])
        self.assertEqual(after['step'], 1)


    def test_tokenizer_mismatch_rejected_before_training(self):
        from tokenizer import Tokenizer
        tok = Tokenizer()
        tok.train(b'ababababab', 257)
        tok.save(self.root / 'checkpoints/tokenizer.json')
        before = self.checkpoint.read_bytes()
        with self.assertRaises(SystemExit) as error:
            self.invoke('--resume')
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(before, self.checkpoint.read_bytes())

    def test_merge_identity_guard_with_same_vocabulary(self):
        from tokenizer import Tokenizer
        tok = Tokenizer()
        tok.train(b'ababababab', 257)
        # Construct a consistent tiny checkpoint with a 257-token vocabulary.
        model = train.Brain(train.Config(width=16, layers=1, heads=2, context=8, vocab=257))
        saved = dict(self.saved, config=vars(model.config), model=model.state_dict(),
                     tokenizer_sha256=tok.fingerprint())
        torch.save(saved, self.checkpoint)
        other = Tokenizer()
        other.train(b'cdcdcdcdcd', 257)
        other.save(self.root / 'checkpoints/tokenizer.json')
        with patch.object(train, 'Brain') as constructor:
            with self.assertRaises(SystemExit) as error:
                self.invoke('--resume')
            self.assertEqual(error.exception.code, 2)
            constructor.assert_not_called()

    def test_output_directory_isolation(self):
        before = self.checkpoint.read_bytes()
        alternate = self.root / 'other-run'
        self.invoke('--output-dir', str(alternate), '--width', '16', '--layers', '1',
                    '--heads', '2', '--context', '8', '--vocab', '256')
        self.assertTrue((alternate / 'latest.pt').exists())
        self.assertEqual(before, self.checkpoint.read_bytes())
        self.invoke('--output-dir', str(alternate), '--resume')
        self.assertEqual(torch.load(alternate / 'latest.pt', weights_only=True)['step'], 2)

    def test_nonfinite_gradient_preserves_last_checkpoint(self):
        before = self.checkpoint.read_bytes()
        with patch.object(torch.nn.utils, 'clip_grad_norm_', side_effect=RuntimeError('nonfinite gradient')):
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                self.invoke('--resume')
        self.assertEqual(before, self.checkpoint.read_bytes())
        self.assertEqual(json.loads((self.root / 'checkpoints/status.json').read_text())['status'], 'error')

    def test_legacy_cache_not_reused(self):
        cache_dir = self.root / 'data/.cache'
        for p in cache_dir.glob('*.bin'):
            p.unlink()
        # This was previously accepted solely because corpus and vocab size matched.
        legacy = cache_dir / f"{self.saved['data_sha256'][:16]}-256.bin"
        legacy.write_bytes(b'corrupt legacy data')
        self.invoke('--resume')
        self.assertEqual(legacy.read_bytes(), b'corrupt legacy data')
        self.assertEqual(len(list(cache_dir.glob('v2-*.bin'))), 1)

    def test_response_finetune_and_resume(self):
        self.data.write_text(('### Instruction:\nChoose a color\n\n### Response:\nBlue is a color.\n<|end|>\n\n') * 100)
        self.invoke('--finetune', '--loss-mode', 'response')
        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(saved['loss_mode'], 'response')
        self.assertEqual(saved['stage'], 'finetune')
        self.assertEqual(saved['step'], 1)
        self.invoke('--resume')
        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(saved['loss_mode'], 'response')
        self.assertEqual(saved['step'], 2)
        with self.assertRaises(SystemExit):
            self.invoke('--resume', '--loss-mode', 'all')

    def test_new_finetune_corpus_resets_phase_history(self):
        self.data.write_text('First fine tuning text with novel words.\n' * 130)
        self.invoke('--finetune')
        self.invoke('--resume')
        self.assertEqual(torch.load(self.checkpoint, weights_only=True)['step'], 2)
        self.data.write_text('Second fine tuning text with other words.\n' * 130)
        self.invoke('--finetune')
        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(saved['step'], 1)
        self.assertEqual(saved['starting_step'], 2)
        self.assertEqual(len(saved['history']), 2)

    def test_resume_inherits_learning_rate_unless_explicit(self):
        self.invoke('--resume', '--lr', '0.00001')
        self.invoke('--resume')
        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], 0.00001)
        self.invoke('--resume', '--lr', '0.00002')
        saved = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(saved['optimizer']['param_groups'][0]['lr'], 0.00002)

    def test_init_from_branches_without_modifying_parent(self):
        before = self.checkpoint.read_bytes()
        self.data.write_text(('### Instruction:\nName a color\n\n### Response:\nBlue.\n<|end|>\n\n') * 100)
        destination = self.root/'response-experiment'
        self.invoke('--init-from', str(self.checkpoint.parent), '--output-dir', str(destination),
                    '--loss-mode', 'response')
        self.assertEqual(self.checkpoint.read_bytes(), before)
        child = torch.load(destination/'latest.pt', weights_only=True)
        self.assertEqual(child['parent_checkpoint_sha256'], hashlib.sha256(before).hexdigest())
        self.assertEqual(child['loss_mode'], 'response')
        self.assertEqual(child['stage'], 'finetune')
        self.assertEqual(child['step'], 1)
        self.assertTrue((destination/'tokenizer.json').exists())

    def test_retry_preserves_original_step_budget(self):
        original_step = torch.optim.AdamW.step
        calls = 0
        def interrupted_step(optimizer, *args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 22:
                raise RuntimeError('simulated interrupted chunk')
            return original_step(optimizer, *args, **kwargs)
        with patch.dict(os.environ, {'AVEN_BUDGET_ID': 'test-chunk-1'}):
            with patch.object(torch.optim.AdamW, 'step', interrupted_step):
                with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                    self.invoke('--resume', '--steps', '40')
            saved = torch.load(self.checkpoint, weights_only=True)
            self.assertEqual(saved['step'], 20)
            self.assertEqual(saved['budget_target'], 41)
            self.invoke('--resume', '--steps', '40')
            saved = torch.load(self.checkpoint, weights_only=True)
            self.assertEqual(saved['step'], 41)
            before = {k: v.clone() for k, v in saved['model'].items()}
            # A crash after the final save must not train the entire budget again.
            self.invoke('--resume', '--steps', '40')
            saved = torch.load(self.checkpoint, weights_only=True)
            self.assertEqual(saved['step'], 41)
            self.assertTrue(all(torch.equal(value, saved['model'][key]) for key, value in before.items()))
        with patch.dict(os.environ, {'AVEN_BUDGET_ID': 'test-chunk-2'}):
            self.invoke('--resume', '--steps', '5')
            self.assertEqual(torch.load(self.checkpoint, weights_only=True)['step'], 46)

    def test_cpu_dropout_resume_matches_uninterrupted_updates(self):
        self.invoke('--resume', '--dropout', '0.2')
        start = self.checkpoint.read_bytes()
        self.invoke('--resume', '--steps', '4')
        uninterrupted = torch.load(self.checkpoint, weights_only=True)
        self.checkpoint.write_bytes(start)
        self.invoke('--resume', '--steps', '2')
        self.invoke('--resume', '--steps', '2')
        resumed = torch.load(self.checkpoint, weights_only=True)
        self.assertEqual(uninterrupted['step'], resumed['step'])
        self.assertTrue(all(torch.equal(value, resumed['model'][key]) for key, value in uninterrupted['model'].items()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
