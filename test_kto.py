import contextlib
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from brain import Brain, Config
from tokenizer import Tokenizer
import kto
from artifact_io import file_sha256, validate_tokenizer


class KTOTests(unittest.TestCase):
    def test_derive_kto_examples_splits_each_pair_into_one_of_each_label(self):
        rows = [dict(prompt='p', response_a='good', response_b='bad', winner='a')]
        examples = kto.derive_kto_examples(rows)
        self.assertEqual(len(examples), 2)
        labels = {e['label'] for e in examples}
        self.assertEqual(labels, {'desirable', 'undesirable'})
        desirable = next(e for e in examples if e['label'] == 'desirable')
        undesirable = next(e for e in examples if e['label'] == 'undesirable')
        self.assertEqual(desirable['response'], 'good')
        self.assertEqual(undesirable['response'], 'bad')

    def test_tiny_training_preserves_policy_checkpoint_and_saves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'checkpoints'
            out.mkdir()
            tok = Tokenizer()
            tok.save(out/'tokenizer.json')
            model = Brain(Config(width=16, layers=1, heads=2, context=128))
            torch.save(dict(config=vars(model.config), model=model.state_dict(), stage='finetune',
                             tokenizer_sha256=tok.fingerprint(), data_sha256='abc', dataset='data/instructions.txt'),
                       out/'latest.pt')
            before = file_sha256(out/'latest.pt')
            rows = [dict(prompt=f'Choose {i}', response_a='Good answer.', response_b='Bad answer.', winner='a')
                    for i in range(10)]
            with patch.object(kto, 'ROOT', root), patch.object(kto.preferences, 'all_decided', return_value=rows), \
                    contextlib.redirect_stdout(io.StringIO()):
                kto.train_kto(SimpleNamespace(steps=2, lr=0.0001, beta=0.1, batch_size=4, val_fraction=0.25))
            result = torch.load(out/'kto_policy.pt', weights_only=True)
            self.assertEqual(result['parent_checkpoint_sha256'], before)
            self.assertEqual(result['tokenizer_sha256'], tok.fingerprint())
            self.assertEqual(result['stage'], 'kto')
            self.assertTrue(result['train_count'] > 0)
            self.assertTrue(result['val_count'] > 0)
            # Base policy checkpoint must be untouched -- same rule as DPO/PPO.
            self.assertEqual(file_sha256(out/'latest.pt'), before)

    def test_rejects_non_finetuned_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'checkpoints'
            out.mkdir()
            tok = Tokenizer()
            tok.save(out/'tokenizer.json')
            model = Brain(Config(width=16, layers=1, heads=2, context=128))
            torch.save(dict(config=vars(model.config), model=model.state_dict(), stage='pretrain',
                             tokenizer_sha256=tok.fingerprint()), out/'latest.pt')
            with patch.object(kto, 'ROOT', root), self.assertRaises(SystemExit):
                kto.train_kto(SimpleNamespace(steps=2, lr=0.0001, beta=0.1, batch_size=4, val_fraction=0.25))

    def test_downstream_tokenizer_guard(self):
        tok = Tokenizer()
        with self.assertRaisesRegex(ValueError, 'merge rules'):
            validate_tokenizer(dict(config={'vocab': 256}, tokenizer_sha256='wrong'), tok, 'Policy')

    def test_too_few_examples_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'checkpoints'
            out.mkdir()
            tok = Tokenizer()
            tok.save(out/'tokenizer.json')
            model = Brain(Config(width=16, layers=1, heads=2, context=128))
            torch.save(dict(config=vars(model.config), model=model.state_dict(), stage='finetune',
                             tokenizer_sha256=tok.fingerprint()), out/'latest.pt')
            # 4 distinct-prompt comparisons clears split_comparisons' own minimum
            # (>=4 comparisons, >=2 prompts), but after x2 derivation and an
            # 0.25 val split, train lands under this test's batch_size=8 --
            # exercising THIS script's own "label more" guard, not the earlier one.
            rows = [dict(prompt=f'p{i}', response_a='a', response_b='b', winner='a') for i in range(4)]
            with patch.object(kto, 'ROOT', root), patch.object(kto.preferences, 'all_decided', return_value=rows), \
                    self.assertRaisesRegex(SystemExit, 'Label more comparisons'):
                kto.train_kto(SimpleNamespace(steps=2, lr=0.0001, beta=0.1, batch_size=8, val_fraction=0.25))


if __name__ == '__main__':
    unittest.main()
