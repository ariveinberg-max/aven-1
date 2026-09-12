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
import dpo
from artifact_io import file_sha256, validate_tokenizer


class DPOTrainingTests(unittest.TestCase):
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
                    for i in range(8)]
            with patch.object(dpo, 'ROOT', root), patch.object(dpo.preferences, 'all_decided', return_value=rows), \
                    contextlib.redirect_stdout(io.StringIO()):
                dpo.train_dpo(SimpleNamespace(steps=2, lr=0.0001, beta=0.1, val_fraction=0.25))
            result = torch.load(out/'dpo_policy.pt', weights_only=True)
            self.assertEqual(result['parent_checkpoint_sha256'], before)
            self.assertEqual(result['tokenizer_sha256'], tok.fingerprint())
            self.assertEqual(result['stage'], 'dpo')
            self.assertEqual(result['data_report']['val_prompt_count'], 2)
            self.assertEqual(result['data_report']['train_prompt_count'], 6)
            self.assertTrue(0 <= result['heldout_accuracy'] <= 1)
            # The base policy checkpoint must be untouched -- DPO's output is a
            # separate file, never auto-promoted (same rule as PPO's output).
            self.assertEqual(file_sha256(out/'latest.pt'), before)
            self.assertFalse((out/'latest.pt').read_bytes() == b'')

    def test_preference_context_is_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            dpo.encode_pair(Tokenizer(), 'a' * 200, 'answer', 64)

    def test_response_logprob_only_scores_response_tokens(self):
        tok = Tokenizer()
        model = Brain(Config(width=16, layers=1, heads=2, context=128, vocab=tok.vocab_size))
        ids, prefix_len = dpo.encode_pair(tok, 'hi', 'ok', 128)
        self.assertGreater(prefix_len, 0)
        self.assertLess(prefix_len, ids.shape[1])
        logprob = dpo.response_logprob(model, ids, prefix_len)
        self.assertTrue(torch.isfinite(logprob))
        self.assertEqual(logprob.numel(), 1)

    def test_downstream_tokenizer_guard(self):
        tok = Tokenizer()
        with self.assertRaisesRegex(ValueError, 'merge rules'):
            validate_tokenizer(dict(config={'vocab': 256}, tokenizer_sha256='wrong'), tok, 'Policy')

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
            with patch.object(dpo, 'ROOT', root), self.assertRaises(SystemExit):
                dpo.train_dpo(SimpleNamespace(steps=2, lr=0.0001, beta=0.1, val_fraction=0.25))


if __name__ == '__main__':
    unittest.main()
