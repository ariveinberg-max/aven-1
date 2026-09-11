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
import train_reward
from artifact_io import file_sha256, validate_tokenizer


class RewardTrainingTests(unittest.TestCase):
    def test_tiny_training_preserves_policy_and_saves_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'checkpoints'
            out.mkdir()
            tok = Tokenizer()
            tok.save(out/'tokenizer.json')
            model = Brain(Config(width=16, layers=1, heads=2, context=128))
            torch.save(dict(config=vars(model.config), model=model.state_dict(), stage='finetune', tokenizer_sha256=tok.fingerprint()), out/'latest.pt')
            before = file_sha256(out/'latest.pt')
            rows = [dict(prompt=f'Choose {i}', response_a='Good.', response_b='Bad.', winner='a') for i in range(8)]
            with patch.object(train_reward, 'ROOT', root), patch.object(train_reward.preferences, 'all_decided', return_value=rows), contextlib.redirect_stdout(io.StringIO()):
                train_reward.train_reward(SimpleNamespace(steps=2, lr=0.001, val_fraction=0.25))
            reward = torch.load(out/'reward.pt', weights_only=True)
            self.assertEqual(reward['parent_checkpoint_sha256'], before)
            self.assertEqual(reward['tokenizer_sha256'], tok.fingerprint())
            self.assertEqual(reward['data_report']['val_prompt_count'], 2)
            self.assertEqual(reward['data_report']['train_prompt_count'], 6)
            self.assertEqual(file_sha256(out/'latest.pt'), before)
            self.assertTrue(0 <= reward['heldout_accuracy'] <= 1)

    def test_preference_context_is_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            train_reward.encode_pair(Tokenizer(), 'a' * 200, 'answer', 64)

    def test_downstream_tokenizer_guard(self):
        tok = Tokenizer()
        with self.assertRaisesRegex(ValueError, 'merge rules'):
            validate_tokenizer(dict(config={'vocab': 256}, tokenizer_sha256='wrong'), tok, 'Reward')
