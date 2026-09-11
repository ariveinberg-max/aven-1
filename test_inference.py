import unittest
from unittest.mock import patch
import torch
from brain import Brain, Config
from inference import cached_logits


class InferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        self.model = Brain(Config(width=32, layers=2, heads=4, context=16)).eval()

    def test_cached_logits_match_full_forward(self):
        ids = torch.randint(0, 256, (2, 16))
        for scale in [None, 0.3]:
            for block in self.model.blocks:
                block.attn_scale = scale
            logits, cache = cached_logits(self.model, ids[:, :5])
            torch.testing.assert_close(logits, self.model(ids[:, :5])[0][:, -1:], atol=1e-6, rtol=1e-5)
            for end in range(6, 17):
                logits, cache = cached_logits(self.model, ids[:, end-1:end], cache)
                torch.testing.assert_close(logits, self.model(ids[:, :end])[0][:, -1:], atol=1e-6, rtol=1e-5)

    def test_greedy_generation_matches_across_rollover(self):
        original = {k: v.clone() for k, v in self.model.state_dict().items()}
        for prompt in ['abc', 'a' * 16, 'a' * 30, '🌱']:
            cached = self.model.generate(prompt, count=24, temperature=0, use_cache=True)
            plain = self.model.generate(prompt, count=24, temperature=0, use_cache=False)
            self.assertEqual(cached[0], plain[0])
            self.assertEqual(cached[1], plain[1])
        for key, value in self.model.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_marker_inside_token_and_prompt(self):
        class Tokenizer:
            def encode(self, text): return [65]
            def decode(self, ids): return 'answer<|end|>extra' if ids else ''
        prompt = 'Literal <|end|> in prompt: '
        text, _ = self.model.generate(prompt, 5, 0, tokenizer=Tokenizer(), stop_text='<|end|>')
        self.assertEqual(text, prompt + 'answer')

    def test_invalid_settings_and_cache_mode(self):
        for temperature in [-1, float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                self.model.generate('x', temperature=temperature)
        self.model.train()
        with self.assertRaises(ValueError):
            cached_logits(self.model, torch.tensor([[1]]))
