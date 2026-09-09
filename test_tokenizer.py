"""Regression coverage for the O(n log n) heap-based encode_ids -- must
produce byte-identical output to the original sequential full-pass
algorithm (kept here only as a reference for comparison, not in
tokenizer.py) at any corpus size."""
import unittest
from tokenizer import Tokenizer


def naive_encode_ids(tok, raw_bytes):
    ids = list(raw_bytes)
    for pair, idx in sorted(tok.merges.items(), key=lambda kv: kv[1]):
        ids = tok._merge(ids, pair, idx)
    return ids


class TokenizerEncodeTests(unittest.TestCase):
    def setUp(self):
        text = (b'the quick brown fox jumps over the lazy dog. ' * 200
                + b'she sells seashells by the seashore. ' * 200)
        self.tok = Tokenizer()
        self.tok.train(text, 320)
        self.text = text

    def test_matches_naive_reference_on_training_text(self):
        self.assertEqual(naive_encode_ids(self.tok, self.text), self.tok.encode_ids(self.text))

    def test_matches_naive_reference_on_unseen_text(self):
        unseen = b'a completely different sentence the tokenizer never trained on!'
        self.assertEqual(naive_encode_ids(self.tok, unseen), self.tok.encode_ids(unseen))

    def test_empty_and_tiny_inputs(self):
        for sample in (b'', b'a', b'ab', b'aaa'):
            self.assertEqual(naive_encode_ids(self.tok, sample), self.tok.encode_ids(sample))

    def test_decode_round_trips(self):
        ids = self.tok.encode_ids(self.text)
        self.assertEqual(self.tok.decode(ids), self.text.decode('utf-8'))

    def test_faster_than_naive_at_this_scale(self):
        import time
        t0 = time.time(); naive_encode_ids(self.tok, self.text); t_naive = time.time() - t0
        t0 = time.time(); self.tok.encode_ids(self.text); t_fast = time.time() - t0
        self.assertLess(t_fast, t_naive)


if __name__ == '__main__':
    unittest.main()
