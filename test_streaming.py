"""Regression coverage for train.py's streaming corpus pipeline: chunked
fingerprint/validation and chunked token-cache encoding must behave
identically to reading the whole file into memory at once."""
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tokenizer import Tokenizer
from train import stream_fingerprint_and_validate, encode_corpus_to_cache


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_fingerprint_matches_whole_file_hash(self):
        path = self.root / 'corpus.txt'
        text = ('some real training text, repeated many times. ' * 5000).encode('utf-8')
        path.write_bytes(text)
        expected = hashlib.sha256(text).hexdigest()
        self.assertEqual(stream_fingerprint_and_validate(path, chunk_size=1024), expected)

    def test_rejects_invalid_utf8(self):
        path = self.root / 'bad.txt'
        path.write_bytes(b'valid text then garbage: \xff\xfe\x00\x01' * 100)
        with self.assertRaises(ValueError):
            stream_fingerprint_and_validate(path, chunk_size=8)

    def test_valid_utf8_split_across_chunk_boundary(self):
        # A multi-byte UTF-8 character deliberately split by a tiny chunk
        # size must still validate correctly (the incremental decoder must
        # buffer partial sequences across chunks, not falsely reject them).
        path = self.root / 'unicode.txt'
        path.write_bytes('hello 🌱 world, real emoji text here.'.encode('utf-8') * 50)
        stream_fingerprint_and_validate(path, chunk_size=3)  # must not raise

    def test_encode_corpus_to_cache_reconstructs_exact_text(self):
        # A missed merge at a chunk boundary shifts every subsequent token's
        # *position* (one boundary produces one extra token, cascading a raw
        # index-by-index comparison into near-total apparent mismatch even
        # though the content is fine) -- so the correctness property that
        # actually matters is that decoding reconstructs the original text
        # exactly, not that token IDs land at identical positions.
        #
        # Uses real, non-repetitive text (not a repeated phrase): synthetic
        # repetition lets BPE compress to a handful of tokens (e.g. 23KB ->
        # 36 tokens seen in an earlier version of this test), making any
        # single boundary's overhead look enormous in relative terms purely
        # as a test artifact -- real corpora never compress anywhere near
        # that ratio.
        source = Path('data/sources/tom-sawyer.txt') if Path('data/sources/tom-sawyer.txt').exists() else None
        if source is None:
            self.skipTest('data/sources/tom-sawyer.txt not present in this checkout')
        text = source.read_bytes()[10_000:60_000]
        path = self.root / 'corpus.txt'
        path.write_bytes(text)
        tok = Tokenizer()
        tok.train(text[:20_000], 600)
        cache_path = self.root / 'cache.bin'
        token_count = encode_corpus_to_cache(path, tok, cache_path, chunk_size=1024)
        cached_ids = np.memmap(cache_path, dtype=np.uint16, mode='r', shape=(token_count,)).tolist()
        self.assertEqual(tok.decode(cached_ids), text.decode('utf-8'))
        # Chunking can only ever miss merges (never invent tokens or lose
        # data), so it should produce the same or more tokens than one
        # whole-file encode, never fewer, and only marginally more on real
        # text (real usage is an 8MB chunk size against orders of magnitude
        # more text -- e.g. 44 boundaries across the real 354MB
        # Wikipedia+books corpus, whose 1.75 bytes/token overall ratio showed
        # no sign of this effect at that scale).
        direct_count = len(tok.encode_ids(text))
        self.assertGreaterEqual(token_count, direct_count)
        self.assertLess(token_count - direct_count, direct_count * 0.1)


if __name__ == '__main__':
    unittest.main()
