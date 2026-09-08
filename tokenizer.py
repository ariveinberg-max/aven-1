"""Byte-pair encoding, trained from scratch on your own corpus.

No pretrained vocabulary is downloaded — this is the same algorithm GPT-style
models use (iteratively merge the most frequent adjacent byte pair), trained
here on whatever text you compile into the corpus. A trained tokenizer is
tied to the corpus it learned from and is saved next to the checkpoint that
uses it.
"""
import json


class Tokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @staticmethod
    def _stats(ids):
        counts = {}
        for a, b in zip(ids, ids[1:]):
            counts[(a, b)] = counts.get((a, b), 0) + 1
        return counts

    @staticmethod
    def _merge(ids, pair, idx):
        out = []
        i = 0
        while i < len(ids):
            if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
                out.append(idx)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def train(self, raw_bytes, vocab_size):
        if vocab_size < 256:
            raise ValueError('vocab_size must be at least 256 (the raw bytes).')
        ids = list(raw_bytes)
        merges = {}
        vocab = {i: bytes([i]) for i in range(256)}
        for i in range(vocab_size - 256):
            stats = self._stats(ids)
            if not stats:
                break
            pair = max(stats, key=stats.get)
            if stats[pair] < 2:
                break
            idx = 256 + i
            ids = self._merge(ids, pair, idx)
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
        self.merges = merges
        self.vocab = vocab

    def encode(self, text):
        ids = list(text.encode('utf-8'))
        while len(ids) >= 2:
            stats = self._stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            ids = self._merge(ids, pair, self.merges[pair])
        return ids

    def encode_ids(self, raw_bytes):
        """Encode already-decoded bytes (e.g. a whole corpus) without a utf-8 round trip."""
        ids = list(raw_bytes)
        for pair, idx in sorted(self.merges.items(), key=lambda kv: kv[1]):
            ids = self._merge(ids, pair, idx)
        return ids

    def decode(self, ids):
        return b''.join(self.vocab[i] for i in ids).decode('utf-8', errors='replace')

    def save(self, path):
        ordered = sorted(self.merges.items(), key=lambda kv: kv[1])
        path.write_text(json.dumps({'merges': [[a, b] for (a, b), _ in ordered]}), encoding='utf-8')

    def load(self, path):
        data = json.loads(path.read_text(encoding='utf-8'))
        vocab = {i: bytes([i]) for i in range(256)}
        merges = {}
        idx = 256
        for a, b in data['merges']:
            merges[(a, b)] = idx
            vocab[idx] = vocab[a] + vocab[b]
            idx += 1
        self.merges = merges
        self.vocab = vocab
        return self
